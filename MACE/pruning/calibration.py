import ase.io
import torch
from tqdm import tqdm
from mace import data
from mace.tools import torch_geometric
from mace.tools.utils import AtomicNumberTable


def collect_taylor(model, structures, device):
    print(f"Counting calibration structures: {structures}", flush=True)
    total = sum(1 for _ in ase.io.iread(structures, index=':'))
    if total == 0:
        raise ValueError('The calibration dataset contains no structures.')
    samples = {}

    def hook(name):
        def forward_hook(module, inputs, output):
            activation = output.detach()

            def backward_hook(gradient):
                score = (gradient.detach() * activation).sum(dim=0)
                samples.setdefault(name, []).append(torch.nan_to_num(score, nan=0.0).cpu())

            output.register_hook(backward_hook)
        return forward_hook

    names = ['node_embedding']
    for i in range(2):
        names.extend(f'interactions.{i}.{name}' for name in ('linear_up', 'conv_tp', 'linear', 'skip_tp'))
        names.extend(f'products.{i}.{name}' for name in ('symmetric_contractions', 'linear'))
    handles = [model.get_submodule(name).register_forward_hook(hook(name)) for name in names]
    model.to(device)
    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    count = 0
    print(f"Calibrating on {device}: {structures}", flush=True)
    progress = tqdm(total=total, desc="Calibration", unit="structure", dynamic_ncols=True)
    try:
        for atoms in ase.io.iread(structures, index=':'):
            graph = data.AtomicData.from_config(
                data.config_from_atoms(atoms), z_table=z_table,
                cutoff=float(model.r_max), heads=getattr(model, 'heads', None),
            )
            batch = torch_geometric.batch.Batch.from_data_list([graph]).to(device)
            model.zero_grad(set_to_none=True)
            result = model(batch.to_dict(), compute_force=False, compute_stress=False)
            result['energy'].sum().backward()
            count += 1
            progress.update(1)
    finally:
        progress.close()
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        model.cpu()
    if not samples:
        raise ValueError('The calibration dataset contains no structures.')
    print(f"Calibration complete: {count} structures", flush=True)
    return samples


def compute_taylor_score(samples_list, irreps_str):
    from e3nn import o3

    irreps = o3.Irreps(irreps_str)
    num_channels = irreps[0].mul

    stacked = torch.stack(samples_list, dim=0)

    scores_per_sample = []
    for sample in stacked:
        channel_scores = torch.zeros(num_channels)
        for irrep_slice in irreps.slices():
            irrep_data = sample[irrep_slice].reshape(num_channels, -1).sum(dim=1).abs()
            channel_scores += irrep_data
        scores_per_sample.append(channel_scores)

    return torch.stack(scores_per_sample, dim=0).mean(dim=0)


def score_irreps(config):
    """Feature layouts used by the three supported source checkpoints."""
    from e3nn import o3
    from mace.modules.irreps_tools import tp_out_irreps_with_instructions

    hidden = config["hidden_irreps"]
    channels = {irrep.mul for irrep in hidden}
    if len(channels) != 1 or (hidden.lmax, hidden[0].mul) not in ((1, 256), (2, 128), (2, 224)):
        raise ValueError("Calibration supports MP L1 C256, MP L2 C128 and OFF L2 C224 source models.")
    width = hidden[0].mul
    scalar = f"{width}x0e"
    angular = "+".join(f"{width}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(4))
    intermediate, _ = tp_out_irreps_with_instructions(
        hidden, o3.Irreps("1x0e+1x1o+1x2e+1x3o"), o3.Irreps(angular),
    )
    return {
        "node_embedding": scalar,
        "interactions.0.linear_up": scalar,
        "interactions.0.conv_tp": angular,
        "interactions.0.linear": angular,
        "interactions.0.skip_tp": angular if width == 224 else str(hidden),
        "products.0.symmetric_contractions": str(hidden),
        "products.0.linear": str(hidden),
        "interactions.1.linear_up": str(hidden),
        "interactions.1.conv_tp": str(intermediate),
        "interactions.1.linear": str(intermediate),
        "interactions.1.skip_tp": scalar,
        "products.1.symmetric_contractions": scalar,
        "products.1.linear": scalar,
    }


def model_structure(model, config):
    return {
        "hidden_irreps": str(config["hidden_irreps"]),
        "atomic_numbers": [int(z) for z in model.atomic_numbers],
        "r_max": float(model.r_max),
        "num_interactions": len(model.interactions),
    }


def load_scores(path, model, config):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["structure"] != model_structure(model, config):
        raise ValueError("Calibration scores do not match the source model structure.")
    scores = payload["scores"]
    for name, irreps in score_irreps(config).items():
        channels = int(irreps.split("x", 1)[0])
        if name not in scores or scores[name].shape != (channels,):
            raise ValueError(f"Invalid calibration scores for {name}.")
        if not torch.isfinite(scores[name]).all():
            raise ValueError(f"Non-finite calibration scores for {name}.")
    return scores


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Compute MACE channel importance scores.")
    parser.add_argument("--model", required=True, help="Source MACE checkpoint")
    parser.add_argument("--data", required=True, help="Calibration structures in ASE format")
    parser.add_argument("--output", required=True, help="Output score file")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from model_building import build_model

    torch.manual_seed(42)
    config, model = build_model(args.model)
    layouts = score_irreps(config)
    samples = collect_taylor(model, args.data, args.device)
    scores = {name: compute_taylor_score(samples[name], irreps)
              for name, irreps in layouts.items()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "source_model": args.model,
        "structure": model_structure(model, config),
        "scores": scores,
    }, output)
    print(f"Saved scores: {output} ({len(scores)} layers)", flush=True)


if __name__ == "__main__":
    main()

