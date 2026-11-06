import argparse
from pathlib import Path

import torch


def prune(model, model_config_foundation, taylor=None, *, target_l=1, target_channels=128):
    from e3nn import o3

    hidden = model_config_foundation['hidden_irreps']
    if {irrep.mul for irrep in hidden} != {224} or hidden.lmax != 2:
        raise ValueError("Supported source: MACE-OFF L2 C224.")
    if not isinstance(target_channels, int) or not 1 <= target_channels <= 224:
        raise ValueError("Target channels must be an integer from 1 to 224.")
    if not isinstance(target_l, int) or not 0 <= target_l <= 2:
        raise ValueError("Target L must be an integer from 0 to 2.")
    if target_l == 2 and target_channels == 224:
        return model


    def select_channels(*names):
        if target_channels == 224:
            return torch.arange(224)
        if taylor is None:
            raise ValueError("Reducing channel width requires calibration scores.")
        score = taylor[names[0]]
        for name in names[1:]:
            score = score + taylor[name]
        return torch.sort(torch.topk(score, target_channels).indices).values

    indices_sorted_1 = select_channels("node_embedding")

    linear = o3.Linear(
        o3.Irreps("10x0e"),
        o3.Irreps(f"{target_channels}x0e"),
        internal_weights=True,
        shared_weights=True,
    )
    with torch.no_grad():
        linear.weight.copy_(model.node_embedding.linear.weight.view(10, 224)[:, indices_sorted_1].reshape(-1))
    model.node_embedding.linear = linear

    indices_sorted_2 = select_channels("interactions.0.conv_tp")

    linear_up = o3.Linear(
        o3.Irreps(f"{target_channels}x0e"),
        o3.Irreps(f"{target_channels}x0e"),
        internal_weights=True,
        shared_weights=True,
    )
    with torch.no_grad():
        linear_up.weight.copy_(model.interactions[0].linear_up.weight.view(224, 224)[indices_sorted_1, :][:, indices_sorted_2].reshape(-1))
    model.interactions[0].linear_up = linear_up

    from e3nn import nn
    from e3nn.util import prod

    conv_tp_weights = nn.FullyConnectedNet(
        [8, 64, 64, 64, 4 * target_channels],
        torch.nn.functional.silu,
    )

    weight_segments = []
    flat_weight_index = 0
    for instruction in model.interactions[0].conv_tp.instructions:
        weight_segments.append(model.interactions[0].conv_tp_weights.layer3.weight[:, flat_weight_index:flat_weight_index + 224][:, indices_sorted_2])
        flat_weight_index += prod(instruction.path_shape)
    weight = torch.cat(weight_segments, dim=1)

    with torch.no_grad():
        conv_tp_weights.layer0.weight.copy_(model.interactions[0].conv_tp_weights.layer0.weight)
        conv_tp_weights.layer1.weight.copy_(model.interactions[0].conv_tp_weights.layer1.weight)
        conv_tp_weights.layer2.weight.copy_(model.interactions[0].conv_tp_weights.layer2.weight)
        conv_tp_weights.layer3.weight.copy_(weight)
    model.interactions[0].conv_tp_weights = conv_tp_weights

    from mace.modules.irreps_tools import tp_out_irreps_with_instructions

    irreps_mid, instructions = tp_out_irreps_with_instructions(
        o3.Irreps(f"{target_channels}x0e"),
        o3.Irreps("1x0e+1x1o+1x2e+1x3o"),
        o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e+{target_channels}x3o"),
    )

    conv_tp = o3.TensorProduct(
        o3.Irreps(f"{target_channels}x0e"),
        o3.Irreps("1x0e+1x1o+1x2e+1x3o"),
        irreps_mid,
        instructions=instructions,
        shared_weights=False,
        internal_weights=False,
    )

    model.interactions[0].conv_tp = conv_tp

    indices_sorted_3 = select_channels("interactions.0.linear")

    weight_segments = []
    flat_weight_index = 0
    for instruction in model.interactions[0].linear.instructions:
        weight_segments.append(model.interactions[0].linear.weight[flat_weight_index:flat_weight_index + prod(instruction.path_shape)].view(224, 224)[indices_sorted_2, :][:, indices_sorted_3].reshape(-1))
        flat_weight_index += prod(instruction.path_shape)
    weight = torch.cat(weight_segments, dim=0)

    linear = o3.Linear(
        o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e+{target_channels}x3o"),
        o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e+{target_channels}x3o"),
        internal_weights=True,
        shared_weights=True,
    )
    with torch.no_grad():
        linear.weight.copy_(weight)
    model.interactions[0].linear = linear

    indices_sorted_4 = select_channels("products.0.symmetric_contractions")

    weight_segments = []
    flat_weight_index = 0
    for instruction in model.interactions[0].skip_tp.instructions:
        weight_segments.append(model.interactions[0].skip_tp.weight[flat_weight_index:flat_weight_index + prod(instruction.path_shape)].view(224, 10, 224)[indices_sorted_3, :, :][:, :, indices_sorted_4].reshape(-1))
        flat_weight_index += prod(instruction.path_shape)
    weight = torch.cat(weight_segments, dim=0)

    skip_tp = o3.FullyConnectedTensorProduct(
        o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e+{target_channels}x3o"),
        o3.Irreps("10x0e"),
        o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e+{target_channels}x3o"),
        shared_weights=True,
        internal_weights=True,
    )
    with torch.no_grad():
        skip_tp.weight.copy_(weight)
    model.interactions[0].skip_tp = skip_tp

    from mace.modules.irreps_tools import reshape_irreps

    model.interactions[0].reshape = reshape_irreps(o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e+{target_channels}x3o"), cueq_config=model_config_foundation["cueq_config"])

    from mace.modules.symmetric_contraction import SymmetricContraction

    symmetric_contractions = SymmetricContraction(
        irreps_in=o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e+{target_channels}x3o"),
        irreps_out=o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e"),
        correlation=model_config_foundation['correlation'],
        num_elements=model_config_foundation['num_elements'],
        use_reduced_cg=model_config_foundation['use_reduced_cg'],
    )
    with torch.no_grad():
        symmetric_contractions.contractions[0].weights_max.copy_(
            model.products[0].symmetric_contractions.contractions[0].weights_max[:, :, indices_sorted_4]
        )
        symmetric_contractions.contractions[0].weights[0].copy_(
            model.products[0].symmetric_contractions.contractions[0].weights[0][:, :, indices_sorted_4]
        )
        symmetric_contractions.contractions[0].weights[1].copy_(
            model.products[0].symmetric_contractions.contractions[0].weights[1][:, :, indices_sorted_4]
        )
        symmetric_contractions.contractions[1].weights_max.copy_(
            model.products[0].symmetric_contractions.contractions[1].weights_max[:, :, indices_sorted_4]
        )
        symmetric_contractions.contractions[1].weights[0].copy_(
            model.products[0].symmetric_contractions.contractions[1].weights[0][:, :, indices_sorted_4]
        )
        symmetric_contractions.contractions[1].weights[1].copy_(
            model.products[0].symmetric_contractions.contractions[1].weights[1][:, :, indices_sorted_4]
        )
        symmetric_contractions.contractions[2].weights_max.copy_(
            model.products[0].symmetric_contractions.contractions[2].weights_max[:, :, indices_sorted_4]
        )
        symmetric_contractions.contractions[2].weights[0].copy_(
            model.products[0].symmetric_contractions.contractions[2].weights[0][:, :, indices_sorted_4]
        )
        symmetric_contractions.contractions[2].weights[1].copy_(
            model.products[0].symmetric_contractions.contractions[2].weights[1][:, :, indices_sorted_4]
        )
    model.products[0].symmetric_contractions = symmetric_contractions

    indices_sorted_5 = select_channels("products.0.linear")

    weight_segments = []
    flat_weight_index = 0
    for instruction in model.products[0].linear.instructions:
        weight_segments.append(model.products[0].linear.weight[flat_weight_index:flat_weight_index + prod(instruction.path_shape)].view(224, 224)[indices_sorted_4, :][:, indices_sorted_5].reshape(-1))
        flat_weight_index += prod(instruction.path_shape)
    weight = torch.cat(weight_segments, dim=0)

    linear = o3.Linear(
        o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e"),
        o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e"),
        internal_weights=True,
        shared_weights=True,
    )
    with torch.no_grad():
        linear.weight.copy_(weight)
    model.products[0].linear = linear

    linear = o3.Linear(
        o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e"),
        o3.Irreps("1x0e"),
        internal_weights=True,
        shared_weights=True,
    )
    with torch.no_grad():
        linear.weight.copy_(model.readouts[0].linear.weight[indices_sorted_5])
    model.readouts[0].linear = linear

    indices_sorted_6 = select_channels("interactions.1.conv_tp")

    weight_segments = []
    flat_weight_index = 0
    for instruction in model.interactions[1].linear_up.instructions:
        weight_segments.append(model.interactions[1].linear_up.weight[flat_weight_index:flat_weight_index + prod(instruction.path_shape)].view(224, 224)[indices_sorted_5, :][:, indices_sorted_6].reshape(-1))
        flat_weight_index += prod(instruction.path_shape)
    weight = torch.cat(weight_segments, dim=0)

    linear_up = o3.Linear(
        o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e"),
        o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e"),
        internal_weights=True,
        shared_weights=True,
    )
    with torch.no_grad():
        linear_up.weight.copy_(weight)
    model.interactions[1].linear_up = linear_up

    from e3nn import nn

    conv_tp_weights = nn.FullyConnectedNet(
        [8, 64, 64, 64, 17 * target_channels],
        torch.nn.functional.silu,
    )

    weight_segments = []
    flat_weight_index = 0
    for instruction in model.interactions[1].conv_tp.instructions:
        weight_segments.append(model.interactions[1].conv_tp_weights.layer3.weight[:, flat_weight_index:flat_weight_index + 224][:, indices_sorted_6])
        flat_weight_index += prod(instruction.path_shape)
    weight = torch.cat(weight_segments, dim=1)

    with torch.no_grad():
        conv_tp_weights.layer0.weight.copy_(model.interactions[1].conv_tp_weights.layer0.weight)
        conv_tp_weights.layer1.weight.copy_(model.interactions[1].conv_tp_weights.layer1.weight)
        conv_tp_weights.layer2.weight.copy_(model.interactions[1].conv_tp_weights.layer2.weight)
        conv_tp_weights.layer3.weight.copy_(weight)
    model.interactions[1].conv_tp_weights = conv_tp_weights

    from mace.modules.irreps_tools import tp_out_irreps_with_instructions

    irreps_mid, instructions = tp_out_irreps_with_instructions(
        o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e"),
        o3.Irreps("1x0e+1x1o+1x2e+1x3o"),
        o3.Irreps(f"{target_channels}x0e+{target_channels}x0e+{target_channels}x0e+{target_channels}x1o+{target_channels}x1o+{target_channels}x1o+{target_channels}x1o+{target_channels}x1o+{target_channels}x2e+{target_channels}x2e+{target_channels}x2e+{target_channels}x2e+{target_channels}x2e+{target_channels}x3o+{target_channels}x3o+{target_channels}x3o+{target_channels}x3o"),
    )

    conv_tp = o3.TensorProduct(
        o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e"),
        o3.Irreps("1x0e+1x1o+1x2e+1x3o"),
        irreps_mid,
        instructions=instructions,
        shared_weights=False,
        internal_weights=False,
    )

    model.interactions[1].conv_tp = conv_tp

    indices_sorted_7 = select_channels("products.1.symmetric_contractions")

    weight_segments = []
    flat_weight_index = 0
    for instruction in model.interactions[1].linear.instructions:
        weight_segments.append(model.interactions[1].linear.weight[flat_weight_index:flat_weight_index + prod(instruction.path_shape)].view(224, 224)[indices_sorted_6, :][:, indices_sorted_7].reshape(-1))
        flat_weight_index += prod(instruction.path_shape)
    weight = torch.cat(weight_segments, dim=0)

    linear = o3.Linear(
        o3.Irreps(f"{target_channels}x0e+{target_channels}x0e+{target_channels}x0e+{target_channels}x1o+{target_channels}x1o+{target_channels}x1o+{target_channels}x1o+{target_channels}x1o+{target_channels}x2e+{target_channels}x2e+{target_channels}x2e+{target_channels}x2e+{target_channels}x2e+{target_channels}x3o+{target_channels}x3o+{target_channels}x3o+{target_channels}x3o"),
        o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e+{target_channels}x3o"),
        internal_weights=True,
        shared_weights=True,
    )
    with torch.no_grad():
        linear.weight.copy_(weight)
    model.interactions[1].linear = linear

    indices_sorted_8 = select_channels("interactions.1.skip_tp", "products.1.linear")

    skip_tp = o3.FullyConnectedTensorProduct(
        o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e"),
        o3.Irreps("10x0e"),
        o3.Irreps(f"{target_channels}x0e"),
        shared_weights=True,
        internal_weights=True,
    )
    with torch.no_grad():
        skip_tp.weight.copy_(model.interactions[1].skip_tp.weight.view(224, 10, 224)[indices_sorted_5, :, :][:, :, indices_sorted_8].reshape(-1))
    model.interactions[1].skip_tp = skip_tp

    model.interactions[1].reshape = reshape_irreps(o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e+{target_channels}x3o"), cueq_config=model_config_foundation["cueq_config"])

    from mace.modules.symmetric_contraction import SymmetricContraction

    symmetric_contractions = SymmetricContraction(
        irreps_in=o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e+{target_channels}x3o"),
        irreps_out=o3.Irreps(f"{target_channels}x0e"),
        correlation=model_config_foundation['correlation'],
        num_elements=model_config_foundation['num_elements'],
        use_reduced_cg=model_config_foundation['use_reduced_cg'],
    )
    with torch.no_grad():
        symmetric_contractions.contractions[0].weights_max.copy_(
            model.products[1].symmetric_contractions.contractions[0].weights_max[:, :, indices_sorted_7]
        )
        symmetric_contractions.contractions[0].weights[0].copy_(
            model.products[1].symmetric_contractions.contractions[0].weights[0][:, :, indices_sorted_7]
        )
        symmetric_contractions.contractions[0].weights[1].copy_(
            model.products[1].symmetric_contractions.contractions[0].weights[1][:, :, indices_sorted_7]
        )
    model.products[1].symmetric_contractions = symmetric_contractions

    linear = o3.Linear(
        o3.Irreps(f"{target_channels}x0e"),
        o3.Irreps(f"{target_channels}x0e"),
        internal_weights=True,
        shared_weights=True,
    )
    with torch.no_grad():
        linear.weight.copy_(model.products[1].linear.weight.view(224, 224)[indices_sorted_7, :][:, indices_sorted_8].reshape(-1))
    model.products[1].linear = linear

    linear_1 = o3.Linear(
        o3.Irreps(f"{target_channels}x0e"),
        o3.Irreps("16x0e"),
        internal_weights=True,
        shared_weights=True,
    )
    with torch.no_grad():
        linear_1.weight.copy_(model.readouts[1].linear_1.weight.view(224, 16)[indices_sorted_8, :].reshape(-1))
    model.readouts[1].linear_1 = linear_1

    if target_l == 2:
        return model
    if target_l == 0:
        from prune_mp import prune_order
        return prune_order(model, model_config_foundation, target_l=0,
                           target_channels=target_channels)

    from e3nn import o3
    from mace.modules.symmetric_contraction import SymmetricContraction

    symmetric_contractions = SymmetricContraction(
        irreps_in=o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e+{target_channels}x3o"),
        irreps_out=o3.Irreps(f"{target_channels}x0e+{target_channels}x1o"),
        correlation=model_config_foundation['correlation'],
        num_elements=model_config_foundation['num_elements'],
        use_reduced_cg=model_config_foundation['use_reduced_cg'],
    )
    with torch.no_grad():
        symmetric_contractions.contractions[0].weights_max.copy_(
            model.products[0].symmetric_contractions.contractions[0].weights_max
        )
        symmetric_contractions.contractions[0].weights[0].copy_(
            model.products[0].symmetric_contractions.contractions[0].weights[0]
        )
        symmetric_contractions.contractions[0].weights[1].copy_(
            model.products[0].symmetric_contractions.contractions[0].weights[1]
        )
        symmetric_contractions.contractions[1].weights_max.copy_(
            model.products[0].symmetric_contractions.contractions[1].weights_max
        )
        symmetric_contractions.contractions[1].weights[0].copy_(
            model.products[0].symmetric_contractions.contractions[1].weights[0]
        )
        symmetric_contractions.contractions[1].weights[1].copy_(
            model.products[0].symmetric_contractions.contractions[1].weights[1]
        )
    model.products[0].symmetric_contractions = symmetric_contractions

    from e3nn.util import prod

    linear = o3.Linear(
        o3.Irreps(f"{target_channels}x0e+{target_channels}x1o"),
        o3.Irreps(f"{target_channels}x0e+{target_channels}x1o"),
        internal_weights=True,
        shared_weights=True,
    )
    with torch.no_grad():
        linear.weight.copy_(model.products[0].linear.weight[:prod(model.products[0].linear.instructions[0].path_shape) + prod(model.products[0].linear.instructions[1].path_shape)])
    model.products[0].linear = linear

    linear = o3.Linear(
        o3.Irreps(f"{target_channels}x0e+{target_channels}x1o"),
        o3.Irreps("1x0e"),
        internal_weights=True,
        shared_weights=True,
    )
    with torch.no_grad():
        linear.weight.copy_(model.readouts[0].linear.weight)
    model.readouts[0].linear = linear

    linear_up = o3.Linear(
        o3.Irreps(f"{target_channels}x0e+{target_channels}x1o"),
        o3.Irreps(f"{target_channels}x0e+{target_channels}x1o"),
        internal_weights=True,
        shared_weights=True,
    )
    with torch.no_grad():
        linear_up.weight.copy_(model.interactions[1].linear_up.weight[:prod(model.interactions[1].linear_up.instructions[0].path_shape) + prod(model.interactions[1].linear_up.instructions[1].path_shape)])
    model.interactions[1].linear_up = linear_up

    i_out_set = set()
    for instruction in model.interactions[1].conv_tp.instructions:
        if instruction.i_in1 == 0 or instruction.i_in1 == 1 :
            i_out_set.add(instruction.i_out)

    weight_segments = []
    flat_weight_index = 0
    for instruction in model.interactions[1].linear.instructions:
        if instruction.i_in in i_out_set:
            weight_segments.append(model.interactions[1].linear.weight[flat_weight_index:flat_weight_index + prod(instruction.path_shape)])
        flat_weight_index += prod(instruction.path_shape)
    weight = torch.cat(weight_segments, dim=0)

    linear = o3.Linear(
        o3.Irreps(f"{2 * target_channels}x0e+{3 * target_channels}x1o+{3 * target_channels}x2e+{2 * target_channels}x3o"),
        o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e+{target_channels}x3o"),
        internal_weights=True,
        shared_weights=True,
    )
    with torch.no_grad():
        linear.weight.copy_(weight)
    model.interactions[1].linear = linear

    from e3nn import nn

    conv_tp_weights = nn.FullyConnectedNet(
        [8, 64, 64, 64, 10 * target_channels],
        torch.nn.functional.silu,
    )

    weight_segments = []
    flat_weight_index = 0
    for instruction in model.interactions[1].conv_tp.instructions:
        if instruction.i_in1 == 0 or instruction.i_in1 == 1 :
            weight_segments.append(model.interactions[1].conv_tp_weights.layer3.weight[:, flat_weight_index:flat_weight_index + prod(instruction.path_shape)])
        flat_weight_index += prod(instruction.path_shape)
    weight = torch.cat(weight_segments, dim=1)

    with torch.no_grad():
        conv_tp_weights.layer0.weight.copy_(model.interactions[1].conv_tp_weights.layer0.weight)
        conv_tp_weights.layer1.weight.copy_(model.interactions[1].conv_tp_weights.layer1.weight)
        conv_tp_weights.layer2.weight.copy_(model.interactions[1].conv_tp_weights.layer2.weight)
        conv_tp_weights.layer3.weight.copy_(weight)
    model.interactions[1].conv_tp_weights = conv_tp_weights

    from mace.modules.irreps_tools import tp_out_irreps_with_instructions

    irreps_mid, instructions = tp_out_irreps_with_instructions(
        o3.Irreps(f"{target_channels}x0e+{target_channels}x1o"),
        o3.Irreps("1x0e+1x1o+1x2e+1x3o"),
        o3.Irreps(f"{2 * target_channels}x0e+{3 * target_channels}x1o+{3 * target_channels}x2e+{2 * target_channels}x3o"),
    )

    conv_tp = o3.TensorProduct(
        o3.Irreps(f"{target_channels}x0e+{target_channels}x1o"),
        o3.Irreps("1x0e+1x1o+1x2e+1x3o"),
        irreps_mid,
        instructions=instructions,
        shared_weights=False,
        internal_weights=False,
    )

    model.interactions[1].conv_tp = conv_tp

    skip_tp = o3.FullyConnectedTensorProduct(
        o3.Irreps(f"{target_channels}x0e+{target_channels}x1o"),
        o3.Irreps("10x0e"),
        o3.Irreps(f"{target_channels}x0e"),
        shared_weights=True,
        internal_weights=True,
    )
    with torch.no_grad():
        skip_tp.weight.copy_(model.interactions[1].skip_tp.weight)
    model.interactions[1].skip_tp = skip_tp
    return model


def main():
    parser = argparse.ArgumentParser(description="Prune MACE-OFF23 large with configurable L and channels")
    parser.add_argument("--model", required=True, help="Source MACE checkpoint")
    parser.add_argument("--output", required=True, help="Pruned MACE checkpoint")
    parser.add_argument("--scores", help="Calibration scores, required only when reducing channel width")
    parser.add_argument("--target-channels", type=int, default=128, help="Target channel count (default: 128)")
    parser.add_argument("--target-l", type=int, choices=(0, 1, 2), default=1, help="Target order (default: 1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from model_building import build_model

    torch.manual_seed(args.seed)
    config, model = build_model(args.model)
    if not 1 <= args.target_channels <= 224:
        parser.error("--target-channels must be between 1 and 224")
    hidden = config["hidden_irreps"]
    source_channels = hidden[0].mul
    original_parameters = sum(p.numel() for p in model.parameters())
    print(f"Pruning: L{hidden.lmax} C{source_channels} -> L{args.target_l} C{args.target_channels}", flush=True)
    scores = None
    if args.target_channels < 224:
        if not args.scores:
            parser.error("--scores is required for channel pruning")
        from calibration import load_scores
        print(f"Using scores: {args.scores}", flush=True)
        scores = load_scores(args.scores, model, config)
    else:
        print("Keeping all channels. Calibration scores are not required.", flush=True)
    model = prune(model, config, scores, target_l=args.target_l, target_channels=args.target_channels)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, output)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {original_parameters:,} -> {parameters:,}", flush=True)
    print(f"Saved model: {output}", flush=True)


if __name__ == "__main__":
    main()
