import argparse
from pathlib import Path

import torch


def needs_calibration(config, target_l, target_channels):
    source = config["hidden_irreps"]
    source_l = source.lmax
    channels = {irrep.mul for irrep in source}
    if not isinstance(target_channels, int) or not 1 <= target_channels <= max(channels):
        raise ValueError("Target channels must be a positive integer no larger than the source width.")
    if not isinstance(target_l, int) or not 0 <= target_l <= source_l:
        raise ValueError("Target L must be an integer between 0 and the source L.")
    if (source_l, tuple(channels)) in ((2, (128,)), (1, (256,))):
        return target_channels < next(iter(channels))
    raise ValueError(
        "Supported MP sources: L2 C128 and L1 C256."
    )


def prune_channels(model, model_config_foundation, taylor, target_l, target_channels=128):
    from mace.modules.blocks import RadialEmbeddingBlock
    from e3nn import o3

    source_l = model_config_foundation['hidden_irreps'].lmax
    source_channels = model_config_foundation['hidden_irreps'][0].mul
    hidden = o3.Irreps([(target_channels, (l, (-1) ** l)) for l in range(source_l + 1)])

    def select_channels(*names):
        if target_channels == source_channels:
            return torch.arange(source_channels)
        if taylor is None:
            raise ValueError("Reducing channel width requires calibration scores.")
        score = taylor[names[0]]
        for name in names[1:]:
            score = score + taylor[name]
        return torch.sort(torch.topk(score, target_channels).indices).values

    if source_l == 1:
        model.radial_embedding = RadialEmbeddingBlock(
            r_max=6.0,
            num_bessel=10,
            num_polynomial_cutoff=5.0,
            radial_type="bessel",
            distance_transform=None,
            apply_cutoff=True,
        )

        model.r_max = torch.tensor(6.0)

    from e3nn import o3

    indices_sorted_1 = select_channels("node_embedding")

    linear = o3.Linear(
        o3.Irreps("89x0e"),
        o3.Irreps(f"{target_channels}x0e"),
        internal_weights=True,
        shared_weights=True,
    )
    with torch.no_grad():
        linear.weight.copy_(model.node_embedding.linear.weight.view(89, source_channels)[:, indices_sorted_1].reshape(-1))
    model.node_embedding.linear = linear

    indices_sorted_2 = select_channels("interactions.0.conv_tp")

    linear_up = o3.Linear(
        o3.Irreps(f"{target_channels}x0e"),
        o3.Irreps(f"{target_channels}x0e"),
        internal_weights=True,
        shared_weights=True,
    )
    with torch.no_grad():
        linear_up.weight.copy_(model.interactions[0].linear_up.weight.view(source_channels, source_channels)[indices_sorted_1, :][:, indices_sorted_2].reshape(-1))
    model.interactions[0].linear_up = linear_up

    from e3nn import nn
    from e3nn.util import prod

    conv_tp_weights = nn.FullyConnectedNet(
        [10, 64, 64, 64, 4 * target_channels],
        torch.nn.functional.silu,
    )

    weight_segments = []
    flat_weight_index = 0
    for instruction in model.interactions[0].conv_tp.instructions:
        weight_segments.append(model.interactions[0].conv_tp_weights.layer3.weight[:, flat_weight_index:flat_weight_index + source_channels][:, indices_sorted_2])
        flat_weight_index += prod(instruction.path_shape)
    weight = torch.cat(weight_segments, dim=1)

    with torch.no_grad():
        if source_l == 2:
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

    indices_sorted_3 = select_channels("products.0.symmetric_contractions")

    weight_segments = []
    flat_weight_index = 0
    for instruction in model.interactions[0].linear.instructions:
        weight_segments.append(model.interactions[0].linear.weight[flat_weight_index:flat_weight_index + prod(instruction.path_shape)].view(source_channels, source_channels)[indices_sorted_2, :][:, indices_sorted_3].reshape(-1))
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

    indices_sorted_4 = select_channels("interactions.0.skip_tp", "products.0.linear")

    skip_tp = o3.FullyConnectedTensorProduct(
        o3.Irreps(f"{target_channels}x0e"),
        o3.Irreps("89x0e"),
        hidden,
        shared_weights=True,
        internal_weights=True,
    )
    with torch.no_grad():
        skip_tp.weight.copy_(model.interactions[0].skip_tp.weight.view(source_channels, 89, source_channels)[indices_sorted_1, :, :][:, :, indices_sorted_4].reshape(-1))
    model.interactions[0].skip_tp = skip_tp

    from mace.modules.irreps_tools import reshape_irreps

    model.interactions[0].reshape = reshape_irreps(o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e+{target_channels}x3o"), cueq_config=model_config_foundation["cueq_config"])

    from mace.modules.symmetric_contraction import SymmetricContraction

    symmetric_contractions = SymmetricContraction(
        irreps_in=o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e+{target_channels}x3o"),
        irreps_out=hidden,
        correlation=model_config_foundation['correlation'],
        num_elements=model_config_foundation['num_elements'],
        use_reduced_cg=model_config_foundation['use_reduced_cg'],
    )
    with torch.no_grad():
        for target, source in zip(symmetric_contractions.contractions,
                                  model.products[0].symmetric_contractions.contractions):
            target.weights_max.copy_(source.weights_max[:, :, indices_sorted_3])
            for target_weight, source_weight in zip(target.weights, source.weights):
                target_weight.copy_(source_weight[:, :, indices_sorted_3])
    model.products[0].symmetric_contractions = symmetric_contractions

    weight_segments = []
    flat_weight_index = 0
    for instruction in model.products[0].linear.instructions:
        weight_segments.append(model.products[0].linear.weight[flat_weight_index:flat_weight_index + prod(instruction.path_shape)].view(source_channels, source_channels)[indices_sorted_3, :][:, indices_sorted_4].reshape(-1))
        flat_weight_index += prod(instruction.path_shape)
    weight = torch.cat(weight_segments, dim=0)

    linear = o3.Linear(
        hidden,
        hidden,
        internal_weights=True,
        shared_weights=True,
    )
    with torch.no_grad():
        linear.weight.copy_(weight)
    model.products[0].linear = linear

    linear = o3.Linear(
        hidden,
        o3.Irreps("1x0e"),
        internal_weights=True,
        shared_weights=True,
    )
    with torch.no_grad():
        linear.weight.copy_(model.readouts[0].linear.weight[indices_sorted_4])
    model.readouts[0].linear = linear

    indices_sorted_5 = select_channels("interactions.1.conv_tp")

    weight_segments = []
    flat_weight_index = 0
    for instruction in model.interactions[1].linear_up.instructions:
        weight_segments.append(model.interactions[1].linear_up.weight[flat_weight_index:flat_weight_index + prod(instruction.path_shape)].view(source_channels, source_channels)[indices_sorted_4, :][:, indices_sorted_5].reshape(-1))
        flat_weight_index += prod(instruction.path_shape)
    weight = torch.cat(weight_segments, dim=0)

    linear_up = o3.Linear(
        hidden,
        hidden,
        internal_weights=True,
        shared_weights=True,
    )
    with torch.no_grad():
        linear_up.weight.copy_(weight)
    model.interactions[1].linear_up = linear_up

    from e3nn import nn

    conv_tp_weights = nn.FullyConnectedNet(
        [10, 64, 64, 64, (10 if source_l == 1 else 17) * target_channels],
        torch.nn.functional.silu,
    )

    weight_segments = []
    flat_weight_index = 0
    for instruction in model.interactions[1].conv_tp.instructions:
        weight_segments.append(model.interactions[1].conv_tp_weights.layer3.weight[:, flat_weight_index:flat_weight_index + source_channels][:, indices_sorted_5])
        flat_weight_index += prod(instruction.path_shape)
    weight = torch.cat(weight_segments, dim=1)

    with torch.no_grad():
        if source_l == 2:
            conv_tp_weights.layer0.weight.copy_(model.interactions[1].conv_tp_weights.layer0.weight)
        conv_tp_weights.layer1.weight.copy_(model.interactions[1].conv_tp_weights.layer1.weight)
        conv_tp_weights.layer2.weight.copy_(model.interactions[1].conv_tp_weights.layer2.weight)
        conv_tp_weights.layer3.weight.copy_(weight)
    model.interactions[1].conv_tp_weights = conv_tp_weights

    from mace.modules.irreps_tools import tp_out_irreps_with_instructions

    irreps_mid, instructions = tp_out_irreps_with_instructions(
        hidden,
        o3.Irreps("1x0e+1x1o+1x2e+1x3o"),
        o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e+{target_channels}x3o"),
    )

    conv_tp = o3.TensorProduct(
        hidden,
        o3.Irreps("1x0e+1x1o+1x2e+1x3o"),
        irreps_mid,
        instructions=instructions,
        shared_weights=False,
        internal_weights=False,
    )

    model.interactions[1].conv_tp = conv_tp

    indices_sorted_6 = select_channels("products.1.symmetric_contractions")

    weight_segments = []
    flat_weight_index = 0
    for instruction in model.interactions[1].linear.instructions:
        weight_segments.append(model.interactions[1].linear.weight[flat_weight_index:flat_weight_index + prod(instruction.path_shape)].view(source_channels, source_channels)[indices_sorted_5, :][:, indices_sorted_6].reshape(-1))
        flat_weight_index += prod(instruction.path_shape)
    weight = torch.cat(weight_segments, dim=0)

    linear = o3.Linear(
        irreps_mid if target_l < source_l else irreps_mid.simplify(),
        o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e+{target_channels}x3o"),
        internal_weights=True,
        shared_weights=True,
    )
    with torch.no_grad():
        linear.weight.copy_(weight)
    model.interactions[1].linear = linear

    indices_sorted_7 = select_channels("interactions.1.skip_tp", "products.1.linear")

    skip_tp = o3.FullyConnectedTensorProduct(
        hidden,
        o3.Irreps("89x0e"),
        o3.Irreps(f"{target_channels}x0e"),
        shared_weights=True,
        internal_weights=True,
    )
    with torch.no_grad():
        skip_tp.weight.copy_(model.interactions[1].skip_tp.weight.view(source_channels, 89, source_channels)[indices_sorted_4, :, :][:, :, indices_sorted_7].reshape(-1))
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
            model.products[1].symmetric_contractions.contractions[0].weights_max[:, :, indices_sorted_6]
        )
        symmetric_contractions.contractions[0].weights[0].copy_(
            model.products[1].symmetric_contractions.contractions[0].weights[0][:, :, indices_sorted_6]
        )
        symmetric_contractions.contractions[0].weights[1].copy_(
            model.products[1].symmetric_contractions.contractions[0].weights[1][:, :, indices_sorted_6]
        )
    model.products[1].symmetric_contractions = symmetric_contractions

    linear = o3.Linear(
        o3.Irreps(f"{target_channels}x0e"),
        o3.Irreps(f"{target_channels}x0e"),
        internal_weights=True,
        shared_weights=True,
    )
    with torch.no_grad():
        linear.weight.copy_(model.products[1].linear.weight.view(source_channels, source_channels)[indices_sorted_6, :][:, indices_sorted_7].reshape(-1))
    model.products[1].linear = linear

    linear_1 = o3.Linear(
        o3.Irreps(f"{target_channels}x0e"),
        o3.Irreps("16x0e"),
        internal_weights=True,
        shared_weights=True,
    )
    with torch.no_grad():
        linear_1.weight.copy_(model.readouts[1].linear_1.weight.view(source_channels, 16)[indices_sorted_7, :].reshape(-1))
    model.readouts[1].linear_1 = linear_1
    return model


def prune_order(model, model_config_foundation, *, target_l=0, target_channels=128, reinitialize_radial=False):
    from e3nn import o3
    from mace.modules.blocks import RealAgnosticResidualInteractionBlock
    from mace.modules.irreps_tools import tp_out_irreps_with_instructions

    hidden = o3.Irreps([(target_channels, (l, (-1) ** l)) for l in range(target_l + 1)])
    elements = o3.Irreps(f"{model_config_foundation['num_elements']}x0e")
    intermediate, instructions = tp_out_irreps_with_instructions(
        hidden, model.interactions[1].conv_tp.irreps_in2,
        model.interactions[1].linear.irreps_out,
    )

    if isinstance(model.interactions[0], RealAgnosticResidualInteractionBlock):
        skip_tp = o3.FullyConnectedTensorProduct(
            o3.Irreps(f"{target_channels}x0e"),
            elements,
            hidden,
            shared_weights=True,
            internal_weights=True,
        )
        with torch.no_grad():
            skip_tp.weight.copy_(model.interactions[0].skip_tp.weight)
        model.interactions[0].skip_tp = skip_tp

    from mace.modules.symmetric_contraction import SymmetricContraction

    symmetric_contractions = SymmetricContraction(
        irreps_in=o3.Irreps(f"{target_channels}x0e+{target_channels}x1o+{target_channels}x2e+{target_channels}x3o"),
        irreps_out=hidden,
        correlation=model_config_foundation['correlation'],
        num_elements=model_config_foundation['num_elements'],
        use_reduced_cg=model_config_foundation['use_reduced_cg'],
    )
    with torch.no_grad():
        for target, source in zip(symmetric_contractions.contractions,
                                  model.products[0].symmetric_contractions.contractions):
            target.weights_max.copy_(source.weights_max)
            for target_weight, source_weight in zip(target.weights, source.weights):
                target_weight.copy_(source_weight)
    model.products[0].symmetric_contractions = symmetric_contractions

    from e3nn.util import prod

    linear = o3.Linear(
        hidden,
        hidden,
        internal_weights=True,
        shared_weights=True,
    )
    with torch.no_grad():
        linear.weight.copy_(model.products[0].linear.weight[:sum(prod(ins.path_shape) for ins in model.products[0].linear.instructions if ins.i_in <= target_l)])
    model.products[0].linear = linear

    linear = o3.Linear(
        hidden,
        o3.Irreps("1x0e"),
        internal_weights=True,
        shared_weights=True,
    )
    with torch.no_grad():
        linear.weight.copy_(model.readouts[0].linear.weight)
    model.readouts[0].linear = linear

    linear_up = o3.Linear(
        hidden,
        hidden,
        internal_weights=True,
        shared_weights=True,
    )
    with torch.no_grad():
        linear_up.weight.copy_(model.interactions[1].linear_up.weight[:sum(prod(ins.path_shape) for ins in model.interactions[1].linear_up.instructions if ins.i_in <= target_l)])
    model.interactions[1].linear_up = linear_up

    i_out_set = set()
    for instruction in model.interactions[1].conv_tp.instructions:
        if instruction.i_in1 <= target_l:
            i_out_set.add(instruction.i_out)

    weight_segments = []
    flat_weight_index = 0
    for instruction in model.interactions[1].linear.instructions:
        if instruction.i_in in i_out_set:
            weight_segments.append(model.interactions[1].linear.weight[flat_weight_index:flat_weight_index + prod(instruction.path_shape)])
        flat_weight_index += prod(instruction.path_shape)
    weight = torch.cat(weight_segments, dim=0)

    linear = o3.Linear(
        intermediate,
        model.interactions[1].linear.irreps_out,
        internal_weights=True,
        shared_weights=True,
    )
    with torch.no_grad():
        linear.weight.copy_(weight)
    model.interactions[1].linear = linear

    from e3nn import nn

    conv_tp_weights = nn.FullyConnectedNet(
        [model.interactions[1].conv_tp_weights.layer0.weight.shape[0], 64, 64, 64, len(i_out_set) * target_channels],
        torch.nn.functional.silu,
    )

    weight_segments = []
    flat_weight_index = 0
    for instruction in model.interactions[1].conv_tp.instructions:
        if instruction.i_in1 <= target_l:
            weight_segments.append(model.interactions[1].conv_tp_weights.layer3.weight[:, flat_weight_index:flat_weight_index + prod(instruction.path_shape)])
        flat_weight_index += prod(instruction.path_shape)
    weight = torch.cat(weight_segments, dim=1)

    with torch.no_grad():
        if not reinitialize_radial:
            conv_tp_weights.layer0.weight.copy_(model.interactions[1].conv_tp_weights.layer0.weight)
        conv_tp_weights.layer1.weight.copy_(model.interactions[1].conv_tp_weights.layer1.weight)
        conv_tp_weights.layer2.weight.copy_(model.interactions[1].conv_tp_weights.layer2.weight)
        conv_tp_weights.layer3.weight.copy_(weight)
    model.interactions[1].conv_tp_weights = conv_tp_weights

    conv_tp = o3.TensorProduct(
        hidden,
        o3.Irreps("1x0e+1x1o+1x2e+1x3o"),
        intermediate,
        instructions=instructions,
        shared_weights=False,
        internal_weights=False,
    )

    model.interactions[1].conv_tp = conv_tp

    skip_tp = o3.FullyConnectedTensorProduct(
        hidden,
        elements,
        o3.Irreps(f"{target_channels}x0e"),
        shared_weights=True,
        internal_weights=True,
    )
    with torch.no_grad():
        skip_tp.weight.copy_(model.interactions[1].skip_tp.weight)
    model.interactions[1].skip_tp = skip_tp
    return model



def prune(model, config, taylor=None, *, target_l=0, target_channels=128):
    channel_pruning = needs_calibration(config, target_l, target_channels)
    if channel_pruning and taylor is None:
        raise ValueError("Channel pruning requires calibration scores.")
    source_l = config["hidden_irreps"].lmax
    if target_l == source_l and not channel_pruning:
        return model
    mp_l1_source = source_l == 1
    if mp_l1_source or channel_pruning:
        model = prune_channels(model, config, taylor, target_l, target_channels)
    if target_l < source_l:
        # Preserve the MP L1 C256 recipe's radial-layer initialization.
        model = prune_order(model, config, target_l=target_l, target_channels=target_channels, reinitialize_radial=mp_l1_source)
    return model


def main():
    parser = argparse.ArgumentParser(description="Prune a supported MACE-MP model with configurable L and channels.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-l", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--scores", help="Calibration scores, required only when reducing channel width")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-channels", type=int, default=128, help="Target channel count (default: 128)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from model_building import build_model

    torch.manual_seed(args.seed)
    config, model = build_model(args.model)
    try:
        channel_pruning = needs_calibration(config, args.target_l, args.target_channels)
    except ValueError as error:
        parser.error(str(error))
    hidden = config["hidden_irreps"]
    source_channels = hidden[0].mul
    original_parameters = sum(p.numel() for p in model.parameters())
    print(f"Pruning: L{hidden.lmax} C{source_channels} -> L{args.target_l} C{args.target_channels}", flush=True)
    scores = None
    if channel_pruning:
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
