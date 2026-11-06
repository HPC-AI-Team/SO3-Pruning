import argparse
from copy import deepcopy
from pathlib import Path

import torch


def prune(model, model_0):
    from e3nn import o3
    from e3nn.util import prod
    from e3nn import nn

    linear = o3.Linear(
        o3.Irreps("128x0e"),
        o3.Irreps("224x0e+64x1e+32x2e"),
    )
    with torch.no_grad():
        linear.weight.copy_(model[3].linear.weight.view(128, 256)[:, :224].reshape(-1))
    model[3].linear = linear

    weight_nn = nn.FullyConnectedNet(
        [8, 64, 64, 384],
        torch.nn.functional.silu,  # gate
    )

    weight_segments = []
    flat_weight_index = 0
    for instruction in model[5].convolution.instructions:
        if instruction.i_out != 3:
            weight_segments.append(model[5].weight_nn.layer2.weight[:, flat_weight_index:flat_weight_index + prod(instruction.path_shape)])
        flat_weight_index += prod(instruction.path_shape)
    weight = torch.cat(weight_segments, dim=1)

    with torch.no_grad():
        weight_nn.layer0.weight.copy_(model[5].weight_nn.layer0.weight)
        weight_nn.layer1.weight.copy_(model[5].weight_nn.layer1.weight)
        weight_nn.layer2.weight.copy_(weight)
    model[5].weight_nn = weight_nn

    conv_tp = o3.TensorProduct(**model_0[5].convolution_kwargs)
    model[5].convolution = conv_tp

    weight_segments = []
    flat_weight_index = 0
    for instruction in model[6].linear.instructions:
        if instruction.i_out == 0:
            weight_segments.append(model[6].linear.weight[flat_weight_index:flat_weight_index + prod(instruction.path_shape)].view(128, 256)[:, :224].reshape(-1))
        elif instruction.i_out != 3:
            weight_segments.append(model[6].linear.weight[flat_weight_index:flat_weight_index + prod(instruction.path_shape)])
        flat_weight_index += prod(instruction.path_shape)
    weight = torch.cat(weight_segments, dim=0)

    linear = o3.Linear(
        o3.Irreps("128x0e+128x1e+128x2e"),
        o3.Irreps("224x0e+64x1e+32x2e"),
    )
    with torch.no_grad():
        linear.weight.copy_(weight)
    model[6].linear = linear

    model[8].gate = model_0[8].gate

    for base in (9, 15, 21):
        weight_segments = []
        flat_weight_index = 0
        for instruction in model[base + 0].linear.instructions:
            if instruction.i_out == 0:
                weight_segments.append(model[base + 0].linear.weight[flat_weight_index:flat_weight_index + prod(instruction.path_shape)].view(128, 256)[:, :224].reshape(-1))
            elif instruction.i_out != 3:
                weight_segments.append(model[base + 0].linear.weight[flat_weight_index:flat_weight_index + prod(instruction.path_shape)])
            flat_weight_index += prod(instruction.path_shape)
        weight = torch.cat(weight_segments, dim=0)

        linear = o3.Linear(
            o3.Irreps("128x0e+64x1e+32x2e"),
            o3.Irreps("224x0e+64x1e+32x2e"),
        )
        with torch.no_grad():
            linear.weight.copy_(weight)
        model[base + 0].linear = linear

        weight_segments = []
        flat_weight_index = 0
        for instruction in model[base + 1].linear.instructions:
            if instruction.i_out != 3:
                weight_segments.append(model[base + 1].linear.weight[flat_weight_index:flat_weight_index + prod(instruction.path_shape)])
            flat_weight_index += prod(instruction.path_shape)
        weight = torch.cat(weight_segments, dim=0)

        linear = o3.Linear(
            o3.Irreps("128x0e+64x1e+32x2e"),
            o3.Irreps("128x0e+64x1e+32x2e"),
        )
        with torch.no_grad():
            linear.weight.copy_(weight)
        model[base + 1].linear = linear

        weight_nn = nn.FullyConnectedNet(
            [8, 64, 64, 960],
            torch.nn.functional.silu,  # gate
        )

        weight_segments = []
        flat_weight_index = 0
        for instruction in model[base + 2].convolution.instructions:
            if instruction.i_in1 != 3 and instruction.i_in2 != 3 and instruction.i_out < 24:
                weight_segments.append(model[base + 2].weight_nn.layer2.weight[:, flat_weight_index:flat_weight_index + prod(instruction.path_shape)])
            flat_weight_index += prod(instruction.path_shape)
        weight = torch.cat(weight_segments, dim=1)

        with torch.no_grad():
            weight_nn.layer0.weight.copy_(model[base + 2].weight_nn.layer0.weight)
            weight_nn.layer1.weight.copy_(model[base + 2].weight_nn.layer1.weight)
            weight_nn.layer2.weight.copy_(weight)
        model[base + 2].weight_nn = weight_nn

        conv_tp = o3.TensorProduct(**model_0[base + 2].convolution_kwargs)
        model[base + 2].convolution = conv_tp

        weight_segments = []
        flat_weight_index = 0
        for instruction in model[base + 3].linear.instructions:
            if instruction.i_out == 0:
                weight_segments.append(model[base + 3].linear.weight[flat_weight_index:flat_weight_index + prod(instruction.path_shape)].view(256, 256)[:224, :][:, :224].reshape(-1))
            elif instruction.i_out == 1:
                weight_segments.append(model[base + 3].linear.weight[flat_weight_index:flat_weight_index + prod(instruction.path_shape)].view(480, 64)[:384, :].reshape(-1))
            elif instruction.i_out == 2:
                weight_segments.append(model[base + 3].linear.weight[flat_weight_index:flat_weight_index + prod(instruction.path_shape)].view(544, 32)[:352, :].reshape(-1))
            flat_weight_index += prod(instruction.path_shape)
        weight = torch.cat(weight_segments, dim=0)

        linear = o3.Linear(
            o3.Irreps("224x0e+384x1e+352x2e"),
            o3.Irreps("224x0e+64x1e+32x2e"),
        )
        with torch.no_grad():
            linear.weight.copy_(weight)
        model[base + 3].linear = linear

        model[base + 5].gate = model_0[base + 5].gate

    linear = o3.Linear(
        o3.Irreps("128x0e+64x1e+32x2e"),
        o3.Irreps("128x0e"),
    )
    with torch.no_grad():
        linear.weight.copy_(model[27].linear.weight)
    model[27].linear = linear

    weight_segments = []
    flat_weight_index = 0
    for instruction in model[28].linear.instructions:
        if instruction.i_out != 3:
            weight_segments.append(model[28].linear.weight[flat_weight_index:flat_weight_index + prod(instruction.path_shape)])
        flat_weight_index += prod(instruction.path_shape)
    weight = torch.cat(weight_segments, dim=0)

    linear = o3.Linear(
        o3.Irreps("128x0e+64x1e+32x2e"),
        o3.Irreps("128x0e+64x1e+32x2e"),
    )
    with torch.no_grad():
        linear.weight.copy_(weight)
    model[28].linear = linear

    weight_nn = nn.FullyConnectedNet(
        [8, 64, 64, 224],
        torch.nn.functional.silu,  # gate
    )

    weight_segments = []
    flat_weight_index = 0
    for instruction in model[29].convolution.instructions:
        if instruction.i_out != 3:
            weight_segments.append(model[29].weight_nn.layer2.weight[:, flat_weight_index:flat_weight_index + prod(instruction.path_shape)])
        flat_weight_index += prod(instruction.path_shape)
    weight = torch.cat(weight_segments, dim=1)

    with torch.no_grad():
        weight_nn.layer0.weight.copy_(model[29].weight_nn.layer0.weight)
        weight_nn.layer1.weight.copy_(model[29].weight_nn.layer1.weight)
        weight_nn.layer2.weight.copy_(weight)
    model[29].weight_nn = weight_nn

    conv_tp = o3.TensorProduct(**model_0[29].convolution_kwargs)
    model[29].convolution = conv_tp

    linear = o3.Linear(
        o3.Irreps("224x0e"),
        o3.Irreps("128x0e"),
    )
    with torch.no_grad():
        linear.weight.copy_(model[30].linear.weight.view(256, 128)[:224].reshape(-1))
    model[30].linear = linear
    return model


def main():
    parser = argparse.ArgumentParser(description='Prune SevenNet-l3i5 to the SevenNet-0 architecture (L=2)')
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    from sevenn.checkpoint import SevenNetCheckpoint
    from sevenn.model_build import build_E3_equivariant_model

    print(f'Loading: {args.model}', flush=True)
    source = SevenNetCheckpoint(args.model)
    config = deepcopy(source.config)
    expected_irreps = ['128x0e'] + ['128x0e+64x1e+32x2e+32x3e'] * 4 + ['128x0e']
    from e3nn import o3

    if (config['lmax'] != 3 or config['num_convolution_layer'] != 5
            or config['channel'] != 128
            or [o3.Irreps(x) for x in config['irreps_manual']]
            != [o3.Irreps(x) for x in expected_irreps]):
        raise ValueError('Expected the SevenNet-l3i5 architecture (L3, 5 layers).')
    model = source.build_model(enable_cueq=False, enable_flash=False).cpu()
    print('Pruning: SevenNet-l3i5 -> L2', flush=True)
    config['lmax'] = 2
    config['irreps_manual'] = ['128x0e'] + ['128x0e+64x1e+32x2e'] * 4 + ['128x0e']
    config['cuequivariance_config'] = {'use': False}
    model_0 = build_E3_equivariant_model(config)
    model = prune(model, model_0)
    model_0.load_state_dict(model.state_dict(), strict=True)

    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    for key in ('optimizer_state_dict', 'scheduler_state_dict', 'epoch'):
        checkpoint.pop(key, None)
    checkpoint['config'] = config
    checkpoint['model_state_dict'] = model_0.state_dict()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f'Parameters: {sum(p.numel() for p in model_0.parameters()):,}')
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
