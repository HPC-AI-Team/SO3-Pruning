"""Build a source MACE model with the tensor layout used by pruning."""

import os
import torch
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
torch.set_default_dtype(torch.float64)

import warnings
warnings.filterwarnings(
    "ignore",
    message="The TorchScript type system doesn't support instance-level annotations"
)

from argparse import Namespace
from typing import Optional
from mace.calculators.foundations_models import mace_mp, mace_mp_names, mace_off
import logging

from mace.tools.scripts_utils import extract_config_mace_model
from mace.tools.utils import AtomicNumberTable
from mace.tools.model_script_utils import _build_model
from mace.tools.finetuning_utils import load_foundations_elements

def build_model(foundation_model):
    logging.basicConfig(level=logging.WARNING)

    print(f"Loading model: {foundation_model}", flush=True)
    args = Namespace()
    args.foundation_model = foundation_model
    args.device = "cpu"
    args.default_dtype = "float64"

    model_foundation: Optional[torch.nn.Module] = None
    valid_mace_mp_models = [name for name in mace_mp_names if name is not None]

    if args.foundation_model in valid_mace_mp_models:
        logging.debug(
            f"Using foundation model mace-mp-0 {args.foundation_model} as initial checkpoint."
        )
        calc = mace_mp(
            model=args.foundation_model,
            device=args.device,
            default_dtype=args.default_dtype,
        )
        model_foundation = calc.models[0]
    elif args.foundation_model in ["small_off", "medium_off", "large_off"]:
        model_type = args.foundation_model.split("_")[0]
        logging.debug(
            f"Using foundation model mace-off-2023 {model_type} as initial checkpoint. ASL license."
        )
        calc = mace_off(
            model=model_type,
            device=args.device,
            default_dtype=args.default_dtype,
        )
        model_foundation = calc.models[0]
    else:
        model_foundation = torch.load(
            args.foundation_model, map_location=args.device, weights_only=False
        )
        logging.debug(
            f"Using foundation model {args.foundation_model} as initial checkpoint."
        )
    args.r_max = model_foundation.r_max.item()
    foundation_model_avg_num_neighbors = model_foundation.interactions[
        0
    ].avg_num_neighbors

    args.model = model_foundation.__class__.__name__
    if args.model == "AtomicDipolesMACE":
        atomic_energies = None
        dipole_only = True
        args.compute_dipole = True
        args.compute_energy = False
        args.compute_forces = False
        args.compute_virials = False
        args.compute_stress = False
        args.compute_polarizability = False
    elif args.model == "AtomicDielectricMACE":
        atomic_energies = None
        dipole_only = False
        args.compute_dipole = True
        args.compute_energy = False
        args.compute_forces = False
        args.compute_virials = False
        args.compute_stress = False
        args.compute_polarizability = True
    else:
        dipole_only = False
        if args.model == "EnergyDipolesMACE":
            args.compute_dipole = True
            args.compute_energy = True
            args.compute_forces = True
            args.compute_virials = False
            args.compute_stress = False
            args.compute_polarizability = False
        else:
            args.compute_energy = True
            args.compute_dipole = False
            args.compute_polarizability = False

    # Build new model
    if model_foundation is not None and args.model in [
        "MACE",
        "ScaleShiftMACE",
        "MACELES",
    ]:
        logging.debug("Loading FOUNDATION model")
        model_config_foundation = extract_config_mace_model(model_foundation)

        foundation_z_table = AtomicNumberTable(
            [int(z) for z in model_foundation.atomic_numbers]
        )
        model_config_foundation["atomic_numbers"] = foundation_z_table.zs
        model_config_foundation["num_elements"] = len(foundation_z_table)
        z_table = foundation_z_table
        logging.debug(
            f"Using all elements from foundation model: {foundation_z_table.zs}"
        )

        args.max_L = model_config_foundation["hidden_irreps"].lmax

        args.avg_num_neighbors = model_config_foundation["avg_num_neighbors"]
        args.model = (
            "FoundationMACELES" if args.model == "MACELES" else "FoundationMACE"
        )
        heads = model_config_foundation["heads"]
        model_config = model_config_foundation

        logging.debug("Model configuration extracted from foundation model")
        logging.debug(
            f"Message passing with hidden irreps {model_config_foundation['hidden_irreps']})"
        )
        logging.debug(
            f"{model_config_foundation['num_interactions']} layers, each with correlation order: {model_config_foundation['correlation']} (body order: {model_config_foundation['correlation']+1}) and spherical harmonics up to: l={model_config_foundation['max_ell']}"
        )
        logging.debug(
            f"Radial cutoff: {model_config_foundation['r_max']} A (total receptive field for each atom: {model_config_foundation['r_max'] * model_config_foundation['num_interactions']} A)"
        )
        logging.debug(
            f"Distance transform for radial basis functions: {model_config_foundation['distance_transform']}"
        )

    model = _build_model(args, model_config, model_config_foundation, heads)

    model = load_foundations_elements(
        model,
        model_foundation,
        z_table,
        load_readout=True,
        max_L=args.max_L,
    )

    hidden = model_config_foundation["hidden_irreps"]
    channels = hidden[0].mul
    print(f"Loaded model: L{hidden.lmax}, {channels} channels", flush=True)

    return model_config_foundation, model
