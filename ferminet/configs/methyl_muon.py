import os

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["JAX_ENABLE_X64"] = "1"

from ferminet.utils import system
from ferminet import base_config


def get_config():
    # Get default options.
    cfg = base_config.default()
    # Set up molecule
    cfg.system.electrons = (5, 4, 1)

    #cfg.system.molecule = [
    #    system.Atom("C", (1.90428, 1.34926, 0.28818), units="bohr"),
    #    system.Atom("H", (2.30584, 3.34047, 0.54178), units="bohr"),
    #    system.Atom("H", (3.42929, -0.01587, 0.32636), units="bohr"),
    #]

    cfg.system.molecule = [
        system.Atom("C", (1.90428, 1.34926, 0.28818), units="bohr"),
        system.Atom("H", (2.31057794, 3.36396396, 0.54477218), units="bohr"),
        system.Atom("H", (3.44728334, -0.03197694,  0.32681048), units="bohr"),
    ]

    # Set training hyperparameters
    cfg.batch_size = 4096
    cfg.pretrain.iterations = 0

    cfg.optim.lr.rate = 0.01
    cfg.optim.clip_median = True

    cfg.network.network_type = "gcmnnwf"
    cfg.network.gcmnnwf.charges = (-1, -1, 1)
    cfg.network.gcmnnwf.masses = (1.0, 1.0, 206.768)
    cfg.network.gcmnnwf.spins = (1, -1, 1)
    cfg.network.gcmnnwf.kinds = (0, 0, 1)
    cfg.network.psiformer.use_layer_norm = True

    cfg.system.make_local_energy_fn = "ferminet.af.hamiltonian.local_energy"
    cfg.system.make_local_energy_kwargs = {
        "species_charges": cfg.network.gcmnnwf.charges,
        "species_masses": cfg.network.gcmnnwf.masses,
    }

    cfg.log.save_frequency = 1000

    return cfg