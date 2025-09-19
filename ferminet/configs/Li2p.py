from ferminet import base_config
from ferminet.utils import system
from absl import logging
from ferminet import train
from ferminet.constants import MUON_MASS, PROTON_MASS

import jax.numpy as jnp
import sys

def get_config():
  # Get default options.
  cfg = base_config.default()

  # Set up molecule
  cfg.system.particles = (3, 3, 1)
  cfg.system.charges = (-1., -1., 1.)
  cfg.system.masses = (1., 1., 1.)
  cfg.system.molecule = [system.Atom('Li', (0, 0, 0)),
                         system.Atom('Li', (3.348, 0, 0))]

  # Set training hyperparameters
  cfg.batch_size = 4096
  cfg.pretrain.iterations = 0
  cfg.optim.iterations = 5000 + 100

  cfg.optim.laplacian = "folx"

  cfg.log.save_frequency = 5000
  cfg.log.save_path = "checkpoint_training"
  cfg.log.restore_path = "checkpoint_training"

  return cfg

logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)
cfg = get_config()
train.train(cfg)
