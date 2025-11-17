import sys

from absl import logging
from ferminet.utils import system
from ferminet import base_config
from ferminet import train
from ferminet.configs import atom

# Optional, for also printing training progress to STDOUT.
# If running a script, you can also just use the --alsologtostderr flag.

logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)

# Define H2 molecule
cfg = base_config.default()
cfg.system.electrons = (1,1)  # (alpha electrons, beta electrons)
cfg.system.molecule = [system.Atom('H', (0, 0, -1)), system.Atom('H', (0, 0, 1))]

# Set training parameters
cfg.batch_size = 4096
cfg.pretrain.iterations = 5
cfg.mcmc.burn_in = 20

train.train(cfg, wandb_monitoring=False)