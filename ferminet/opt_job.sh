#!/bin/bash
#SBATCH --job-name=chlorine_ion
#SBATCH --output=isolated_neon.out
#SBATCH --gpus=1
#SBATCH --time=10:00:00

hostname
nvidia-smi --list-gpus

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate test

# Make sure pip is up to date (optional)
pip install --upgrade pip

# Clone repo if not already present
if [ ! -d "ferminet_parv" ]; then
    git clone https://github.com/Parvfect/ferminet_parv.git
fi

cd ferminet_parv
git pull
git checkout minsr

# Install package in editable mode (once, or if updated)
pip install -e .

# Optional: ensure dependencies
pip install wandb matplotlib "jax[cuda12]"

# Write wandb API key
echo "69cef1b6210baf838412836686b88c74500362ac" > ferminet/wandb_api_key.txt

# Test jax devices
python -c "import jax; print(jax.devices())"

export NVIDIA_TF32_OVERRIDE=0

export JAX_DEFAULT_MATMUL_PRECISION='float32'
# Run ferminet
ferminet --config ferminet/configs/c2h4.py \
    --config.batch_size 4096 \
    --config.mcmc.burn_in 100 \
    --config.pretrain.iterations 0 \
    --config.optim.optimizer $1 \
    --config.log.save_path '/projects/s5i/parv'