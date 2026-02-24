
import jax
import importlib
import logging
from tqdm import tqdm
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import physical_constants
from ferminet import checkpoint
from ferminet import networks
from ferminet.configs import muonioum
from ferminet import utils
from ferminet import mcmc

logger = logging.getLogger(__name__)

host_batch_size = 512
n_gpu = 1
ckpt_restore_filename = "muonioum/qmcjax_ckpt_003326.npz"

cfg = muonioum.get_config()
(t_init,
data,
params,
opt_state_ckpt,
mcmc_width_ckpt,
density_state_ckpt) = checkpoint.restore(
    ckpt_restore_filename, host_batch_size)
logger.info("Loaded checkpoint")

ndim = data.positions.shape[-1]
nspins = cfg.system.particles
n_gpus = data.positions.shape[0]  # First pmapped axis

def get_batch_network(cfg):

    charges = jnp.array([atom.charge for atom in cfg.system.molecule])
    use_complex = cfg.network.get('complex', False)

    if cfg.network.make_feature_layer_fn:
        feature_layer_module, feature_layer_fn = (
            cfg.network.make_feature_layer_fn.rsplit('.', maxsplit=1))
        feature_layer_module = importlib.import_module(feature_layer_module)
        make_feature_layer: networks.MakeFeatureLayer = getattr(
            feature_layer_module, feature_layer_fn
        )
        feature_layer = make_feature_layer(
            natoms=charges.shape[0],
            nspins=cfg.system.particles,
            ndim=cfg.system.ndim,
            **cfg.network.make_feature_layer_kwargs)
    else:
        feature_layer = networks.make_ferminet_features(
            natoms=charges.shape[0],
            nspins=cfg.system.particles,
            ndim=cfg.system.ndim,
            rescale_inputs=cfg.network.get('rescale_inputs', False),
            )


    envelope_module, envelope_fn = (
        cfg.network.make_envelope_fn.rsplit('.', maxsplit=1))
    envelope_module = importlib.import_module(envelope_module)
    make_envelope = getattr(envelope_module, envelope_fn)
    envelope = make_envelope(**cfg.network.make_envelope_kwargs)

    network = networks.make_fermi_net(
            nspins,
            charges,
            ndim=cfg.system.ndim,
            determinants=cfg.network.determinants,
            states=cfg.system.states,
            envelope=envelope,
            feature_layer=feature_layer,
            jastrow=cfg.network.get('jastrow', 'default'),
            bias_orbitals=cfg.network.bias_orbitals,
            full_det=cfg.network.full_det,
            rescale_inputs=cfg.network.get('rescale_inputs', False),
            complex_output=use_complex,
            **cfg.network.ferminet,
        )

    signed_network = network.apply
    if cfg.system.get('states', 0):
        if cfg.optim.objective == 'vmc_overlap':
            logabs_network = networks.make_state_trace(signed_network,
                                                        cfg.system.states)
        else:
            logabs_network = utils.select_output(
                networks.make_total_ansatz(signed_network,
                                            cfg.system.get('states', 0),
                                            complex_output=use_complex), 1)
    else:
        logabs_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]

    batch_network = jax.vmap(
        logabs_network, in_axes=(None, 0, 0, 0, 0), out_axes=0
    )  # batched network

    batch_network = jax.pmap(batch_network)
    return batch_network


batch_network = get_batch_network(cfg=cfg)
logger.info("Loaded network")

atoms = jnp.stack([jnp.array(atom.coords) for atom in cfg.system.molecule])
atoms_to_mcmc = atoms if cfg.mcmc.scale_by_nuclear_distance else None
num_states = cfg.system.get('states', 0) or 1
mcmc_step = mcmc.make_mcmc_step(
      batch_network,
      host_batch_size,
      steps=cfg.mcmc.steps,
      atoms=atoms_to_mcmc,
      blocks=cfg.mcmc.blocks * num_states,
)
logger.info("Loaded sampler")

key = jax.random.key(seed=1)
pos = jax.random.uniform(key, (n_gpu, host_batch_size, ndim))

# Changing all inputs
psi = lambda params, data: batch_network(params, data.positions, data.spins, data.atoms, data.charges)

# Fixing other things, moving only positions
psi_pos = lambda params, pos: batch_network(params, pos, data.spins, data.atoms, data.charges)


"""
Useful running stuff
1. Sampling
data, pmove = mcmc_step(params, data, key, mcmc_width_ckpt)

2. Network evaluation at pos
psi(params, data)

3. Key splitting
key, subkey = jax.random.split(key, num=2)
"""

def compute_observable(pos):
    electron_samples = ...
    muon_samples = ...
    muon_electron_seperations_cart = ...
    muon_electron_seperations = ...

    n_bins = 4000 // 20
    max_radius = 20 // 20

    bin_heights, bin_edges = np.histogram(
        muon_electron_seperations, bins=n_bins, range=(0, max_radius)
    )
    bin_normalization = (4/3) * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)
    bin_heights_normalized = ...

    electron_x = ...
    electron_y = ...

    muon_x = ...
    muon_y = ...

    return jnp.mean(pos)

key, subkey = jax.random.split(key)
data, pmove = mcmc_step(params, data, subkey, mcmc_width_ckpt)

n_samples = 1e5
n_steps = int(n_samples / host_batch_size)
position_data = jnp.zeros((n_steps, n_gpus, pos.shape[1], pos.shape[2]))
mcmc_step = jax.jit(mcmc_step)

for i in tqdm(range(n_steps)):
    key, subkey = jax.random.split(key)
    data, pmove = mcmc_step(params, data, subkey, mcmc_width_ckpt)
    position_data = position_data.at[i].set(data.positions)


total_points = (
    position_data.shape[0] * position_data.shape[1] * position_data.shape[2]
)

slice_num = position_data.shape[0]
electron_samples = position_data[:, :, :, :3].reshape(slice_num, host_batch_size, -1, 3)
muon_samples = position_data[:, :, :, 3:].reshape(slice_num, host_batch_size, -1, 3)

muon_electron_seperations_cart = electron_samples - muon_samples
muon_electron_seperations = np.linalg.norm(muon_electron_seperations_cart, axis=3)

n_bins = 4000 // 20
max_radius = 20 // 20

bin_heights, bin_edges = np.histogram(
    muon_electron_seperations, bins=n_bins, range=(0, max_radius)
)
bin_normalisation = (4 / 3) * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)
bin_heights_normalised = bin_heights / (bin_normalisation * total_points)

electron_x = electron_samples[:, :, :, 0].flatten()
electron_y = electron_samples[:, :, :, 1].flatten()

# Extract muon positions (second particle)
muon_x = muon_samples[:, :, :, 0].flatten()
muon_y = muon_samples[:, :, :, 1].flatten()

plt.figure()
plt.plot(
    (bin_edges[1:] + bin_edges[:-1]) / 2, bin_heights_normalised, marker="o", ls=""
)
plt.xlabel("Separation (bohr)")
plt.ylabel("Normalised Counts")
plt.title("Muon-Electron Separation Normalised Histogram")
plt.savefig("plots/muonium_separation_normalised.png")

plt.figure()
plt.plot((bin_edges[1:] + bin_edges[:-1]) / 2, bin_heights, marker="o", ls="")
plt.xlabel("Separation (bohr)")
plt.ylabel("Normalised Counts")
plt.title("Muon-Electron Separation Normalised Histogram")
plt.savefig("plots/muonium_seperation.png")
# fit an exponential to the bin_heights_normalised to interpolate at r=0

# Define the exponential function to fit
def exp_func(r, A, B):
    B = 2 * 0.9951857
    return A * np.exp(-B * r)

# Bin centers
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
y_data = bin_heights_normalised

# Perform the curve fitting
popt, pcov = curve_fit(exp_func, bin_centers, bin_heights_normalised)

# Extract fitting parameters
A_fit, B_fit = popt

# Generate fitted data for plotting
r_fine = np.linspace(0, max_radius, 1000)
print("Optimised", popt)
y_fitted = exp_func(r_fine, *popt)

# Plot the original data and the fitted function
plt.figure()
plt.plot(bin_centers, y_data, "o", label="Data")
plt.plot(r_fine, y_fitted, "-", label="Fitted Exponential")
plt.xlabel("Separation (bohr)")
plt.ylabel("Normalized Counts")
plt.title("Muon-Electron Separation with Exponential Fit")
plt.legend()
plt.savefig("plots/muonium_separation_fit.png")

# Interpolate at r=0
y_at_r0 = exp_func(0, *popt)

print(f"Interpolated value at r=0: {y_at_r0}")

# Constants

y_in_SI = y_at_r0 / ((physical_constants["Bohr radius"][0]) ** 3)

mu_e = physical_constants["electron mag. mom."][0]  # J·T^{-1} = J A S^2 KG^-1
mu_mu = physical_constants["muon mag. mom."][0]  # J·T^{-1} = J A S^2 KG^-1
mu_0 = physical_constants["vacuum mag. permeability"][0]  # N·A^{-2}
hbar = physical_constants["reduced Planck constant"][0]  # J·s

g_e = 2.002_319_304_360_92  # Electron g-factor
g_mu = -2.002_331_841_23  # Muon g-factor

gamma_e = (g_e * mu_e) / (hbar)  # units of T-1 S-1 = KG-1 S1 A1
gamma_mu = (g_mu * mu_mu) / (hbar)  # units of  T-1 S-1 = KG-1 S1 A1

# N = KG M S^-2

# gamma_e * gamma_mu = A2 S2 KG-2
# gamma_e * gamma_mu * mu_0 = N S2 KG-2 = KG M

# gamma_e * gamma_mu * mu_0 * hbar * |psi|^2 = J S KG M-2

# Fermi contact coupling constant A
A_constant = (mu_0 * 2 * hbar) / (3)
A_value = A_constant * gamma_e * gamma_mu * y_in_SI
A_value_freq = A_value / (2 * np.pi)
print(f"Fermi contact coupling A Hz : {A_value_freq}")
print(gamma_e, gamma_mu, A_constant)