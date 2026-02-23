import functools
import importlib
import os
import time
from typing import Optional, Mapping, Sequence, Tuple, Union
import pickle
import json

from absl import logging
from ferminet import constants
from ferminet import curvature_tags_and_blocks
from ferminet import envelopes
from ferminet import hamiltonian
from ferminet import loss as qmc_loss_functions
from ferminet import networks
from ferminet import observables
from ferminet import pretrain
from ferminet import psiformer
from ferminet.utils import utils
from ferminet.stochastic_reconfiguration import \
  make_sr_opt_update_step, make_sr_training_step
from ferminet.time_evolution import \
  make_td_opt_update_step, make_time_evolution_step
from ferminet.time_evolution_low_sample_limit import make_td_opt_update_step_full_solve, \
  make_time_evolution_step_low_sample_limit

import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import kfac_jax
import optax


def device_setup(cfg):
    # Device logging
  num_devices = jax.local_device_count()
  num_hosts = jax.device_count() // num_devices
  num_states = cfg.system.get('states', 0) or 1  # avoid 0/1 confusion
  logging.info('Starting QMC with %i XLA devices per host '
               'across %i hosts.', num_devices, num_hosts)
  if cfg.batch_size % (num_devices * num_hosts) != 0:
    raise ValueError('Batch size must be divisible by number of devices, '
                     f'got batch size {cfg.batch_size} for '
                     f'{num_devices * num_hosts} devices.')
  host_batch_size = cfg.batch_size // num_hosts  # batch size per host
  total_host_batch_size = host_batch_size * num_states
  device_batch_size = host_batch_size // num_devices  # batch size per device
  data_shape = (num_devices, device_batch_size)

  return num_devices, num_hosts, num_states, host_batch_size, total_host_batch_size, device_batch_size, data_shape


def get_feature_layer(cfg):
  
  charges = jnp.array([atom.charge for atom in cfg.system.molecule])

  if cfg.network.make_feature_layer_fn:
    feature_layer_module, feature_layer_fn = (
        cfg.network.make_feature_layer_fn.rsplit('.', maxsplit=1))
    feature_layer_module = importlib.import_module(feature_layer_module)
    make_feature_layer: networks.MakeFeatureLayer = getattr(
        feature_layer_module, feature_layer_fn
    )
    feature_layer = make_feature_layer(
        natoms=charges.shape[0],
        nspins=cfg.system.electrons,
        ndim=cfg.system.ndim,
        **cfg.network.make_feature_layer_kwargs)
  else:
    feature_layer = networks.make_ferminet_features(
        natoms=charges.shape[0],
        nspins=cfg.system.electrons,
        ndim=cfg.system.ndim,
        rescale_inputs=cfg.network.get('rescale_inputs', False),
    )

  return feature_layer

def get_envelope(cfg):
  if cfg.network.make_envelope_fn:
    envelope_module, envelope_fn = (
        cfg.network.make_envelope_fn.rsplit('.', maxsplit=1))
    envelope_module = importlib.import_module(envelope_module)
    make_envelope = getattr(envelope_module, envelope_fn)
    envelope = make_envelope(**cfg.network.make_envelope_kwargs)  # type: envelopes.Envelope
  else:
    envelope = envelopes.make_isotropic_envelope()
  return envelope


def get_network(cfg, key):

  feature_layer = get_feature_layer(cfg)
  envelope = get_envelope(cfg)
  
  use_complex = cfg.network.get('complex', False)

  charges = jnp.array([atom.charge for atom in cfg.system.molecule])
  nspins = cfg.system.electrons

  if cfg.network.network_type == 'ferminet':
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
  elif cfg.network.network_type == 'psiformer':

    network = psiformer.make_fermi_net(
        nspins,
        charges,
        ndim=cfg.system.ndim,
        determinants=cfg.network.determinants,
        states=cfg.system.states,
        envelope=envelope,
        feature_layer=feature_layer,
        jastrow=cfg.network.get('jastrow', 'default'),
        bias_orbitals=cfg.network.bias_orbitals,
        rescale_inputs=cfg.network.get('rescale_inputs', False),
        complex_output=use_complex,
        **cfg.network.psiformer,
    )
  
  key, subkey = jax.random.split(key)
  params = network.init(subkey)
  params = kfac_jax.utils.replicate_all_local_devices(params)
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

  return network, params, signed_network, logabs_network, batch_network


def get_observable_functions(
    cfg, data, signed_network, ckpt_save_path, density_state_ckpt):

  density_update, s2_matrix_file, dipole_matrix_file, density_matrix_file = None, None, None, None
  nspins = cfg.system.electrons
  observable_fns = {}
  observable_states = {}  # only relevant for density matrix

  # Set up logging and observables
  train_schema = ['step', 'energy', 'ewmean', 'ewvar', 'pmove']
  if cfg.observables.s2:
    observable_fns['s2'] = observables.make_s2(
        signed_network,
        nspins,
        states=cfg.system.states)
    observable_states['s2'] = None
    train_schema += ['s2']
    if cfg.system.states:
      s2_matrix_file = open(
          os.path.join(ckpt_save_path, 's2_matrix.npy'), 'ab')
  if cfg.observables.dipole:
    observable_fns['dipole'] = observables.make_dipole(
        signed_network,
        states=cfg.system.states)
    observable_states['dipole'] = None
    train_schema += ['mu_x', 'mu_y', 'mu_z']
    if cfg.system.states:
      dipole_matrix_file = open(
          os.path.join(ckpt_save_path, 'dipole_matrix.npy'), 'ab')
  # Do this *before* creating density matrix function, as that is a special case
  observable_fns = observables.make_observable_fns(observable_fns)
  
  if cfg.observables.density:
    (observable_states['density'],
     density_update,
     observable_fns['density']) = observables.make_density_matrix(
         signed_network, data.positions, cfg, density_state_ckpt)
    # Because the density matrix can be quite large, even without excited
    # states, we always save it directly to .npy file instead of writing to CSV
    density_matrix_file = open(
        os.path.join(ckpt_save_path, 'density_matrix.npy'), 'ab')
    # custom pmaping just for density matrix function
    pmap_density_axes = observables.DensityState(t=None,
                                                 positions=0,
                                                 probabilities=0,
                                                 move_width=0,
                                                 pmove=None,
                                                 mo_coeff=None)
    pmap_fn = constants.pmap(observable_fns['density'],
                             in_axes=(0, 0, pmap_density_axes))
    observable_fns['density'] = lambda *a, **kw: pmap_fn(*a, **kw).mean(0)

  return observable_fns, observable_states, density_update, s2_matrix_file, dipole_matrix_file, density_matrix_file, train_schema


def optimizer_setup(
    cfg, params, data, opt_state_ckpt, evaluate_loss, sharded_key):
  # Construct and setup optimizer
  
  theta_dot, opt_state = None, None

  def learning_rate_schedule(
      t_: jnp.ndarray) -> jnp.ndarray:
      return cfg.optim.lr.rate * jnp.power(
          (1.0 / (1.0 + (t_/cfg.optim.lr.delay))), cfg.optim.lr.decay)

  if cfg.optim.optimizer == 'none':
    optimizer = None
  elif cfg.optim.optimizer == 'adam':
    optimizer = optax.chain(
        optax.scale_by_adam(**cfg.optim.adam),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1.))
  elif cfg.optim.optimizer == 'lamb':
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(eps=1e-7),
        optax.scale_by_trust_ratio(),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1))
  elif cfg.optim.optimizer == 'kfac':
    # Differentiate wrt parameters (argument 0)
    val_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)
    optimizer = kfac_jax.Optimizer(
        val_and_grad,
        l2_reg=cfg.optim.kfac.l2_reg,
        norm_constraint=cfg.optim.kfac.norm_constraint,
        value_func_has_aux=True,
        value_func_has_rng=True,
        learning_rate_schedule=learning_rate_schedule,
        curvature_ema=cfg.optim.kfac.cov_ema_decay,
        inverse_update_period=cfg.optim.kfac.invert_every,
        min_damping=cfg.optim.kfac.min_damping,
        num_burnin_steps=0,
        register_only_generic=cfg.optim.kfac.register_only_generic,
        estimation_mode='fisher_exact',
        multi_device=True,
        pmap_axis_name=constants.PMAP_AXIS_NAME,
        auto_register_kwargs=dict(
            graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS,
        ),
        # debug=True
    )
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    opt_state = optimizer.init(params, subkeys, data)
    opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state

  elif cfg.optim.optimizer == 'sr' or cfg.td.time_evolution:

    if cfg.td.time_evolution:
      
      optimizer = optax.chain(
        optax.scale(cfg.td.parameter_step),)  # the -1j is done within the update itself

    else:
      optimizer = optax.chain(
        store_last_gradient(),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1.),)
  else:
    raise ValueError(f'Not a recognized optimizer: {cfg.optim.optimizer}')

  return optimizer, theta_dot, opt_state


def get_training_step_function(
    cfg, optimizer, mcmc_step, evaluate_loss, params, opt_state_ckpt, batch_network, batch_network_complex,
    opt_state):
  
  if not optimizer:
    opt_state = None
    step = make_training_step(
        mcmc_step=mcmc_step,
        optimizer_step=make_loss_step(evaluate_loss))
    
  elif isinstance(optimizer, optax.GradientTransformation):
    # optax/optax-compatible optimizer (ADAM, LAMB, ...)
    opt_state = jax.pmap(optimizer.init)(params)

    if cfg.td.time_evolution:
      n_electrons = sum(
        int(round(atom.charge)) for atom in cfg.system.molecule)
      
      if cfg.td.solver == 'psuedoinverse':
        accumulate_samples, conduct_timestep = make_td_opt_update_step_full_solve(
          evaluate_loss, batch_network_complex, cfg.td.iterations_per_timestep,
          cfg.td.regularization.ac, cfg.td.regularization.rc, cfg.td.regularization.type
        )
        step = make_time_evolution_step_low_sample_limit(
        mcmc_step=mcmc_step, optimizer=optimizer,
        accumulate_samples=accumulate_samples, conduct_timestep=conduct_timestep,
        iterations_per_timestep=cfg.td.iterations_per_timestep,
        n_electrons=n_electrons, burn_in_per_timestep=cfg.td.burn_in_per_timestep,
        reset_if_nan=cfg.td.reset_if_nan,
        time_integration=cfg.td.time_integration
      )
      
      elif cfg.td.solver == 'iterative':
    
        logging.info("Making iterative step for time evolution")
        accumulate_samples, conduct_timestep = make_td_opt_update_step(
          evaluate_loss=evaluate_loss, batch_network=batch_network_complex, damping=cfg.td.damping,
          iterations_per_timestep=cfg.td.iterations_per_timestep
        )

        step = make_time_evolution_step(
        mcmc_step=mcmc_step, optimizer=optimizer,
        accumulate_samples=accumulate_samples, conduct_timestep=conduct_timestep,
        iterations_per_timestep=cfg.td.iterations_per_timestep,
        n_electrons=n_electrons,
        cg_iterations=cfg.td.cg_iterations,
        burn_in_per_timestep=cfg.td.burn_in_per_timestep,
        time_integration=cfg.td.time_integration,
        reset_if_nan=cfg.td.reset_if_nan
      )
      else:
        raise NotImplementedError("No other solver implemented for td!")
      """
        if cfg.td.estimate_error:
          err_step = cg_err_estimator(
          mcmc_step=mcmc_step, optimizer=optimizer,
          accumulate_samples=accumulate_samples, conduct_timestep=conduct_timestep,
          iterations_per_timestep=cfg.td.iterations_per_timestep,
          n_electrons=n_electrons,
          cg_iterations=cfg.td.cg_iterations
          )
          step = err_step   # Needs to work in conjunction later, fine for testing
      """
      
    if cfg.optim.optimizer == 'sr' and not cfg.td.time_evolution:  # For now while I'm using optax
      
      if not cfg.td.time_evolution:  # Be careful with this game!
        opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state

      #if opt_state_ckpt is not None and not cfg.td.time_evolution:  # Tricky for td
      #  opt_state = tuple(opt_state_ckpt)
      
      step = make_sr_training_step(
        mcmc_step=mcmc_step,
        optimizer_step=make_sr_opt_update_step(
          evaluate_loss, optimizer, batch_network,
          damping=cfg.optim.sr.damping),
        reset_if_nan=cfg.optim.reset_if_nan)
    
  elif isinstance(optimizer, kfac_jax.Optimizer):
    step = make_kfac_training_step(
        mcmc_step=mcmc_step,
        damping=cfg.optim.kfac.damping,
        optimizer=optimizer,
        reset_if_nan=cfg.optim.reset_if_nan)
  else:
    step = make_training_step(
        mcmc_step=mcmc_step,
        optimizer_step=make_opt_update_step(evaluate_loss, optimizer),
        reset_if_nan=cfg.optim.reset_if_nan)

  return step, opt_state
  

def get_loss_function(
    cfg, log_network, logabs_network, local_energy, log_network_for_loss):
  
  use_complex = cfg.network.get('complex', False)
  overlap_weight = None

  if cfg.optim.objective == 'vmc':
    evaluate_loss = qmc_loss_functions.make_loss(
        log_network if use_complex else logabs_network,
        local_energy,
        clip_local_energy=cfg.optim.clip_local_energy,
        clip_from_median=cfg.optim.clip_median,
        center_at_clipped_energy=cfg.optim.center_at_clip,
        complex_output=use_complex,
        max_vmap_batch_size=cfg.optim.get('max_vmap_batch_size', 0),
    )
  elif cfg.optim.objective == 'wqmc':
    evaluate_loss = qmc_loss_functions.make_wqmc_loss(
        log_network if use_complex else logabs_network,
        local_energy,
        clip_local_energy=cfg.optim.clip_local_energy,
        clip_from_median=cfg.optim.clip_median,
        center_at_clipped_energy=cfg.optim.center_at_clip,
        complex_output=use_complex,
        max_vmap_batch_size=cfg.optim.get('max_vmap_batch_size', 0),
        vmc_weight=cfg.optim.get('vmc_weight', 1.0)
    )
  elif cfg.optim.objective == 'vmc_overlap':
    if not cfg.system.states:
      raise ValueError('Overlap penalty only works with excited states')
    if cfg.optim.overlap.weights is None:
      overlap_weight = tuple([1./(1.+x) for x in range(cfg.system.states)])
      overlap_weight = tuple([x/sum(overlap_weight) for x in overlap_weight])
    else:
      assert len(cfg.optim.overlap.weights) == cfg.system.states
      overlap_weight = cfg.optim.overlap.weights
    evaluate_loss = qmc_loss_functions.make_energy_overlap_loss(
        log_network_for_loss,
        local_energy,
        clip_local_energy=cfg.optim.clip_local_energy,
        clip_from_median=cfg.optim.clip_median,
        center_at_clipped_energy=cfg.optim.center_at_clip,
        overlap_penalty=cfg.optim.overlap.penalty,
        overlap_weight=overlap_weight,
        complex_output=cfg.network.get('complex', False),
        max_vmap_batch_size=cfg.optim.get('max_vmap_batch_size', 0))
  else:
    raise ValueError(f'Not a recognized objective: {cfg.optim.objective}')
  
  return evaluate_loss, overlap_weight

def make_local_energy_functions(cfg, signed_network, charges):

  nspins=cfg.system.electrons
  use_complex = cfg.network.get('complex', False)

  laplacian_method = cfg.optim.get('laplacian', 'default')
  
  if cfg.system.make_local_energy_fn:
    if laplacian_method != 'default':
      raise NotImplementedError(f'Laplacian method {laplacian_method}'
                                'not yet supported by custom local energy fns.')
    if cfg.optim.objective == 'vmc_overlap':
      raise NotImplementedError('Overlap penalty not yet supported for custom'
                                'local energy fns.')
    local_energy_module, local_energy_fn = (
        cfg.system.make_local_energy_fn.rsplit('.', maxsplit=1))
    local_energy_module = importlib.import_module(local_energy_module)
    make_local_energy = getattr(local_energy_module, local_energy_fn)  # type: hamiltonian.MakeLocalEnergy
    local_energy_fn = make_local_energy(
        f=signed_network,
        charges=charges,
        nspins=nspins,
        use_scan=False,
        complex_output=use_complex,
        states=cfg.system.get('states', 0),
        **cfg.system.make_local_energy_kwargs)
  else:
    pp_symbols = cfg.system.get('pp', {'symbols': None}).get('symbols')
    local_energy_fn = hamiltonian.local_energy(
        f=signed_network,
        charges=charges,
        nspins=nspins,
        use_scan=False,
        complex_output=use_complex,
        laplacian_method=laplacian_method,
        states=cfg.system.get('states', 0),
        state_specific=(cfg.optim.objective == 'vmc_overlap'),
        pp_type=cfg.system.get('pp', {'type': 'ccecp'}).get('type'),
        pp_symbols=pp_symbols if cfg.system.get('use_pp') else None,
        cfg=cfg)

  if cfg.optim.get('spin_energy', 0.0) > 0.0:
    # Minimize <H + c * S^2> instead of just <H>
    # Create a new local_energy function that takes the weighted sum of
    # the local energy and the local spin magnitude.
    local_s2_fn = observables.make_s2(
        signed_network,
        nspins=nspins,
        states=cfg.system.states)
    def local_energy_and_s2_fn(params, keys, data):
      local_energy, aux_data = local_energy_fn(params, keys, data)
      s2 = local_s2_fn(params, data, None)
      weight = cfg.optim.get('spin_energy', 0.0)
      if cfg.system.states:
        aux_data = aux_data + weight * s2
        local_energy_and_s2 = local_energy + weight * jnp.trace(s2)
      else:
        local_energy_and_s2 = local_energy + weight * s2
      return local_energy_and_s2, aux_data
    local_energy = local_energy_and_s2_fn
  else:
    local_energy = local_energy_fn

  return local_energy

def conduct_pretraining_hf(
    cfg, t_init, spins, network, params, data, batch_network, device_batch_size,
    sharded_key):
  
  if (
      t_init == 0
      and cfg.pretrain.method == 'hf'
      and cfg.pretrain.iterations > 0
    ):
    nspins = cfg.system.electrons

    if cfg.system.pyscf_mol:
      cfg.system.pyscf_mol.build()
      core_electrons = {
          atom: ecp_table[0]
          for atom, ecp_table in cfg.system.pyscf_mol._ecp.items()  # pylint: disable=protected-access
      }
      ecp = cfg.system.pyscf_mol.ecp
    else:
      ecp = {}
      core_electrons = {}

    if cfg.pretrain.method == 'hf' and cfg.pretrain.iterations > 0:
      hartree_fock = pretrain.get_hf(
          pyscf_mol=cfg.system.get('pyscf_mol'),
          molecule=cfg.system.molecule,
          nspins=nspins,
          restricted=False,
          basis=cfg.pretrain.basis,
          ecp=ecp,
          core_electrons=core_electrons,
          states=cfg.system.states,
          excitation_type=cfg.pretrain.get('excitation_type', 'ordered'))
      # broadcast the result of PySCF from host 0 to all other hosts
      hartree_fock.mean_field.mo_coeff = multihost_utils.broadcast_one_to_all(
          hartree_fock.mean_field.mo_coeff
      )

      pretrain_spins = spins[0, 0]
      batch_orbitals = jax.vmap(
          network.orbitals, in_axes=(None, 0, 0, 0, 0), out_axes=0
      )
      sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
      params, data.positions = pretrain.pretrain_hartree_fock(
          params=params,
          positions=data.positions,
          spins=pretrain_spins,
          atoms=data.atoms,
          charges=data.charges,
          batch_network=batch_network,
          batch_orbitals=batch_orbitals,
          network_options=network.options,
          sharded_key=subkeys,
          electrons=cfg.system.electrons,
          scf_approx=hartree_fock,
          iterations=cfg.pretrain.iterations,
          batch_size=device_batch_size,
          scf_fraction=cfg.pretrain.get('scf_fraction', 0.0),
          states=cfg.system.states,
      )
  return params, data
