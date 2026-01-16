
import jax
from jax import lax
import jax.numpy as jnp
import functools
import chex
import optax
from absl import logging
from typing import Optional, Mapping, Sequence, Tuple, Union
from ferminet import networks
from ferminet import constants



def make_td_opt_update_step_full_solve(
    evaluate_loss, batch_network, 
    iterations_per_timestep=10, ac=1e-5, rc=1e-4):
  """Helper functions for td simulation - see make_time_evolution_step"""

  # Differentiate wrt parameters (argument 0)
  loss_and_grad = jax.value_and_grad(
    evaluate_loss, argnums=0, has_aux=True)
  
  def accumulate_samples_(
      params, key, data, time,local_energies, iteration):
    """Accumulates samples for right side of the equation $X epsilon$"""

    (loss, aux_data), grad = loss_and_grad(params, key, data, time)
    energies = aux_data.local_energy
    batch_size = energies.shape[0]

    local_energies = lax.dynamic_update_slice(
      local_energies,
      aux_data.local_energy,
      (iteration * batch_size,)
    )

    return loss, aux_data, local_energies
  
  def conduct_timestep_(
      params, key, position_arr, data, local_energies):

    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
    
    Np = flat_params.shape[0]
    Ns = position_arr.shape[0]

    """
    data = networks.FermiNetData(
          positions=position_arr,
          spins=jnp.repeat(
            data.spins, repeats=iterations_per_timestep, axis=0),
          atoms=jnp.repeat(
            data.atoms, repeats=iterations_per_timestep, axis=0),
          charges=jnp.repeat(
            data.charges, repeats=iterations_per_timestep, axis=0)
      )  # Might be a better way to do this
    """
    
    O_means = jnp.zeros(Np, dtype=complex)
    grad_vector = jnp.zeros(Np, dtype=complex)
    F = jnp.zeros((Np, Np), dtype=complex)

    local_energies_centered = local_energies - jnp.mean(local_energies)

    #build_O = jax.vmap(lambda s: jax.jvp(f, (params,), (unravel_fn(s),))[1])

    chunk_size = 512
    n_iterations = int(position_arr.shape[0] / chunk_size)  # Assuming this is generally divisible by 512, otherwise there will be annoying cases

    # Building Fisher and grads in chunks
    for i in range(n_iterations):

      def f(params):
        psi = batch_network(
        params, lax.dynamic_slice(position_arr, (i * chunk_size, position_arr.shape[1]), (chunk_size, position_arr.shape[1])),
        data.spins, data.atoms, data.charges)
        return psi
      
      build_O = jax.vmap(lambda s: jax.jvp(f, (params,), (unravel_fn(s),))[1])

      O_s = build_O(jnp.eye(Np))  # You get Np x Ns O here

      F += O_s @ O_s.T
      grad_vector += O_s @ lax.dynamic_slice(
          local_energies_centered,
          (i * chunk_size,),
          (chunk_size,)
          )
      O_means += jnp.sum(O_s, axis=1)
      

    fisher = (F - O_means) / Ns
    grad_vector /= Ns

    
    #O = build_O(jnp.eye(flat_params.shape[0]))  # This is Np x Ns

    #O = jax.jacfwd(f)(flat_params).T

    # Create a regularization function here
    def invert_and_regularize_fisher(fisher):
      rh, s, vh = jnp.linalg.svd(a=fisher, hermitian=True)
      # Smooth cutoff - Medvidovic et al (2023)
      lambda2 = jnp.maximum(ac, rc * jnp.max(s)**2)
      ratio6 = (lambda2 / (s ** 2 + 1e-40)) ** 6
      eff_rank = 1 / (1 + ratio6)
      eff_rank = 1.0

      #eff_rank = jnp.sum(ratio6)
      #f = 1.0 / (1.0 + ratio6)
      #max_eig = jnp.max(s)
      #regularized_term = jnp.max(ac, max_eig**2 * rc )  # TODO: Fix tracer error
      #regularized_term = ac

      #eff_rank = 1/(1 + (regularized_term / s**2) ** 6)  # TODO: Replace this with max eigenvalue weighting - should bring the error down
      s_inv = jnp.where(s < ac, 0, (1/s)) # From Medvidovic et al (2023)

      eff_rank = jnp.sum(jnp.where(s < ac, 1, 0))
      
      SR_inv = (rh * s_inv) @ vh

      return SR_inv, eff_rank, s

    
    #O_alpha = jnp.mean(O, axis=1)  # Per parameter mean over samples
    #O_centered = O - jnp.expand_dims#(O_alpha, axis=1).repeat(Ns, axis=1) # Centering O
    #O_fisher = O / jnp.sqrt(Ns)
    #fisher = (O - jnp.mean(O, axis=1, keepdims=True) / jnp.sqrt(Ns)) @ jnp.conjugate(O / jnp.sqrt(Ns)).T
    #fisher = O_fisher @ jnp.conjugate(O_fisher).T  # Okay this probably makes it real, that's fine

    #### Corrections ####
    #correction_term = O @ jnp.reshape(psis, (Ns, 1))
    #fisher_correction = correction_term @ correction_term.T
    #grad_vector_correction = 1j * (correction_term) * jnp.mean(local_energies)
    ### Ending corrections

    #local_energy_mean = jnp.mean(local_energies)
    #locs_centered = local_energies - local_energy_mean
    #grad_vector = (jnp.conjugate(O) @ jnp.reshape(locs_centered, (
    #  Ns, 1)))
    
    #grad_vector = grad_vector / Ns

    SR_inv, eff_rank, eigs = invert_and_regularize_fisher(fisher)
   
    #grad_vector += grad_vector_correction
    #print(grad_vector.dtype)

    theta_dot = jnp.real(SR_inv @ grad_vector)  # Real param evolution
    #theta_dot = grad_vector
    theta_dot = theta_dot.reshape(Np,)
    
    A = jnp.conjugate(theta_dot) @ fisher @ theta_dot  # Fisher part of residual
    B = jnp.imag(
      jnp.conjugate(grad_vector) * theta_dot)  # Force vector part of residual
    r = fisher @ theta_dot + 1j * grad_vector  # Inversion checker
    r2 = A + B  # Carleo's residual - integrated infidelity
    metrics = {
      "force_vector_residual": A,
      "grad_vector": grad_vector,
      "theta_dot": theta_dot,
      "S_residual": B,
      "linear_system_residual": r,
      "r2": r2,
      "effective_rank": eff_rank,
      "grad_vector": grad_vector,
      "eigenvalues": eigs,
      "energies": local_energies,
      "energies_mean": jnp.mean(local_energies)
    }

    return theta_dot, metrics
  return accumulate_samples_, conduct_timestep_



def make_time_evolution_step_low_sample_limit(
    mcmc_step,
    optimizer,
    accumulate_samples,
    conduct_timestep,
    iterations_per_timestep,
    n_electrons,
    burn_in_per_timestep=0,
    reset_if_nan: bool = True,
    time_integration = 'rk2',
):
  """Makes time evolution step from Carleo's paper (Nys 2024) by fitting the 
  parameter update to time evolution of the state. For each dt, accumulates 
  samples for (iterations_per_timestep) and then gets $theta_{dot}$ as
  per $XX^T theta_dot = X epsilon$. The resulting $theta_{dot}$ 
  out of make_evolution_step is the final vector after dt.
  RK2 is implemented over the dt calculation, which can be
  adjusted as needed to deal with time integration.
  """
  @functools.partial(constants.pmap,
                      in_axes=(0, 0, 0, None, 0, 0),
                      donate_argnums=(0, 1, 2))
  def step(
    data: networks.FermiNetData,
    params: networks.ParamTree,
    opt_state: optax.OptState,
    time: float,
    key: chex.PRNGKey,
    mcmc_width: jnp.ndarray,
):
    """
    A full update iteration with integration: MCMC steps + optimization
    For a single timestep
    """
    # MCMC loop

    batch_size = data.positions.shape[0]
    mcmc_key, loss_key = jax.random.split(key, num=2)
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(
      params)
    n_params = flat_params.shape[0]
    
    spins, atoms, charges = data.spins, data.atoms, data.charges

    def accumulate_samples_inner_fn(i, carry):
      loss, aux_data, position_arr, data, local_energies, key, time = carry
      mcmc_key, key = jax.random.split(key)

      positions = lax.dynamic_slice(
          position_arr,
          (i * batch_size, 0),
          (batch_size, position_arr.shape[1])
      )

      accumulated_data = networks.FermiNetData(
          positions=positions,
          spins=spins,
          atoms=atoms,
          charges=charges,
      )

      data, pmove = mcmc_step(
        params, data, mcmc_key, mcmc_width)  \
          # TODO: Replace accumulated data with normal data here

      loss_key, key = jax.random.split(key)
      
      loss, aux_data, grad_vector = accumulate_samples(
        params, loss_key, data, time, local_energies, i)
      
      position_arr = lax.dynamic_update_slice(
          position_arr,
          data.positions,
          (batch_size * i, 0)   # (row_offset, col_offset)
      )

      return loss, aux_data, position_arr, data, local_energies, key, time

    logging.info(f"Starting sample accumulation for timestep")

    def rk2_inner_fn(params, key, data, time):
      
      local_energies = jnp.zeros(iterations_per_timestep * batch_size, dtype=complex)

      position_arr = jnp.zeros(
        (iterations_per_timestep * batch_size, n_electrons*3)
      )

      mcmc_key, key = jax.random.split(key)

      data, pmove = mcmc_step(
        params, data, mcmc_key, mcmc_width)
      
      loss_key, key = jax.random.split(key)
      
      loss, aux_data, local_energies = accumulate_samples(
        params, loss_key, data, time, local_energies, 0)
      
      position_arr = lax.dynamic_update_slice(
          position_arr,
          data.positions,
          (batch_size, 0)   # (row_offset, col_offset)
      )
      
      init_carry = (
        loss, aux_data, position_arr, data, local_energies, key, time)
      loss, aux_data, position_arr, data, local_energies, key, time = lax.fori_loop(
          0, iterations_per_timestep, accumulate_samples_inner_fn, init_carry
      )

      theta_dot, metrics = conduct_timestep(
        params, key, position_arr, data, local_energies)
      
      return data, pmove, loss, aux_data, theta_dot, metrics
      
    logging.info("Starting integration first step")
    data, pmove, loss, aux_data, theta_dot_1, metrics = constants.pmean(
      rk2_inner_fn(params, key, data, time))
    theta_dot_1 = theta_dot_1

    if time_integration == 'rk2':
      #theta_dot_1 = -1j * grad_vector_1

      half_updates, _ = optimizer.update(
        unravel_fn(theta_dot_1 * 0.5), opt_state, params)
      params_mid = optax.apply_updates(params, half_updates)

      logging.info("Starting RK2 second step")
      data, pmove, loss, aux_data, theta_dot_2, metrics = constants.pmean(
        rk2_inner_fn(params_mid, key, data, time))
      theta_dot_2 = theta_dot_2
      theta_dot = theta_dot_2
    else:
      theta_dot = theta_dot_1

    logging.info("Updating params")
    # Step 4: Full step update with k2
    
    updates, new_state = optimizer.update(
      theta_dot, opt_state, params)
      
    new_params = optax.apply_updates(
      params, unravel_fn(updates))

    if reset_if_nan:
      new_params = jax.lax.cond(jnp.isnan(loss),
                                lambda _: params,
                                lambda _: new_params,
                                operand=None)
      """            
      new_state = jax.lax.cond(jnp.isnan(loss),
                               lambda _: opt_state,
                               lambda _: new_state,
                               operand=None)
      """
    logging.info("Completed timestep")

    
    return data, new_params, new_state, \
      loss, aux_data, pmove, theta_dot, metrics


  return step