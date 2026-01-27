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
    iterations_per_timestep,
     ac, rc, regularization
):
  """
  Helper functions for td opt update using full psuedoinverse.
  Accumulates fisher (accumulate_samples) over iterations_per_timestep
  Inverts and obtain parameter velocities in conduct_timestep
  """

  loss_and_grad = jax.value_and_grad(
    evaluate_loss, argnums=0, has_aux=True
  )
  def accumulate_samples(
      params, key, data, time, grad_vector,
      fisher, O_means
  ):
    """
    Forms intermidiary Oalpha and/or fisher as samples are accumulated
    """

    (loss, aux_data), grad = loss_and_grad(params, key, data, time)
    flat_grads_, unravel_fn = jax.flatten_util.ravel_pytree(grad)
    energies = aux_data.local_energy
    psi = batch_network(
        params, data.positions, data.spins, data.atoms, data.charges)
    
    n_params = flat_grads_.shape[0]
    batch_size = energies.shape[0]
    Ns_minibatch = data.positions.shape[0]

    def compute_SR_chunked(params, data, chunk_size=512):
      """Compute S = O^T O and mean(O) using chunks."""
        
      flat_params, unravel = jax.flatten_util.ravel_pytree(params)
      Np = flat_params.shape[0]
      O_minibatch = jnp.zeros((Np, Ns_minibatch), dtype=complex)

      # Split Jacobian calculation over minibatches to prevent memory blowup
      for start in range(0, Ns_minibatch, chunk_size):
          end = min(start + chunk_size, Ns_minibatch)

          pos = data.positions[start:end]
          spins = data.spins[start:end]
          atoms = data.atoms[start:end]
          charges = data.charges[start:end]

          f = lambda p: batch_network(
            p, pos, spins, atoms, charges)

          O_chunk = jax.jacfwd(f)(params)
          O_chunk, unravel_fn = jax.flatten_util.ravel_pytree(O_chunk)  # converting to array          
          O_chunk = O_chunk.reshape(n_params, chunk_size)

          O_minibatch = lax.dynamic_update_slice(
            O_minibatch,
            O_chunk,
            (0, start)   # (row_offset, col_offset)
            )
      return O_minibatch

    O_minibatch = compute_SR_chunked(
      params, data, chunk_size=512)
    
    O_means += jnp.sum(O_minibatch, axis=1)
    fisher += O_minibatch @ O_minibatch.T / Ns_minibatch
    
    grad_vector += (O_minibatch @ (
      aux_data.local_energy - loss).reshape(Ns_minibatch, 1)) / Ns_minibatch  # Works since its a sum
    
    return loss, aux_data, grad_vector, fisher, O_means


  def conduct_timestep(
      params, grad_vector, fisher, O_means, n_samples
  ):
    
    # Create a regularization function here
    def invert_and_regularize_fisher(fisher):
      rh, s, vh = jnp.linalg.svd(a=fisher, hermitian=True)
      # Smooth cutoff - Medvidovic et al (2023)

      if regularization == 'spectrum':
        max_eig = jnp.max(s)
        lambda_2 = jnp.maximum(ac, rc * max_eig)
        f = 1 / (1 + (lambda_2 / s) ** 6)
        s_inv =  f/s
        eff_rank = jnp.sum(f)
      elif regularization == 'cutoff':
        lambda2 = jnp.maximum(ac, rc * jnp.max(s) ** 2)
        ratio6 = (lambda2 / (s ** 2 + 1e-40)) ** 6
        eff_rank = 1 / (1 + ratio6)
        eff_rank = 1.0
        s_inv = jnp.where(s < ac, 0, (1/s)) # From Medvidovic et al (2023)
        eff_rank = jnp.sum(jnp.where(s < ac, 1, 0))
      else:
        raise NotImplementedError("This regularization is not implemented!")

      SR_inv = (rh * s_inv) @ vh

      return SR_inv, eff_rank, s

    
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
    n_params = flat_params.shape[0]

    """Corrections
    #print(O.shape)
    correction_term = O @ jnp.reshape(psis, (Ns, 1))
    #print(correction_term.shape)
    fisher_correction = correction_term @ correction_term.T
    grad_vector_correction = 1j * (correction_term) * jnp.mean(local_energies)
    #print(grad_vector_correction.shape)
    #print(fisher_correction.shape)

    #fisher += fisher_correction
    """

    fisher = fisher - (O_means ** 2) / n_samples
    SR_inv, eff_rank, eigs = invert_and_regularize_fisher(fisher)

    theta_dot = jnp.real(SR_inv @ grad_vector.reshape(n_params, 1))  # Real param evolution
    #theta_dot = grad_vector
    theta_dot = theta_dot.reshape(n_params)
    
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
    }

    return theta_dot, metrics
  return accumulate_samples, conduct_timestep



def make_time_evolution_step_low_sample_limit(
    mcmc_step,
    optimizer,
    accumulate_samples,
    conduct_timestep,
    iterations_per_timestep,
    n_electrons,
    burn_in_per_timestep,
    reset_if_nan: bool = False,
    time_integration = 'rk2'
):
  """Makes time evolution step from Carleo's paper (Nys 2024) by fitting the 
  parameter update to time evolution of the state. For each dt, accumulates 
  samples for (iterations_per_timestep) and then gets $\\theta_{dot}$ as
  per $XX^T \\theta_dot = X \\epsilon$. The resulting $\\theta_{dot}$ 
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
    mcmc_width: jnp.ndarray
):
    """
    A full update iteration with integration: MCMC steps + optimization
    For a single timestep
    """
    # MCMC loop

    batch_size = data.positions.shape[0]
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(
      params)
    n_params = flat_params.shape[0]
    n_samples = batch_size * iterations_per_timestep

    def accumulate_samples_inner_fn(i, carry):
      loss, aux_data, data, grad_vector, fisher, O_means, \
        time, key = carry
      mcmc_key, new_key = jax.random.split(key)

      data, pmove = mcmc_step(
        params, data, mcmc_key, mcmc_width)
      
      loss_key, new_key = jax.random.split(key)
      
      loss, aux_data, grad_vector, fisher, O_means = accumulate_samples(
        params, loss_key, data, time, grad_vector, fisher, O_means)
      
      return loss, aux_data, data, grad_vector,\
        fisher, O_means, time, new_key

    logging.info(f"Starting sample accumulation for timestep")

    def rk2_inner_fn(params, key, data, time):
      
      grad_vector = jnp.zeros((n_params, 1), dtype=complex)
      fisher = jnp.zeros((n_params, n_params), dtype=complex)
      O_means = jnp.zeros(n_params, dtype=complex)

      mcmc_key, key = jax.random.split(key, num=2)

      data, pmove = mcmc_step(
        params, data, mcmc_key, mcmc_width)
      
      loss_key, key = jax.random.split(key, num=2)

      loss, aux_data, grad_vector, fisher, O_means = accumulate_samples(
        params, loss_key, data, time, grad_vector, fisher, O_means)
      
      
      init_carry = (
        loss, aux_data, data, grad_vector, fisher, O_means, time, key)
      loss, aux_data, data, grad_vector, fisher, O_means, time, key = lax.fori_loop(
          1, iterations_per_timestep, accumulate_samples_inner_fn, init_carry
      )
      
      theta_dot, metrics = conduct_timestep(
        params, grad_vector, fisher, O_means, n_samples)
      
      return data, pmove, loss, aux_data, theta_dot, metrics

    logging.info("Starting integration first step")
    data, pmove, loss, aux_data, theta_dot_1, metrics = constants.pmean(
      rk2_inner_fn(params, key, data, time))
    theta_dot_1 =  theta_dot_1  # Real param evolution

    if time_integration == 'rk2':

      half_updates, _ = optimizer.update(
        unravel_fn(theta_dot_1 * 0.5), opt_state, params)
      params_mid = optax.apply_updates(params, half_updates)

      logging.info("Starting RK2 second step")
      data, pmove, _, _, theta_dot_2, metrics = constants.pmean(
        rk2_inner_fn(params_mid, key, data, time))
      theta_dot_2 = theta_dot_2  # Removing 1j as per the McLachan varaiaitonal principle
      theta_dot = theta_dot_2
    
    else:
      theta_dot = theta_dot_1

    logging.info("Updating params")
    
    
    updates, new_state = optimizer.update(
      theta_dot, opt_state, params)
    new_params = optax.apply_updates(
      params, unravel_fn(updates))
    
    if not opt_state:
      opt_state = {}

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
