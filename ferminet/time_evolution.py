
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


def make_td_opt_update_step(
    evaluate_loss, batch_network, 
    damping=1e-2, iterations_per_timestep=10):
  """Helper functions for td simulation - see make_time_evolution_step"""

  # Differentiate wrt parameters (argument 0)
  loss_and_grad = jax.value_and_grad(
    
    evaluate_loss, argnums=0, has_aux=True)
  
  def accumulate_samples(
      params, key, data, grad_vector):
    """Accumulates samples for right side of the equation $X\epsilon$"""

    (loss, aux_data), grad = loss_and_grad(params, key, data)
    flat_grads, unravel_fn = jax.flatten_util.ravel_pytree(grad)
    energies = aux_data.local_energy - loss
    batch_size = energies.shape[0]

    return loss, aux_data, grad_vector + flat_grads / iterations_per_timestep
  
  def conduct_timestep(
      params, key, data, grad_vector):
    """
    Solves for $XX^T \theta_dot = X\epsilon$
    $X\epsilon$ is accumulated in the grad_vector
    data passed in over the whole range of samples
    Solves using an iterative solver for better estimates of $\theta_dot$
    """

    _, unravel_fn = jax.flatten_util.ravel_pytree(params)
    batch_size = data.positions.shape[1] // data.spins.shape[0]
  
    def f(params):
      psi = batch_network(
        params, data.positions, data.spins, data.atoms, data.charges)
      return psi
  
    jvp_func = lambda x: jax.linearize(f, params)[1](
      unravel_fn(x))
    
    vjp_func = lambda v: jax.flatten_util.ravel_pytree(
      jax.vjp(f, params)[1](v))[0]
    
    def fisher_matmul(
        v, centre_gradients=True, damping=damping):
      
      log_psi_jac_v = jvp_func(v)
      update_vector = vjp_func(log_psi_jac_v / batch_size)
        
      if centre_gradients:
          update_vector -= jnp.mean(update_vector)
      
      update_vector += damping * v
      return update_vector
    
    x0 = grad_vector  # Using loss grads as guess        
    theta_dot = jax.scipy.sparse.linalg.cg(
      fisher_matmul, grad_vector, x0=x0, maxiter=10000)[0]
          
    return theta_dot
  return accumulate_samples, conduct_timestep



def make_time_evolution_step(
    mcmc_step,
    optimizer,
    accumulate_samples,
    conduct_timestep,
    iterations_per_timestep,
    reset_if_nan: bool = False,
    n_electrons=2
):
  """Makes time evolution step from Carleo's paper (Nys 2024) by fitting the 
  parameter update to time evolution of the state. For each dt, accumulates 
  samples for (iterations_per_timestep) and then gets $\\theta_{dot}$ as
  per $XX^T \\theta_dot = X \\epsilon$. The resulting $\\theta_{dot}$ 
  out of make_evolution_step is the final vector after dt.
  RK2 is implemented over the dt calculation, which can be
  adjusted as needed to deal with time integration.
  """
  #TODO: Manage the pmapping here and the arguments that should
  ##     be passed.
  @functools.partial(constants.pmap,
                    donate_argnums=(0, 1, 2))
  def step(
    data: networks.FermiNetData,
    params: networks.ParamTree,
    opt_state: Optional[optax.OptState],
    key: chex.PRNGKey,
    mcmc_width: jnp.ndarray,
    time_integration_method = 'rk2'
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
    print(data.spins.shape)

    spins, atoms, charges = data.spins, data.atoms, data.charges

    def accumulate_samples_inner_fn(i, carry):
      position_arr, grad_vector, key = carry
      mcmc_key, loss_key = jax.random.split(key)

      positions = lax.dynamic_slice(
          position_arr,
          (i * batch_size, 0),           # start index (row_offset, col_offset)
          (batch_size, position_arr.shape[1])   # slice shape
      )

      accumulated_data = networks.FermiNetData(
          positions=positions,
          spins=spins,
          atoms=atoms,
          charges=charges,
      )

      data, pmove = mcmc_step(
        params, accumulated_data, mcmc_key, mcmc_width)
      
      _, _, grad_vector = accumulate_samples(
        params, key, data, grad_vector)
      
      position_arr = lax.dynamic_update_slice(
          position_arr,
          data.positions,
          (batch_size * i, 0)   # (row_offset, col_offset)
      )

      return position_arr, grad_vector, key

    logging.info(f"Starting sample accumulation for timestep")

    def rk2_inner_fn(params, key, data):

      grad_vector = jnp.zeros(
        (n_params))
      position_arr = jnp.zeros(
        (iterations_per_timestep * batch_size, n_electrons*3)
      )

      data, pmove = mcmc_step(
        params, data, mcmc_key, mcmc_width)
      
      position_arr = lax.dynamic_update_slice(
          position_arr,
          data.positions,
          (batch_size, 0)   # (row_offset, col_offset)
      )
      
      init_carry = (position_arr, grad_vector, key)
      position_arr, final_grad_vector, final_key = lax.fori_loop(
          0, iterations_per_timestep, accumulate_samples_inner_fn, init_carry
      )

      data = networks.FermiNetData(
          positions=position_arr,
          spins=jnp.repeat(
            data.spins, repeats=iterations_per_timestep, axis=0),
          atoms=jnp.repeat(
            data.atoms, repeats=iterations_per_timestep, axis=0),
          charges=jnp.repeat(
            data.charges, repeats=iterations_per_timestep, axis=0)
      )

      theta_dot = conduct_timestep(
        params, key, data, final_grad_vector)
      
      return theta_dot

    if time_integration_method == 'rk2':
      
      logging.info("Starting RK2 first step")
      theta_dot_1 = -1j * constants.pmean(
        rk2_inner_fn(params, key, data))

      half_updates, _ = optimizer.update(
        unravel_fn(theta_dot_1 * 0.5), opt_state, params)
      params_mid = optax.apply_updates(params, half_updates)

      logging.info("Starting RK2 second step")
      theta_dot_2 = -1j * constants.pmean(
        rk2_inner_fn(params_mid, key, data))

      logging.info("Updating params")
      # Step 4: Full step update with k2
      updates, opt_state = optimizer.update(
        unravel_fn(theta_dot_2), opt_state, params)
      new_params = optax.apply_updates(params, updates)

      logging.info("Evaluating final energy")
      #TODO: The last energy move is to be figured out - is it at that timestep?
      data, pmove = mcmc_step(
        new_params, data, mcmc_key, mcmc_width)
      loss, aux_data, _ = accumulate_samples(params, key, data, jnp.zeros((n_params)))

    else:
      raise NotImplementedError(
        "Alternate time integration has not yet been implemented!")
    
    return data, new_params, opt_state, \
      loss, aux_data, pmove

  return step


"""
def cg_err_estimator(mcmc_step, accumulate_samples, conduct_timestep, iterations_per_timestep, n_electrons):
    # The same timestep - does 10 CG steps and calculates variance in the update vector and in the loss grads
    # Then changes the samples accumulated per timestep and does the same again
    # Then changes the max iterations and does it again

    # We don't actually get the inverse matrice we get the solved vector (N_p), so we have to test on that instead
    # Check scale of loss grads changing and then that should give us an idea of the scale of the variance of the estimated inverse of the metric
    
    # Variance per iteration (200 v 2000 v 20000) - 10 iterations
    # Mean value for each (1e2, 1e3, 1e4)
    # Variance for different samples accumulated (5e3, 5e4, 5e5)
  def step(
    data: networks.FermiNetData,
    params: networks.ParamTree,
    opt_state: Optional[optax.OptState],
    key: chex.PRNGKey,
    mcmc_width: jnp.ndarray,
    time_integration_method = 'rk2'
  ):
    
    A full update iteration with integration: MCMC steps + optimization
    For a single timestep
    # MCMC loop

    batch_size = data.positions.shape[0]
    mcmc_key, loss_key = jax.random.split(key, num=2)
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(
      params)
    n_params = flat_params.shape[0]
    spins, atoms, charges = data.spins, data.atoms, data.charges

    def accumulate_samples_inner_fn(i, carry):
      position_arr, grad_vector, key = carry
      mcmc_key, loss_key = jax.random.split(key)

      positions = lax.dynamic_slice(
          position_arr,
          (i * batch_size, 0),           # start index (row_offset, col_offset)
          (batch_size, position_arr.shape[1])   # slice shape
      )

      accumulated_data = networks.FermiNetData(
          positions=positions,
          spins=spins,
          atoms=atoms,
          charges=charges,
      )

      data, pmove = mcmc_step(
        params, accumulated_data, mcmc_key, mcmc_width)
      
      _, _, grad_vector = accumulate_samples(
        params, key, data, grad_vector)
      
      position_arr = lax.dynamic_update_slice(
          position_arr,
          data.positions,
          (512 * i, 0)   # (row_offset, col_offset)
      )

      return position_arr, grad_vector, key

    logging.info(f"Starting sample accumulation for timestep")

    
    def timestep_variance(i, carry):
      ## Variance of gradient update at time t with fixed data - consistency of inverse
      data, grad_vector, theta_dot_arr, key = carry
      theta_dot = conduct_timestep(
        params, key, data, grad_vector)
      
      theta_dot_arr = lax.dynamic_update(
        theta_dot_arr,
        theta_dot,
        i
      )

      return data, grad_vector, theta_dot_arr, key


    def get_per_timestep_update_variance(params, key, data):

      grad_vector = jnp.zeros(
        (n_params))
      position_arr = jnp.zeros(
        (iterations_per_timestep * batch_size, n_electrons*3)
      )

      data, pmove = mcmc_step(
        params, data, mcmc_key, mcmc_width)
      
      position_arr = lax.dynamic_update_slice(
          position_arr,
          data.positions,
          (512, 0)   # (row_offset, col_offset)
      )
      
      init_carry = (position_arr, grad_vector, key)
      position_arr, final_grad_vector, final_key = lax.fori_loop(
          0, iterations_per_timestep, accumulate_samples_inner_fn, init_carry
      )

      # Full positions, final grad vector
      data = networks.FermiNetData(
          positions=position_arr,
          spins=jnp.repeat(
            data.spins, repeats=iterations_per_timestep, axis=0),
          atoms=jnp.repeat(
            data.atoms, repeats=iterations_per_timestep, axis=0),
          charges=jnp.repeat(
            data.charges, repeats=iterations_per_timestep, axis=0)
      )
       
      per_timestep_variance_iterations = 10
      # That's for iterations_per_timestep samples
      theta_dot_arr = jnp.zeros(
        (per_timestep_variance_iterations, n_params))
      init_carry = (data, grad_vector, theta_dot_arr, key)

      data, grad_vector, theta_dot_arr, key = lax.fori_loop(0, per_timestep_variance_iterations, timestep_variance, init_carry)

      #theta_dot = conduct_timestep(
      #  params, key, data, final_grad_vector)
      
      return theta_dot_arr

    # Okay so at the top level we need
    # Per timestep variance for
    # 1. Differing n_samples (change iterations per timestep) (5e3, 5e4, 5e5)
    # 2. Differing maxiter of cg (change max_iterations) (2e1, 2e2, 2e3, 2e4)
    # Let's do it at the same timestep - it should be irrespective of that anyway (think deeply about this)

    # what it returns is theta_dot_mean, theta_dot_variance, loss_mean, loss_variance - so we will have (7 copies of each for n_samples and maxiter comparision)

    # We can make different step functions instead of doing the loop - just change max iter, and then we can put them in a list - and call them individually and sum contributions or something - n_samples, and n_iterations
    # Can just log the values directly - its at the same timestep so they should be the same or give us some relationship to exploit

    return data, new_params, opt_state, \
      loss, aux_data, pmove
  pass
"""