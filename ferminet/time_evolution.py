
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
     ac=1e-5, rc=1e-4
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
      fisher
  ):
    
    # Getting loss and loss grads
    (loss, aux_data), grad = loss_and_grad(params, key, data, time)
    flat_grads, unravel_fn = jax.flatten_util.ravel_pytree(grad)
    energies = aux_data.local_energy - loss
    batch_size = energies.shape[0]
    total_batch_size = batch_size * iterations_per_timestep

    # Getting the Fisher
    def compute_SR_chunked(params, data, chunk_size=256):
        """Compute S = O^T O and mean(O) using chunks."""
        
        N = data.positions.shape[0]

        flat_params, unravel = jax.flatten_util.ravel_pytree(params)
        Np = flat_params.shape[0]

        S = jnp.zeros((Np, Np), dtype=float)
        O_mean_accum = jnp.zeros((Np,), dtype=float)

        total_samples = 0

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            n_chunk = end - start

            pos = data.positions[start:end]
            spins = data.spins[start:end]
            atoms = data.atoms[start:end]
            charges = data.charges[start:end]

            J_chunk = jax.jacrev(
                lambda p: batch_network(p, pos, spins, atoms, charges)
            )(params)

            per_leaf = [
                jnp.reshape(leaf, (leaf.shape[0], -1))
                for leaf in jax.tree_util.tree_leaves(J_chunk)
            ]
            O_chunk = jnp.concatenate(per_leaf, axis=1)

            O_mean_accum += jnp.sum(O_chunk, axis=0)

            S += O_chunk.T @ O_chunk

            total_samples += n_chunk

        # Compute final mean
        O_mean = O_mean_accum / total_samples

        return S, O_mean, unravel

    S, O_mean, unravel = compute_SR_chunked(
      params, data, chunk_size=256)
    
    S = (S - data.positions.shape[0] * jnp.outer(
      O_mean, O_mean)) / total_batch_size  # TODO: Double check where the centering happens

    grad_vector += flat_grads / iterations_per_timestep
    fisher += S

    return loss, aux_data, grad_vector, fisher 

  def conduct_timestep(
      params, key, data, grad_vector, fisher,  batch_size
  ):
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
    n_params = flat_params.shape[0]

    _, s, vh = jnp.linalg.svd(a=fisher, hermitian=True)

    # Smooth cutoff - Medvidovic et al (2023)
    lambda2 = jnp.max(s) ** 2
    log_ratio6 = 6.0 * (jnp.log(lambda2) - jnp.log(s + 1e-40))
    ratio6 = jnp.exp(jnp.clip(log_ratio6, -50, 50))   # safer exponent
    
    eff_rank = jnp.mean(ratio6)
    f = 1.0 / (1.0 + ratio6)
    s_inv = f / (s + 1e-40)  # Prevent blowup

    # Regularizing
    SR_inv = (vh.T * s_inv) @ vh

    # Solve SR * theta_dot = grad_vector
    theta_dot = SR_inv @ grad_vector
    
    r2 = jnp.conjugate(theta_dot) @ fisher @ theta_dot + jnp.imag(
      jnp.conjugate(grad_vector) * theta_dot) # residual - integrated infidelity

    return theta_dot, r2, eff_rank
  return accumulate_samples, conduct_timestep


def make_td_opt_update_step(
    evaluate_loss, batch_network, 
    damping=1e-2, iterations_per_timestep=10):
  """Helper functions for td simulation - see make_time_evolution_step"""

  # Differentiate wrt parameters (argument 0)
  loss_and_grad = jax.value_and_grad(
    evaluate_loss, argnums=0, has_aux=True)
  
  def accumulate_samples(
      params, key, data, time, grad_vector):
    """Accumulates samples for right side of the equation $X epsilon$"""

    (loss, aux_data), grad = loss_and_grad(params, key, data, time)
    flat_grads, unravel_fn = jax.flatten_util.ravel_pytree(grad)
    energies = aux_data.local_energy - loss
    batch_size = energies.shape[0]

    return loss, aux_data, grad_vector + flat_grads / iterations_per_timestep
  
  def conduct_timestep(
      params, key, data, grad_vector, cg_iterations, batch_size,
      iterative=False):
    """
    Solves for $XX^T theta_dot = X epsilon$
    $X epsilon$ is accumulated in the grad_vector
    data passed in over the whole range of samples
    Solves using an iterative solver for better estimates of $theta_dot$
    """

    eigs = None
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
    n_params = flat_params.shape[0]
  
    def f(params):
      psi = batch_network(
        params, data.positions, data.spins, data.atoms, data.charges)
      return psi
    
    f = jax.checkpoint(f)
  
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
    
    if iterative:
      x0 = grad_vector  # Using loss grads as guess        
      theta_dot = jax.scipy.sparse.linalg.cg(
        fisher_matmul, grad_vector, x0=x0,
        maxiter=cg_iterations)[0]
      
    else:
      """
      def grad_i(params, i):
        return jax.flatten_util.ravel_pytree(
            jax.grad(lambda p: f(p)[i])(params)
        )[0]

      # Vectorize across samples → shape [N, n_params]
      O = jax.vmap(
        lambda i: grad_i(params, i))(
          jnp.arange(data.positions.shape[0]))

      O = O - jnp.mean(O, axis=0)
      O = (O.T @ O) / O.shape[0]

      # Solve directly
      _, s, vh = jnp.linalg.svd(a=O, hermitian=True)
      max_eig = jnp.max(s) ** 2
      s = jnp.where(s < 1e-3, (1/s**2)*(1/(1 + ((max_eig * 1e-4) / s**2) ** 6)), (1/s**2))  # From Medvidovic et al (2023)
      # Reconstruct SR^{-1}
      SR_inv = (vh.T * s) @ vh

      # Solve SR * theta_dot = grad_vector
      theta_dot = SR_inv @ grad_vector
      del O
      """

      def compute_SR_chunked(params, data, chunk_size=1024):
        """Compute S = O^T O and mean(O) using chunks."""
        
        N = data.positions.shape[0]

        flat_params, unravel = jax.flatten_util.ravel_pytree(params)
        Np = flat_params.shape[0]

        S = jnp.zeros((Np, Np), dtype=float)
        O_mean_accum = jnp.zeros((Np,), dtype=float)

        total_samples = 0

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            n_chunk = end - start

            # Slice data
            pos = data.positions[start:end]
            spins = data.spins[start:end]
            atoms = data.atoms[start:end]
            charges = data.charges[start:end]

            # J_chunk is pytree: each leaf is (chunk, ...)
            J_chunk = jax.jacrev(
                lambda p: batch_network(p, pos, spins, atoms, charges)
            )(params)

            # Flatten leaves across param dims → (chunk, Np)
            per_leaf = [
                jnp.reshape(leaf, (leaf.shape[0], -1))
                for leaf in jax.tree_util.tree_leaves(J_chunk)
            ]
            O_chunk = jnp.concatenate(per_leaf, axis=1)

            # Accumulate mean
            O_mean_accum += jnp.sum(O_chunk, axis=0)

            # Accumulate S = O^T O
            S += O_chunk.T @ O_chunk

            total_samples += n_chunk

        # Compute final mean
        O_mean = O_mean_accum / total_samples

        return S, O_mean, unravel

      S, O_mean, unravel = compute_SR_chunked(
        params, data, chunk_size=256)
      O = S - data.positions.shape[0] * jnp.outer(O_mean, O_mean)
      #O = O - jnp.mean(O, axis=0)
      O = (O.T @ O) / O.shape[0]

      # Solve directly
      _, s, vh = jnp.linalg.svd(a=O, hermitian=True)

      max_eig = jnp.max(s) ** 2
      s = jnp.where(s < 1e-3, (1/s**2)*(1/(1 + ((max_eig * 1e-4) / s**2) ** 6)), (1/s**2))
      # Reconstruct SR^{-1}
      SR_inv = (vh.T * s) @ vh

      # Solve SR * theta_dot = grad_vector
      theta_dot = SR_inv @ grad_vector
      del O
    
    
    r2 = jnp.conjugate(theta_dot) * fisher_matmul(
      theta_dot) + jnp.imag(jnp.conjugate(grad_vector) * theta_dot) # residual - integrated infidelity
    

    return theta_dot, r2, s
  return accumulate_samples, conduct_timestep



def make_time_evolution_step(
    mcmc_step,
    optimizer,
    accumulate_samples,
    conduct_timestep,
    iterations_per_timestep,
    n_electrons,
    cg_iterations,
    reset_if_nan: bool = False,
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
    opt_state: Optional[optax.OptState],
    time: float,
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
    
    spins, atoms, charges = data.spins, data.atoms, data.charges

    def accumulate_samples_inner_fn(i, carry):
      loss, aux_data, position_arr, grad_vector, key, time = carry
      mcmc_key, loss_key = jax.random.split(key)

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
        params, accumulated_data, mcmc_key, mcmc_width)  \
          # TODO: Replace accumulated data with normal data here
      
      loss, aux_data, grad_vector = accumulate_samples(
        params, key, data, time, grad_vector)
      
      position_arr = lax.dynamic_update_slice(
          position_arr,
          data.positions,
          (batch_size * i, 0)   # (row_offset, col_offset)
      )

      return loss, aux_data, position_arr, grad_vector, key, time

    logging.info(f"Starting sample accumulation for timestep")

    def rk2_inner_fn(params, key, data, time):
      
      grad_vector = jnp.zeros((n_params))
      position_arr = jnp.zeros(
        (iterations_per_timestep * batch_size, n_electrons*3)
      )

      data, pmove = mcmc_step(
        params, data, mcmc_key, mcmc_width)
      
      loss, aux_data, grad_vector = accumulate_samples(
        params, key, data, time, grad_vector)
      
      position_arr = lax.dynamic_update_slice(
          position_arr,
          data.positions,
          (batch_size, 0)   # (row_offset, col_offset)
      )
      
      init_carry = (
        loss, aux_data, position_arr, grad_vector, key, time)
      loss, aux_data, position_arr, final_grad_vector, final_key, time = lax.fori_loop(
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

      theta_dot, r2 = None, None
      
      theta_dot, r2, eigs = conduct_timestep(
        params, key, data, final_grad_vector,
        cg_iterations, batch_size)
      
      return loss, aux_data, theta_dot, r2, final_grad_vector, position_arr, pmove, eigs

    if time_integration_method == 'rk2':
      
      logging.info("Starting RK2 first step")
      loss, aux_data, theta_dot_1, _, _, _, pmove, eigs = constants.pmean(
        rk2_inner_fn(params, key, data, time))
      theta_dot_1 = -1j * theta_dot_1
      #theta_dot_1 = -1j * grad_vector_1

      half_updates, _ = optimizer.update(
        unravel_fn(theta_dot_1 * 0.5), opt_state, params)
      params_mid = optax.apply_updates(params, half_updates)

      logging.info("Starting RK2 second step")
      loss, aux_data, theta_dot_2, r2, _, _, pmove, eigs = constants.pmean(
        rk2_inner_fn(params_mid, key, data, time))
      theta_dot_2 = -1j * theta_dot_2
      #theta_dot_2 = -1j * grad_vector_2

      #theta_dot_2 = theta_dot_1

      logging.info("Updating params")
      # Step 4: Full step update with k2
      updates, opt_state = optimizer.update(
        unravel_fn(theta_dot_2), opt_state, params)
      params = optax.apply_updates(params, updates)

      #logging.info("Evaluating final energy")
      """
      loss, aux_data, _ = accumulate_samples(
        new_params, key, data, time, jnp.zeros((n_params)))
      """

    else:
      raise NotImplementedError(
        "Alternate time integration has not yet been implemented!")
    
    return data, params, opt_state, \
      loss, aux_data, pmove, theta_dot_2, r2, eigs

  return step


def make_time_evolution_step_low_sample_limit(
    mcmc_step,
    optimizer,
    accumulate_samples,
    conduct_timestep,
    iterations_per_timestep,
    n_electrons,
    reset_if_nan: bool = False,
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
    opt_state: Optional[optax.OptState],
    time: float,
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
    
    spins, atoms, charges = data.spins, data.atoms, data.charges

    def accumulate_samples_inner_fn(i, carry):
      loss, aux_data, data, grad_vector, fisher, key, time = carry
      mcmc_key, loss_key = jax.random.split(key)

      data, pmove = mcmc_step(
        params, data, mcmc_key, mcmc_width)
      
      loss, aux_data, grad_vector, fisher = accumulate_samples(
        params, key, data, time, grad_vector, fisher)
      
      return loss, aux_data, data, grad_vector, fisher, key, time

    logging.info(f"Starting sample accumulation for timestep")

    def rk2_inner_fn(params, key, data, time):
      
      grad_vector = jnp.zeros((n_params))
      fisher = jnp.zeros((n_params, n_params)) # TODO: Check memory footprint at this point

      data, pmove = mcmc_step(
        params, data, mcmc_key, mcmc_width)
      
      loss, aux_data, grad_vector, fisher = accumulate_samples(
        params, key, data, time, grad_vector, fisher)
      
      
      init_carry = (
        loss, aux_data, data, grad_vector, fisher, key, time)
      loss, aux_data, data, grad_vector, fisher, key, time = lax.fori_loop(
          0, iterations_per_timestep, accumulate_samples_inner_fn, init_carry
      )
      
      theta_dot, r2, eff_rank = conduct_timestep(
        params, key, data, grad_vector, fisher,
        batch_size)
      
      return data, pmove, loss, aux_data, theta_dot, r2, eff_rank

    if time_integration_method == 'rk2':
      
      logging.info("Starting RK2 first step")
      data, pmove, loss, aux_data, theta_dot_1, r2, eff_rank = constants.pmean(
        rk2_inner_fn(params, key, data, time))
      theta_dot_1 = -1j * theta_dot_1

      half_updates, _ = optimizer.update(
        unravel_fn(theta_dot_1 * 0.5), opt_state, params)
      params_mid = optax.apply_updates(params, half_updates)

      logging.info("Starting RK2 second step")
      data, pmove, loss, aux_data, theta_dot_2, r2, eff_rank = constants.pmean(
        rk2_inner_fn(params_mid, key, data, time))
      theta_dot_2 = -1j * theta_dot_2

      logging.info("Updating params")
      updates, opt_state = optimizer.update(
        unravel_fn(theta_dot_2), opt_state, params)
      params = optax.apply_updates(params, updates)

      logging.info("Completed timestep")

    else:
      raise NotImplementedError(
        "Alternate time integration has not yet been implemented!")
    
    return data, params, opt_state, \
      loss, aux_data, pmove, theta_dot_2, r2, eff_rank

  return step



def cg_err_estimator(
    mcmc_step, optimizer, accumulate_samples, conduct_timestep, iterations_per_timestep, n_electrons,
    cg_iterations):
    # The same timestep - does 10 CG steps and calculates variance in the update vector and in the loss grads
    # Then changes the samples accumulated per timestep and does the same again
    # Then changes the max iterations and does it again

    # We don't actually get the inverse matrice we get the solved vector (N_p), so we have to test on that instead
    # Check scale of loss grads changing and then that should give us an idea of the scale of the variance of the estimated inverse of the metric
    
    # Variance per iteration (200 v 2000 v 20000) - 10 iterations
    # Mean value for each (1e2, 1e3, 1e4)
    # Variance for different samples accumulated (5e3, 5e4, 5e5)
  @functools.partial(constants.pmap,
                      in_axes=(0, 0, 0, None, 0, 0),
                      donate_argnums=(0, 1, 2))
  def step(
    data: networks.FermiNetData,
    params: networks.ParamTree,
    opt_state: Optional[optax.OptState],
    time: float,
    key: chex.PRNGKey,
    mcmc_width: jnp.ndarray,
    time_integration_method = 'rk2'
  ):
    # MCMC loop

    batch_size = data.positions.shape[0]
    mcmc_key, loss_key = jax.random.split(key, num=2)
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(
      params)
    n_params = flat_params.shape[0]
    spins, atoms, charges = data.spins, data.atoms, data.charges

    def accumulate_samples_inner_fn(i, carry):
      position_arr, grad_vector, key, time = carry
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
        params, key, data, time, grad_vector)
      
      position_arr = lax.dynamic_update_slice(
          position_arr,
          data.positions,
          (batch_size * i, 0)   # (row_offset, col_offset)
      )

      return position_arr, grad_vector, key, time

    
    def timestep_variance_inner(i, carry):
      ## Variance of gradient update at time t with fixed data - consistency of inverse
      data, grad_vector, theta_dot_arr, timestep_arr, key, time = carry
      theta_dot = conduct_timestep(
        params, key, data, grad_vector, cg_iterations)
      
      theta_dot_arr = lax.dynamic_update_index_in_dim(
        theta_dot_arr,
        theta_dot,
        i,
        axis=0
      )

      return data, grad_vector, theta_dot_arr, timestep_arr, key, time
    
    def sample_variance_inner(i, carry):
      ## Variance of gradient update at time t with fixed data - consistency of inverse
      data, grad_vector, theta_dot_arr, samples_arr, key, time = carry

      n_samples = samples_arr[i]
      
      data = networks.FermiNetData(
          positions=data.positions.sample(n_samples),
          spins=data.spins[:n_samples, :],
          atoms=data.atoms[:n_samples, :],
          charges=data.charges[:n_samples, :]
      )

      theta_dot = conduct_timestep(
        params, key, data, grad_vector, cg_iterations)
      
      theta_dot_arr = lax.dynamic_update_index_in_dim(
        theta_dot_arr,
        theta_dot,
        i,
        axis=0
      )

      return data, grad_vector, theta_dot_arr, samples_arr, key, time
    
    def cg_variance_inner(i, carry):
      ## Variance of gradient update at time t with fixed data - consistency of inverse
      data, grad_vector, theta_dot_arr, cg_iterations_arr, key, time = carry

      cg_iterations = cg_iterations_arr[i]

      theta_dot = conduct_timestep(
        params, key, data, grad_vector, cg_iterations)
      
      theta_dot_arr = lax.dynamic_update_index_in_dim(
        theta_dot_arr,
        theta_dot,
        i,
        axis=0
      )

      return data, grad_vector, theta_dot_arr, cg_iterations_arr, key, time


    def get_per_timestep_metric(
        params, key, data, time, inner_fn, passed_arr):

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
          (batch_size, 0)
      )
      
      init_carry = (position_arr, grad_vector, key, time)
      position_arr, final_grad_vector, final_key, time = lax.fori_loop(
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
       
      per_timestep_iterations = passed_arr.shape[0]
      # That's for iterations_per_timestep samples
      theta_dot_arr = jnp.zeros(
        (per_timestep_iterations, n_params))
      init_carry = (
        data, grad_vector, theta_dot_arr, passed_arr, key, time)

      data, grad_vector, theta_dot_arr, passed_arr, key, time = lax.fori_loop(
        0, per_timestep_iterations, inner_fn, init_carry)
      
      
      return theta_dot_arr


    n_iterations_arr = jnp.zeros(10)
    samples_arr = jnp.array([1e3, 5e3, 1e4, 5e4, 1e5, 5e5])
    cg_iterations_arr = jnp.array([1e2, 1e3, 1e4, 5e4, 1e5])

    logging.info("Running per timestep variance")
    theta_dot_arr_per_iteration = get_per_timestep_metric(
      params, key, data, time, timestep_variance_inner, n_iterations_arr)
    
    theta_dot_mean_per_iteration = jnp.mean(theta_dot_arr_per_iteration, axis=0)
    theta_dot_var_per_iteration = (theta_dot_arr_per_iteration - theta_dot_mean_per_iteration) ** 2 / n_iterations_arr.shape[0]
    del theta_dot_arr_per_iteration

    logging.info("Running variance with n_cg_iterations")
    theta_dot_arr_w_cg_iterations = get_per_timestep_metric(
      params, key, data, time, cg_variance_inner, cg_iterations_arr
    )
    
    """
    logging.info("Running variance with n_samples accumulated")
    theta_dot_arr_w_samples = get_per_timestep_metric(
      params, key, data, time, sample_variance_inner, samples_arr
    )
    """
    theta_dot_arr_w_samples = jnp.zeros(10)

    # Probably need to double check with the loss and grads as well, but for now let's see if this works and we can extend from there
    return theta_dot_mean_per_iteration, theta_dot_var_per_iteration, theta_dot_arr_w_samples, theta_dot_arr_w_cg_iterations
  return step
