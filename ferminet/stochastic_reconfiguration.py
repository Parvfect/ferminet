
import jax
import jax.numpy as jnp
import functools
import chex
import optax
from typing import Optional, Mapping, Sequence, Tuple, Union
from ferminet import constants
from ferminet import networks


def make_sr_opt_update_step(evaluate_loss,
                         optimizer, batch_network, damping=1e-2):
  """Returns an OptUpdate function for performing a parameter update."""

  # Differentiate wrt parameters (argument 0)
  loss_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)

  def opt_update( 
      params: networks.ParamTree,
      data: networks.FermiNetData,
      opt_state: Optional[optax.OptState],
      key: chex.PRNGKey,
      ntk=False,
      ntk_solver='cg',
      centre_gradients=True
  ):
    """Evaluates the loss and gradients and updates the parameters using optax."""

    (loss, aux_data), grad = loss_and_grad(params, key, data)
    flat_grads, unravel_fn = jax.flatten_util.ravel_pytree(grad)
    energies = aux_data.local_energy - loss
    batch_size = energies.shape[0]
  
    def f(params):
      psi = batch_network(
        params, data.positions, data.spins, data.atoms, data.charges)
      return psi
  
    jvp_func = lambda x: jax.linearize(f, params)[1](
      unravel_fn(x))
    
    vjp_func = lambda v: jax.flatten_util.ravel_pytree(
      jax.vjp(f, params)[1](v))[0]
    
    def fisher_matmul(
        v, centre_gradients=centre_gradients, damping=damping):
      
      log_psi_jac_v = jvp_func(v)
      update_vector = vjp_func(log_psi_jac_v / batch_size)
        
      if centre_gradients:
          update_vector -= jnp.mean(update_vector)
      
      update_vector += damping * v
      return update_vector
    
    x0 = flat_grads  # Using loss grads as guess        
    grads = jax.scipy.sparse.linalg.cg(
      fisher_matmul, flat_grads, x0=x0, maxiter=200)[0]
        
    grads = constants.pmean(grads)  # Handling for multi-gpu
    
    updates, opt_state = optimizer.update(  
      unravel_fn(grads), opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, opt_state, loss, aux_data
  
  return opt_update


def make_sr_training_step(
    mcmc_step,
    optimizer_step,
    reset_if_nan: bool = False,
):
  """Factory to create traning step for non-KFAC optimizers.

  Args:
    mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
      for creating the callable.
    optimizer_step: OptUpdate callable which evaluates the forward and backward
      passes and updates the parameters and optimizer state, as required.
    reset_if_nan: If true, reset the params and opt state to the state at the
      previous step when the loss is NaN

  Returns:
    step, a callable which performs a set of MCMC steps and then an optimization
    update. See the Step protocol for details.
  """
  @functools.partial(constants.pmap,
                    donate_argnums=(0, 1, 2))
  def step(
      data: networks.FermiNetData,
      params: networks.ParamTree,
      opt_state: Optional[optax.OptState],
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray
  ):
    """A full update iteration (except for KFAC): MCMC steps + optimization."""
    # MCMC loop
    mcmc_key, loss_key = jax.random.split(key, num=2)
    
    #mcmc_keys, loss_keys = kfac_jax.utils.p_split(key)
    data, pmove = mcmc_step(
      params, data, mcmc_key, mcmc_width)

    # Optimization step
    new_params, new_state, loss, aux_data = optimizer_step(params,
                                                           data,
                                                           opt_state,
                                                           loss_key)
    if reset_if_nan:
      new_params = jax.lax.cond(jnp.isnan(loss),
                                lambda: params,
                                lambda: new_params)
      new_state = jax.lax.cond(jnp.isnan(loss),
                               lambda: opt_state,
                               lambda: new_state)
    return data, new_params, new_state, loss, aux_data, pmove

  return step
