
import jax


def evaluate_mse_loss_teacher_student

def optimizer_step():
    jax.value_and_grad(evaluate_mse_loss_teacher_student, argnums=0, has_aux=True)
    def optimizer_step_distillation():

        # Evaluate network output through forward pass of the teacher params
        # Evaluate network output through forward pass of the student params
        # loss is mean squared error

        (loss, aux_data), grad = loss_and_grad(params, key, data)
        grad = constants.pmean(grad)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux_data


def make_training_step_distillation(mcmc_step):

    def step(data,
            params,
            opt_state,
            subkeys,
            data_teacher,
            params_teacher,
            mcmc_width):
        

        # Get the mcmc step generated from the teacher network
        mcmc_key, loss_key = jax.random.split(key, num=2)
        data, pmove = mcmc_step(
        params_teacher, data_teacher, mcmc_key, mcmc_width)

        # Optimization step
        new_params, new_state, loss, aux_data = optimizer_step_distillation(params,
                                                            data,
                                                            state,
                                                            loss_key)
        if reset_if_nan:
        new_params = jax.lax.cond(jnp.isnan(loss),
                                    lambda: params,
                                    lambda: new_params)
        new_state = jax.lax.cond(jnp.isnan(loss),
                                lambda: state,
                                lambda: new_state)
        return data, new_params, new_state, loss, aux_data, pmove
