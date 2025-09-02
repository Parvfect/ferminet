
from networks import ParamTree
import optax


class SR:

    velocities: ParamTree  # Previous updates to the parameters

    def __init__(
            self, ntk=False, damping=1e-4, centre_gradients=True, ntk_solver='linear', preconditioning=False, preset_guess='loss', learning_rate_schedule):
        self.ntk = ntk
        self.damping = damping
        self.centre_gradients = centre_gradients
        self.ntk_solver = ntk_solver
        self.preconditioning = preconditioning
        self.preset_guess = preset_guess
        optax_chain = optax.chain(
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1.),)

    updates, opt_state = optimizer.update(grads, opt_state, params)

    def update(grads, state, params):

        updates, chained_opt_state = self.optax_chain(grads, state, params)

        return updates, opt_state