

class SR:

    def __init__(
            self, ntk=False, damping=1e-4, centre_gradients=True, ntk_solver = 'linear', preconditioning = False, preset_guess = 'loss'):
        self.ntk = ntk
        self.damping = damping
        self.centre_gradients = centre_gradients
        self.ntk_solver = ntk_solver
        self.preconditioning = preconditioning
        self.preset_guess = preset_guess
        