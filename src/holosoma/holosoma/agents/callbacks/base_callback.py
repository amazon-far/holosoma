from torch.nn import Module


class RLEvalCallback(Module):
    def __init__(self, config=None, training_loop=None):
        super().__init__()
        self.config = config
        self.training_loop = training_loop
        self.device = self.training_loop.device if training_loop is not None else "cpu"

    def on_pre_evaluate_policy(self):
        pass

    def on_pre_eval_env_step(self, actor_state):
        return actor_state

    def on_post_eval_env_step(self, actor_state):
        return actor_state

    def on_post_evaluate_policy(self):
        pass
