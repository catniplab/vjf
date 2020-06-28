import torch

from .base import Model, Trainer


class VSGPFTrainer(Trainer):
    def __init__(self, model: Model):
        super().__init__(model)
        self.optimizer = torch.optim.Adam(model.trainable_variables)

    def step(self, output, y, u, q):
        """
        :param output: output from filtering
        :param y: observation (batch, dim)
        :param u: input (batch, dim)
        :param q: q_{t-1}
        :return:
        """

        model = self.model
        loss = model.loss(y, output, u, q)
        mu0 = q[0]
        mu1 = output[0][0]
        self.model.system.fit(mu0, mu1)
        self.optimizer.zero_grad()
        loss.backward()
        # ugly gradient-clipping
        torch.nn.utils.clip_grad_value_(
            self.model.trainable_variables, self.config["clip_gradients"]
        )
        self.optimizer.step()

        return loss.detach()
