import torch as t
from typing import Iterable

class Adam:
    """
    Custom implementations of Adam optimizer. 
    Like the PyTorch version, but assumes amsgrad=False and maximize=False.
    https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
    """
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = weight_decay

        self.m = [t.zeros_like(p) for p in self.params]
        self.v = [t.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        self.t += 1
        with t.inference_mode():
            for i, p in enumerate(self.params):
                assert p.grad is not None
                g = p.grad + self.wd * p
                self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g**2
                m_hat = self.m[i] / (1.0 - self.beta1**self.t)
                v_hat = self.v[i] / (1.0 - self.beta2**self.t)
                p -= self.lr * m_hat / (v_hat.sqrt() + self.eps)