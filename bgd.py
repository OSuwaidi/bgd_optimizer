from typing import Callable, Iterable, Any

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import Optimizer


class BGD(Optimizer):
    r"""
    Implements BGD (Bouncing Gradient Descent).
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (default: 0.1)
        beta (float): momentum factor (default: 0.9)
    Typical training procedure:
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            def closure():
                loss = criterion(m(x), y)
                loss.backward()
                return loss.item()
            opt.step(closure)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.3,
        beta: float = 0.9,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if beta < 0.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self._eps = eps

        decay_params: list[torch.nn.Parameter] = []
        decay_params_dims: int = 0

        no_decay_params: list[torch.nn.Parameter] = []
        no_decay_params_dims: int = 0

        for p in params:
            if not p.requires_grad:
                continue
            # Exclude biases and 1D normalization parameters from weight decay
            if weight_decay == 0 or p.ndim == 1:
                no_decay_params.append(p)
                no_decay_params_dims += p.numel()
            else:
                decay_params.append(p)
                decay_params_dims += p.numel()

        device = p.device

        prev_no_decay_params = torch.empty(no_decay_params_dims, device=device)
        prev_no_decay_grad = torch.empty_like(prev_no_decay_params)

        prev_decay_params = torch.empty(decay_params_dims, device=device)
        prev_decay_grad = torch.empty_like(prev_decay_params)

        optim_groups = [
            {
                "params": no_decay_params,
                "prev_params": prev_no_decay_params,
                "prev_grad": prev_no_decay_grad,
                "weight_decay": 0.0,
            }
        ]
        if weight_decay != 0:
            optim_groups.append(
                {
                    "params": decay_params,
                    "prev_params": prev_decay_params,
                    "prev_grad": prev_decay_grad,
                    "weight_decay": weight_decay,
                },
            )

        defaults = dict(lr=lr, beta=beta)
        super().__init__(optim_groups, defaults)  # exposes "self.param_groups" attribute

        self.param_groups: list[dict[str, Any]]

    @torch.no_grad()
    def step(self, closure: Callable[[], float]) -> float:
        # TODO: maybe no need to store prev_params, you can reconstruct it using prev_grad and proposed params
        """
        Performs a single optimization step that involves two backward passes.
        Args:
            closure (callable): A closure that evaluates the model, returns the loss, and performs backpropagation: REQUIRED for BGD.
        The closure should:
          - compute the loss (forward pass)
          - call `loss.backward()` ==> populates `p.grad`
          - return the loss as a Python float
        """

        self.zero_grad(set_to_none=True)

        with torch.enable_grad():
            closure()  # first forward/backward pass

        for group in self.param_groups:
            params = group["params"]
            lr = group["lr"]
            beta = group["beta"]
            wd = group["weight_decay"]

            P = parameters_to_vector(params)
            G = parameters_to_vector([p.grad for p in params])

            if wd > 0.0:
                G.add_(P, alpha=wd)  # coupled weight decay ==> regularized gradient

            group["prev_params"].copy_(P)
            group["prev_grad"].copy_(G)

            vector_to_parameters(P.sub_(G, alpha=lr), params)  # update current model's params to proposed params

        self.zero_grad(set_to_none=True)

        with torch.enable_grad():
            loss = closure()  # second forward/backward pass

        for group in self.param_groups:
            params = group["params"]
            prev_params = group["prev_params"]
            prev_grad = group["prev_grad"]
            lr = group["lr"]
            beta = group["beta"]
            wd = group["weight_decay"]

            P = parameters_to_vector(params)
            G = parameters_to_vector([p.grad for p in params])

            if wd > 0.0:
                G.add_(P, alpha=wd)

            if (prev_grad @ G) < 0.0:  # bounce condition on regularized gradients
                num = (prev_grad @ prev_grad).sqrt_().add_(self._eps)
                w = num.div_(num.add((G @ G).sqrt_()).add_(self._eps))
                vector_to_parameters(prev_params.lerp_(P, weight=w), params)
            else:
                if wd > 0.0:
                    G.sub_(P, alpha=wd)
                    prev_grad.sub_(prev_params, alpha=wd)

                vector_to_parameters(prev_params.mul_(1. - lr * wd).sub_(G.add_(prev_grad), alpha=lr / 2.0),params,)

        return loss
