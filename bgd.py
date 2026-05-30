from typing import Callable

import torch
import torch.nn as nn
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
        self, params, lr: float = 0.3, beta: float = 0.9, weight_decay: float = 0.0
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if beta < 0.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # TODO:
        """
        decay_params = []
        no_decay_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Exclude biases and 1D normalization parameters
            if "bias" in name or param.ndim == 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        """

        # Expose dict kwargs to schedulers via "param_groups"
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self._params: list[nn.Parameter] = [
            p for p in self.param_groups[0]["params"] if p.requires_grad
        ]

        # Flatten entire model:
        with torch.no_grad():
            self.P = parameters_to_vector(self._params)
        self._prev_P: torch.Tensor = torch.empty_like(self.P)
        self._v: torch.Tensor = torch.zeros_like(self.P)
        self._G: torch.Tensor = torch.empty_like(self.P)

    @torch.no_grad()
    def step(self, closure: Callable[[], float]) -> float:
        """
        Performs a single optimization step that involves two backward passes.
        Args:
            closure (callable): A closure that re-evaluates the model
                and returns the loss. REQUIRED for BGD.
        The closure should:
          - compute the loss (forward)
          - call `loss.backward()`
          - return the loss as a float
        """

        # 1. Initialization & Group Params
        self.zero_grad(set_to_none=True)

        group = self.param_groups[0]
        lr = group["lr"]
        beta = group["beta"]
        wd = group.get("weight_decay", 0.0)

        with torch.enable_grad():
            loss = closure()

        # --- Phase 1: Preliminary Update ---
        # Capture current params and gradients
        self._G.copy_(parameters_to_vector([p.grad for p in self._params]))
        self.zero_grad(set_to_none=True)

        if wd > 0.0:
            self.P.mul_(1.0 - lr * wd)

        # Save previous position
        self._prev_P.copy_(self.P)

        # Update velocity (momentum)
        self._v.mul_(beta).add_(self._G)

        # Apply Preliminary Update
        self.P.sub_(self._v, alpha=lr)

        # Write flat params back to the model
        vector_to_parameters(self.P, self._params)

        # --- Phase 2: Lookahead (Second Forward/Backward) ---
        with torch.enable_grad():
            closure()

        # --- Phase 3: Bounce Update ---
        # Get new gradients at the updated position
        self._G.copy_(parameters_to_vector([p.grad for p in self._params]))

        # Check Bounce Condition (Dot product of Velocity and New Gradient)
        if (self._v @ self._G) < 0.0:
            # Bounce: Interpolate back towards previous position
            w = self._G.abs_().sub_(self._v.abs_()).sigmoid_()
            self.P.lerp_(self._prev_P, weight=w)
            self._v.zero_()

        else:
            # Continue Descent Based on Local Gradient
            self.P.sub_(self._G, alpha=lr)

        # Final Write Back
        vector_to_parameters(self.P, self._params)
        return loss
