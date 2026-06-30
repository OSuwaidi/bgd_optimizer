# بسم الله الرحمن الرحيم وبه نستعين

import warnings

from typing import Callable, Iterable, Any

import torch
from torch.optim import Optimizer


def global_bounce(G1: torch.Tensor, G2: torch.Tensor, tau: float = 0.0) -> torch.Tensor:
    # "Global" per optimizer param group (not truly model-global)
    return (G1 @ G2) < (-tau * G1.norm() * G2.norm())


def per_coordinate_bounce(G1: torch.Tensor, G2: torch.Tensor, tau: float = 0.0) -> torch.Tensor:
    return (G1.mul(G2)) < 0.0


def ratio_convex_weights(
        prev_G: torch.Tensor, probe_G: torch.Tensor, eps: float = 1e-8,
        ) -> torch.Tensor:
    numerator = prev_G.abs().add_(eps)
    return numerator.div_(probe_G.abs_().add_(numerator).add_(eps))


def sigmoid_convex_weights(prev_G: torch.Tensor, probe_G: torch.Tensor, eps: float = 1e-8, ) -> torch.Tensor:
    return prev_G.abs().sub_(probe_G.abs_()).sigmoid_()


def full_decay_interpolation(
        prev_P: torch.Tensor,
        prev_G: torch.Tensor,
        weights: torch.Tensor,
        weight_decay: float,
        learning_rate: float,
        coupled_gradient: bool,
        ) -> torch.Tensor:
    r"""
    Performs: :math:`\theta_{t+1} = (1 - \alpha \, \lambda) \theta_t - \alpha \, w \odot g_t`

    ONLY applies to **decoupled** gradients approach! ==> Can't have "CXXF" variant
    """
    if weight_decay > 0.0:
        prev_P.mul_(1 - weight_decay * learning_rate)

    return prev_P.sub_(weights.mul_(prev_G), alpha=learning_rate)


def scaled_decay_interpolation(
        prev_P: torch.Tensor,
        prev_G: torch.Tensor,
        weights: torch.Tensor,
        weight_decay: float,
        learning_rate: float,
        coupled_gradient: bool,
        ) -> torch.Tensor:
    r"""
    Performs: :math:`\theta_{t+1} = (1 - \alpha \, \lambda w) \theta_t - \alpha \, w \odot g_t`
    """
    if weight_decay > 0.0:
        if not coupled_gradient:
            prev_G = prev_G.add(prev_P, alpha=weight_decay)

    return prev_P.sub_(weights.mul_(prev_G), alpha=learning_rate)


class BGD(Optimizer):
    # TODO: Visualize low-rank dynamics across layers during training
    # TODO: Deal with BN layers by temporarily disabling its running-stat updates during the first or second pass
    r"""
    Implements BGD (Bouncing Gradient Descent) optimization algorithm.
    Args:
        params (iterable): iterable of parameters to optimize
        lr (float): base learning rate (default: 0.1)
        beta (float): momentum factor (default: 0.9)
        EMA (bool): whether to use EMA-based momentum or (False) Heavy Ball momentum (default: True)
        weight_decay (float): L2 norm weight decay value (default: 0.0)
        tau (float): bounce condition tolerance; the higher, the more restrictive the condition is (default: 0.0)
    """

    _couple_gradient = False

    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            lr: float = 0.1,
            beta: float = 0.9,
            EMA: bool = True,
            weight_decay: float = 0.0,
            tau: float = 0.0,
            ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= tau < 1.0:
            raise ValueError(f"Invalid tau value: {tau}")

        self.EMA = EMA
        self.tau = tau
        self.param_groups: list[dict[str, Any]]

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

        if not decay_params and not no_decay_params:
            raise ValueError("BGD received no trainable parameters.")

        device = p.device
        if "cuda" not in device.type:
            warnings.warn(f"Model parameters' device is not CUDA, rather is {device.type}!", stacklevel=2)

        beta: torch.Tensor = torch.tensor(beta, device=device)

        no_decay_prev_params = torch.empty(no_decay_params_dims, device=device)
        no_decay_prev_grad = torch.empty_like(no_decay_prev_params)
        no_decay_momentum = torch.zeros_like(no_decay_prev_params)
        no_decay_step_index = torch.zeros_like(no_decay_prev_params)

        decay_prev_params = torch.empty(decay_params_dims, device=device)
        decay_prev_grad = torch.empty_like(decay_prev_params)
        decay_momentum = torch.zeros_like(decay_prev_params)
        decay_step_index = torch.zeros_like(decay_prev_params)

        optim_groups = []

        if no_decay_params:
            optim_groups.append(
                    {
                        "params": no_decay_params,
                        "prev_params": no_decay_prev_params,
                        "prev_grad": no_decay_prev_grad,
                        "momentum": no_decay_momentum,
                        "t": no_decay_step_index,
                        "weight_decay": 0.0,
                        }
                    )
        if decay_params:
            optim_groups.append(
                    {
                        "params": decay_params,
                        "prev_params": decay_prev_params,
                        "prev_grad": decay_prev_grad,
                        "momentum": decay_momentum,
                        "t": decay_step_index,
                        "weight_decay": weight_decay,
                        },
                    )

        defaults = dict(lr=lr, beta=beta)  # shared across all optim/param groups
        super().__init__(optim_groups, defaults)  # exposes "self.param_groups" attribute

    @staticmethod
    def _bounce_condition(
            G1: torch.Tensor, G2: torch.Tensor, tau: float = 0.0,
            ) -> torch.Tensor:
        # True implies bounce
        raise NotImplementedError

    @staticmethod
    def _get_convex_weights(
            prev_G: torch.Tensor, probe_G: torch.Tensor, eps: float = 1e-8,
            ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _interpolate(
            prev_P: torch.Tensor,
            prev_G: torch.Tensor,
            weights: torch.Tensor,
            weight_decay: float,
            learning_rate: float,
            coupled_gradient: bool,
            ) -> torch.Tensor:
        # Interpolation weights are assigned to the probe point, where w=0 ==> prev_params and w=1 ==> probe_params (w/ or w/o weight_decay)
        raise NotImplementedError

    @staticmethod
    def _params_to_vec(params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
        return torch.cat([p.view(-1) for p in params])

    @staticmethod
    def _param_grads_to_vec(params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
        return torch.cat(
                [
                    p.grad.view(-1) if p.grad is not None
                    else torch.zeros_like(p).view(-1)
                    for p in params
                    ]
                )

    @staticmethod
    def _assign_vec_to_params(vec: torch.Tensor, params: Iterable[torch.nn.Parameter]) -> None:
        pointer = 0
        for param in params:
            end = pointer + param.numel()
            param.copy_(vec[pointer: end].view_as(param))
            pointer = end

    @torch.no_grad()
    # pyrefly: ignore [bad-override]
    def step(self, closure: Callable[[], float]) -> float:
        # TODO: maybe no need to store "prev_params" as we can reconstruct it using "prev_grad" and "probe params"
        r"""
        Performs a single optimization step that involves two backward passes.
        Args:
            closure (callable): A callable that evaluates the model, returns the loss, and performs backpropagation: REQUIRED for BGD.
        The closure should:
          - compute the loss (forward pass)
          - call `loss.backward()` --> populates `p.grad`
          - return the loss as a Python float
        Typical closure structure:
            def closure(x, y):
                loss = criterion(model(x), y)
                loss.backward()
                return loss.item()
        Then used in training loop as:
            opt.step(lambda: closure(x, y))
        """

        self.zero_grad(set_to_none=True)

        with torch.enable_grad():
            loss = closure()  # first forward/backward pass

        for group in self.param_groups:
            params = group["params"]
            lr = group["lr"]
            wd = group["weight_decay"]
            m = group["momentum"]
            t = group["t"]
            beta = group["beta"]

            t.add_(1.0)

            P = self._params_to_vec(params)
            G = self._param_grads_to_vec(params)

            if wd > 0.0 and self._couple_gradient:
                G.add_(P, alpha=wd)  # coupled weight decay ==> regularized gradient

            group["prev_params"].copy_(P)
            group["prev_grad"].copy_(G)

            if self.EMA:
                m.lerp_(G, weight=1.0 - beta)
            else:
                m.mul_(beta).add_(G)

            unbias_m = m / (1.0 - beta ** t) if self.EMA else m

            probe_P = P.sub_(unbias_m, alpha=lr)
            self._assign_vec_to_params(probe_P, params)  # update current model's params to probe params

        self.zero_grad(set_to_none=True)

        with torch.enable_grad():
            closure()  # second forward/backward pass

        for group in self.param_groups:
            params = group["params"]
            prev_P = group["prev_params"]
            prev_G = group["prev_grad"]
            lr = group["lr"]
            wd = group["weight_decay"]
            m = group["momentum"]
            t = group["t"]
            beta = group["beta"]

            probe_P = self._params_to_vec(params)
            probe_G = self._param_grads_to_vec(params)

            if wd > 0.0 and self._couple_gradient:
                probe_G.add_(probe_P, alpha=wd)

            bounce_cond = self._bounce_condition(prev_G, probe_G, self.tau)

            if bounce_cond.ndim == 0:
                # Global bounce condition branch
                if bounce_cond.item():
                    unbias_m = m / (1.0 - beta ** t) if self.EMA else m
                    w = self._get_convex_weights(prev_G, probe_G, )
                    new_P = self._interpolate(prev_P, unbias_m, w, wd, lr, self._couple_gradient)

                    # Restart momentum state
                    m.zero_()
                    t.zero_()
                else:
                    if wd > 0.0 and not self._couple_gradient:
                            prev_P.mul_(1.0 - lr * wd)

                    average_G = (prev_G + probe_G) / 2.0
                    if self.EMA:
                        m.add_(probe_G.sub_(prev_G), alpha=(1.0 - beta) / 2.0)
                    else:
                        m.add_(probe_G.sub_(prev_G), alpha=0.5)

                    new_P = prev_P.sub_(average_G, alpha=lr)

            else:  # Per-coordinate bounce condition branch --- TODO: maybe computing full bounce and full non_bounce then using "torch.where()" is more efficient?
                new_P = torch.empty_like(probe_P)

                # Non-bouncing coordinates
                non_bounce = ~bounce_cond
                if wd > 0.0 and not self._couple_gradient:
                        prev_P[non_bounce] *= (1.0 - lr * wd)

                average_G = (prev_G[non_bounce] + probe_G[non_bounce]) / 2.0
                if self.EMA:
                    m[non_bounce] += probe_G[non_bounce].sub_(prev_G[non_bounce]).mul_((1.0 - beta) / 2.0)
                else:
                    m[non_bounce] += probe_G[non_bounce].sub_(prev_G[non_bounce]).mul_(0.5)

                new_P[non_bounce] = prev_P[non_bounce].sub_(average_G, alpha=lr)

                # Bouncing coordinates
                unbias_m = m / (1.0 - beta ** t) if self.EMA else m
                w = self._get_convex_weights(prev_G[bounce_cond], probe_G[bounce_cond], )
                new_P[bounce_cond] = self._interpolate(
                        prev_P[bounce_cond],
                        unbias_m[bounce_cond],
                        w,
                        wd,
                        lr,
                        self._couple_gradient,
                        )

                # Restart momentum state
                m[bounce_cond] *= 0.0
                t[bounce_cond] *= 0.0

            self._assign_vec_to_params(new_P, params)

        return loss


# Naming: <coupling><bounce><weights><interp>
#   coupling: C = coupled,  D = decoupled
#   bounce:   G = global,   P = per-coordinate
#   weights:  R = ratio,    S = sigmoid
#   interp:   F = full-decay, S = scaled-decay
# Slot order is fixed

# NOTE: Coupled gradients are only compatible with scaled decay interpolation.

class DGSS(BGD):
    _couple_gradient = False  # D
    _bounce_condition = staticmethod(global_bounce)  # G
    _get_convex_weights = staticmethod(sigmoid_convex_weights)  # S
    _interpolate = staticmethod(scaled_decay_interpolation)  # S


class CGSS(BGD):
    _couple_gradient = True  # C
    _bounce_condition = staticmethod(global_bounce)  # G
    _get_convex_weights = staticmethod(sigmoid_convex_weights)  # S
    _interpolate = staticmethod(scaled_decay_interpolation)  # S


class DGSF(BGD):
    _couple_gradient = False  # D
    _bounce_condition = staticmethod(global_bounce)  # G
    _get_convex_weights = staticmethod(sigmoid_convex_weights)  # S
    _interpolate = staticmethod(full_decay_interpolation)  # F


class DGRF(BGD):
    _couple_gradient = False  # D
    _bounce_condition = staticmethod(global_bounce)  # G
    _get_convex_weights = staticmethod(ratio_convex_weights)  # R
    _interpolate = staticmethod(full_decay_interpolation)  # F


class DGRS(BGD):
    _couple_gradient = False  # D
    _bounce_condition = staticmethod(global_bounce)  # G
    _get_convex_weights = staticmethod(ratio_convex_weights)  # R
    _interpolate = staticmethod(scaled_decay_interpolation)  # S


class CGRS(BGD):
    _couple_gradient = True  # C
    _bounce_condition = staticmethod(global_bounce)  # G
    _get_convex_weights = staticmethod(ratio_convex_weights)  # R
    _interpolate = staticmethod(scaled_decay_interpolation)  # S


BGD_VARIANTS: tuple[type[BGD], ...] = (
    DGSS,
    CGSS,
    DGSF,
    DGRF,
    DGRS,
    CGRS,
    )
