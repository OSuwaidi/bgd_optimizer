import warnings

from typing import Callable, Iterable, Any

import torch
from torch.optim import Optimizer


def global_bounce(self, G1: torch.Tensor, G2: torch.Tensor, tau: float) -> torch.Tensor:
    # "Global" per optimizer param group (not fully model-global)
    return (G1 @ G2) < (-tau * G1.norm() * G2.norm())


def per_coordinate_bounce(self, G1: torch.Tensor, G2: torch.Tensor, tau: float) -> torch.Tensor:
    return (G1.mul(G2)) < 0.0


def ratio_convex_weights(
        self,
        G1: torch.Tensor, G2: torch.Tensor, eps: float = 1e-8,
        ) -> torch.Tensor:
    numerator = G1.abs().add_(eps)
    return numerator.div_(G2.abs_().add_(numerator).add_(eps))


def sigmoid_convex_weights(self, G1: torch.Tensor, G2: torch.Tensor) -> torch.Tensor:
    return G1.abs().sub_(G2.abs_()).sigmoid_()  # todo: add decaying temperature parameter


def full_decay_interpolation(
        self,
        P1: torch.Tensor,
        G1: torch.Tensor,
        weights: torch.Tensor,
        wd: float,
        lr: float,
        coupled_gradient: bool,
        ) -> torch.Tensor:
    r"""
    Performs: :math:`\theta_{t+1} = (1 - \alpha \, \lambda) \theta_t - \alpha \, w \odot g_t`
    """
    if wd > 0.0:
        if coupled_gradient:
            G1.sub_(P1, alpha=wd)
        P1.mul_(1 - wd * lr)
    return P1.sub_(weights.mul_(G1), alpha=lr)


def scaled_decay_interpolation(
        self,
        P1: torch.Tensor,
        G1: torch.Tensor,
        weights: torch.Tensor,
        wd: float,
        lr: float,
        coupled_gradient: bool,
        ) -> torch.Tensor:
    r"""
    Performs: :math:`\theta_{t+1} = (1 - \alpha \, \lambda w) \theta_t - \alpha \, w \odot g_t`
    """
    if wd > 0.0:
        if not coupled_gradient:
            G1.add_(P1, alpha=wd)
    return P1.sub_(weights.mul_(G1), alpha=lr)


class BGD(Optimizer):
    # TODO: Visualize low-rank dynamics across layers during training
    # TODO: Deal with BN layers by temporarily disabling its running-stat updates during the first or second pass
    r"""
    Implements BGD (Bouncing Gradient Descent) optimization algorithm.
    Args:
        params (iterable): iterable of parameters to optimize
        lr (float): base learning rate (default: 0.1)
        TODO beta (float): momentum factor (default: 0.9)
    """

    _couple_gradient = None

    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            lr: float = 0.1,
            beta: float = 0.9,
            weight_decay: float = 0.0,
            tau: float = 0.0,
            ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if beta < 0.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if tau < 0.0:
            raise ValueError(f"Invalid tau value: {tau}")

        self.tau = tau

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

        prev_no_decay_params = torch.empty(no_decay_params_dims, device=device)
        prev_no_decay_grad = torch.empty_like(prev_no_decay_params)

        prev_decay_params = torch.empty(decay_params_dims, device=device)
        prev_decay_grad = torch.empty_like(prev_decay_params)

        optim_groups = []

        if no_decay_params:
            optim_groups.append(
                    {
                        "params": no_decay_params,
                        "prev_params": prev_no_decay_params,
                        "prev_grad": prev_no_decay_grad,
                        "weight_decay": 0.0,
                        }
                    )
        if decay_params:
            optim_groups.append(
                    {
                        "params": decay_params,
                        "prev_params": prev_decay_params,
                        "prev_grad": prev_decay_grad,
                        "weight_decay": weight_decay,
                        },
                    )

        defaults = dict(lr=lr, beta=beta)  # shared across all optim/param groups
        super().__init__(optim_groups, defaults)  # exposes "self.param_groups" attribute

        self.param_groups: list[dict[str, Any]]

    def _bounce_condition(
            self, prev_G: torch.Tensor, probe_G: torch.Tensor, tau: float
            ) -> torch.Tensor:
        raise NotImplementedError

    def _get_convex_weights(
            self, prev_G: torch.Tensor, probe_G: torch.Tensor,
            ) -> torch.Tensor:
        raise NotImplementedError

    def _interpolate(
            self,
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

            P = self._params_to_vec(params)
            G = self._param_grads_to_vec(params)

            if wd > 0.0:
                if self._couple_gradient:
                    G.add_(P, alpha=wd)  # coupled weight decay ==> regularized gradient

            group["prev_params"].copy_(P)
            group["prev_grad"].copy_(G)

            probe_P = P.sub_(G, alpha=lr)
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

            probe_P = self._params_to_vec(params)
            probe_G = self._param_grads_to_vec(params)

            if wd > 0.0:
                if self._couple_gradient:
                    probe_G.add_(probe_P, alpha=wd)

            bounce_cond = self._bounce_condition(prev_G, probe_G, self.tau)

            if bounce_cond.ndim == 0:
                # Global bounce condition branch
                if bounce_cond.item():
                    w = self._get_convex_weights(prev_G, probe_G)
                    new_P = self._interpolate(prev_P, prev_G, w, wd, lr, self._couple_gradient)
                else:
                    if wd > 0.0:
                        if self._couple_gradient:  # decouple/dergularize the L2 penalty from gradients
                            prev_G.sub_(prev_P, alpha=wd)
                            probe_G.sub_(probe_P, alpha=wd)

                        prev_P.mul_(1.0 - lr * wd)

                    new_P = prev_P.sub_(probe_G.add_(prev_G), alpha=lr / 2.0)

            else:  # Per-coordinate bounce condition branch --- TODO: maybe computing full bounce and full non_bounce then using "torch.where()" is more efficient?
                # Bouncing coordinates
                w = self._get_convex_weights(prev_G[bounce_cond], probe_G[bounce_cond])
                new_P = torch.empty_like(probe_P)
                new_P[bounce_cond] = self._interpolate(
                        prev_P[bounce_cond],
                        prev_G[bounce_cond],
                        w,
                        wd,
                        lr,
                        self._couple_gradient,
                        )

                # Non-bouncing coordinates
                non_bounce = ~bounce_cond
                if wd > 0.0:
                    if self._couple_gradient:  # decouple the L2 penalty from gradients
                        # Boolean indexing returns a *new* temporary copy (rather than views)
                        # Achieve true in-place op via augmented assignment
                        prev_G[non_bounce] -= prev_P[non_bounce].mul_(wd)
                        probe_G[non_bounce] -= probe_P[non_bounce].mul_(wd)

                    prev_P[non_bounce] *= (1.0 - lr * wd)

                new_P[non_bounce] = prev_P[non_bounce].sub_(probe_G[non_bounce].add_(prev_G[non_bounce]).mul_(lr / 2.0))

            self._assign_vec_to_params(new_P, params)

        return loss


# Naming: <coupling><bounce><weights><interp>
#   coupling: C = coupled,  D = decoupled
#   bounce:   G = global,   P = per-coordinate
#   weights:  R = ratio,    S = sigmoid
#   interp:   F = full-decay, S = scaled-decay
# Slot order is fixed


class DGRS(BGD):
    _couple_gradient = False  # D
    _bounce_condition = global_bounce  # G
    _get_convex_weights = ratio_convex_weights  # R
    _interpolate = scaled_decay_interpolation  # S



class CGRS(BGD):
    _couple_gradient = True  # C
    _bounce_condition = global_bounce  # G
    _get_convex_weights = ratio_convex_weights  # R
    _interpolate = scaled_decay_interpolation  # S


class DGRF(BGD):
    _couple_gradient = False  # D
    _bounce_condition = global_bounce  # G
    _get_convex_weights = ratio_convex_weights  # R
    _interpolate = full_decay_interpolation  # F


class CGRF(BGD):
    _couple_gradient = True  # C
    _bounce_condition = global_bounce  # G
    _get_convex_weights = ratio_convex_weights  # R
    _interpolate = full_decay_interpolation  # F


class DGSS(BGD):
    _couple_gradient = False  # D
    _bounce_condition = global_bounce  # G
    _get_convex_weights = sigmoid_convex_weights  # S
    _interpolate = scaled_decay_interpolation  # S


class CGSS(BGD):
    _couple_gradient = True  # C
    _bounce_condition = global_bounce  # G
    _get_convex_weights = sigmoid_convex_weights  # S
    _interpolate = scaled_decay_interpolation  # S


class DGSF(BGD):
    _couple_gradient = False  # D
    _bounce_condition = global_bounce  # G
    _get_convex_weights = sigmoid_convex_weights  # S
    _interpolate = full_decay_interpolation  # F


class CGSF(BGD):
    _couple_gradient = True  # C
    _bounce_condition = global_bounce  # G
    _get_convex_weights = sigmoid_convex_weights  # S
    _interpolate = full_decay_interpolation  # F


class DPRS(BGD):
    _couple_gradient = False  # D
    _bounce_condition = per_coordinate_bounce  # P
    _get_convex_weights = ratio_convex_weights  # R
    _interpolate = scaled_decay_interpolation  # S


class CPRS(BGD):
    _couple_gradient = True  # C
    _bounce_condition = per_coordinate_bounce  # P
    _get_convex_weights = ratio_convex_weights  # R
    _interpolate = scaled_decay_interpolation  # S


class DPRF(BGD):
    _couple_gradient = False  # D
    _bounce_condition = per_coordinate_bounce  # P
    _get_convex_weights = ratio_convex_weights  # R
    _interpolate = full_decay_interpolation  # F


class CPRF(BGD):
    _couple_gradient = True  # C
    _bounce_condition = per_coordinate_bounce  # P
    _get_convex_weights = ratio_convex_weights  # R
    _interpolate = full_decay_interpolation  # F


class DPSS(BGD):
    _couple_gradient = False  # D
    _bounce_condition = per_coordinate_bounce  # P
    _get_convex_weights = sigmoid_convex_weights  # S
    _interpolate = scaled_decay_interpolation  # S


class CPSS(BGD):
    _couple_gradient = True  # C
    _bounce_condition = per_coordinate_bounce  # P
    _get_convex_weights = sigmoid_convex_weights  # S
    _interpolate = scaled_decay_interpolation  # S


class DPSF(BGD):
    _couple_gradient = False  # D
    _bounce_condition = per_coordinate_bounce  # P
    _get_convex_weights = sigmoid_convex_weights  # S
    _interpolate = full_decay_interpolation  # F


class CPSF(BGD):
    _couple_gradient = True  # C
    _bounce_condition = per_coordinate_bounce  # P
    _get_convex_weights = sigmoid_convex_weights  # S
    _interpolate = full_decay_interpolation  # F


BGD_VARIANTS: tuple[type[BGD]] = (
    DGRS,
    CGRS,
    DGRF,
    CGRF,
    DGSS,
    CGSS,
    DGSF,
    CGSF,
    DPRS,
    CPRS,
    DPRF,
    CPRF,
    DPSS,
    CPSS,
    DPSF,
    CPSF,
    )
