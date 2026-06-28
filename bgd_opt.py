from typing import Callable
import numpy as np
import torch
import torch.nn as nn
from math import sqrt

from torch.nn.utils import parameters_to_vector, vector_to_parameters


class BGD_nest:
    r"""
    Args:
        model (callable): model (nn.Module) containing trainable parameters to optimize via BGD
        lr (float): base learning rate
        _eps (float): epsilon for convex weights (default: 1e-8)
    """

    def __init__(
            self,
            model: nn.Module,
            lr: float = 0.1,
            non_blocking: bool = False,
            _eps: float = 1e-8,
            beta: float = 0.9,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")

        self.model = model
        self.trainable_params: list[nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
        with torch.no_grad():
            params_vec = parameters_to_vector(self.trainable_params)

        self._prev_params: torch.Tensor = torch.empty_like(params_vec)
        self._v: torch.Tensor = torch.zeros_like(params_vec)

        del params_vec

        self.device = self.trainable_params[0].device
        self.lr = lr
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps
        self.epoch_losses: list[float] = []
        self.b = beta

    @torch.no_grad()
    def _bounce_update(self, curr_p: torch.Tensor) -> None:
        curr_grad = self._get_grads_vec()  # compute the gradient at the lookahead position
        self._v.mul_(self.b).add_(curr_grad)

        if (self._v @ curr_grad) < 0.:
            # print("bounce!")
            w = curr_grad.abs_().sub_(self._v.abs_()).sigmoid_()
            self._v.zero_()
            curr_p.lerp_(self._prev_params, weight=w)
        else:
            # Update velocity (momentum)
            curr_p.copy_(self._prev_params).sub_(self._v, alpha=self.lr)

        vector_to_parameters(curr_p, self.trainable_params)
        self.model.zero_grad(set_to_none=True)

    def _get_grads_vec(self) -> torch.Tensor:
        return parameters_to_vector([p.grad for p in self.trainable_params])

    def train(self, train_loader: torch.utils.data.DataLoader, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> float:
        self.model.train()
        epoch_loss = 0.
        n_samples = 0
        for x, y in train_loader:
            with torch.no_grad():
                P = parameters_to_vector(self.trainable_params)
                self._prev_params.copy_(P)

                # Apply pure momentum-based (lookahead) update
                P.sub_(self._v, alpha=self.lr)

                # PUSH BACK: Write the updated flat params into the actual model layers
                vector_to_parameters(P, self.trainable_params)

            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            loss = criterion(self.model(x), y)
            bsz = y.shape[0]
            epoch_loss += loss.item() * bsz
            n_samples += bsz
            loss.backward()

            self._bounce_update(P)

        return epoch_loss / n_samples

    @torch.inference_mode()
    def test(self, test_loader: torch.utils.data.dataloader.DataLoader, ) -> float:
        self.model.eval()
        correct = 0
        n_samples = 0
        for x, y in test_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            pred = self.model(x).argmax(dim=1)
            correct += pred.eq(y).sum().item()
            n_samples += y.shape[0]

        accuracy = (100. * correct / n_samples)
        return accuracy


class SelfBGDv:  # BETTER THAN SelfBGDm
    r"""
    Args:
        model (callable): model (nn.Module) containing trainable parameters to optimize via BGD
        lr (float): base learning rate
        _eps (float): epsilon for convex weights (default: 1e-8)
    """

    def __init__(
            self,
            model: nn.Module,
            lr: float = 0.1,
            non_blocking: bool = False,
            _eps: float = 1e-8,
            beta: float = 0.9,

    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        self.model = model
        self.trainable_params: list[nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
        with torch.no_grad():
            params_vec = parameters_to_vector(self.trainable_params)

        self._prev_P: torch.Tensor = torch.empty_like(params_vec)
        self._v: torch.Tensor = torch.zeros_like(params_vec)

        del params_vec

        self.device = self.trainable_params[0].device
        self.lr = lr
        self.b = beta
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps

    @torch.no_grad()
    def _bounce(self) -> None:
        P = parameters_to_vector(self.trainable_params)
        G = self._get_grads_vec()

        self._v.mul_(self.b).add_(G)

        if (self._v @ G) < 0.:
            w = G.abs_().sub_(self._v.abs_()).sigmoid_()
            self._v.zero_()
            P.lerp_(self._prev_P, weight=w)

        else:
            self._prev_P.copy_(P)

            P.sub_(self._v, alpha=self.lr)

        vector_to_parameters(P, self.trainable_params)
        self.model.zero_grad(set_to_none=True)

    def _get_grads_vec(self) -> torch.Tensor:
        return parameters_to_vector([p.grad for p in self.trainable_params])

    def train(self, train_loader: torch.utils.data.DataLoader, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> float:
        self.model.train()
        epoch_loss = 0.
        n_samples = 0
        for x, y in train_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            loss = criterion(self.model(x), y)
            bsz = y.shape[0]
            epoch_loss += loss.item() * bsz
            n_samples += bsz
            loss.backward()
            self._bounce()  # compares inter-batch gradients

        return epoch_loss / n_samples

    @torch.inference_mode()
    def test(self, test_loader: torch.utils.data.dataloader.DataLoader, ) -> float:
        self.model.eval()
        correct = 0
        n_samples = 0
        for x, y in test_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            pred = self.model(x).argmax(dim=1)
            correct += pred.eq(y).sum().item()
            n_samples += y.shape[0]
        accuracy = (100. * correct / n_samples)
        return accuracy


class SelfBGDm:
    r"""
    Args:
        model (callable): model (nn.Module) containing trainable parameters to optimize via BGD
        lr (float): base learning rate
        _eps (float): epsilon for convex weights (default: 1e-8)
    """

    def __init__(
            self,
            model: nn.Module,
            lr: float = 0.1,
            non_blocking: bool = False,
            _eps: float = 1e-8,
            beta: float = 0.9,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")

        self.model = model
        self.trainable_params: list[nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
        with torch.no_grad():
            params_vec = parameters_to_vector(self.trainable_params)

        self._prev_P: torch.Tensor = torch.empty_like(params_vec)
        self._m: torch.Tensor = torch.zeros_like(params_vec)

        del params_vec

        self.device = self.trainable_params[0].device
        self.lr = lr
        self.b = beta
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps
        self.t = 0

    @torch.no_grad()
    def _bounce(self) -> None:
        P = parameters_to_vector(self.trainable_params)
        G = self._get_grads_vec()
        self._m.lerp_(G, weight=(1. - self.b))

        if (self._m @ G) < 0.:
            w = G.abs_().sub_(self._m.abs_()).sigmoid_()
            self._m.zero_()
            self.t = 0
            P.lerp_(self._prev_P, weight=w)

        else:
            self._prev_P.copy_(P)

            self.t += 1
            b_corr = 1. - self.b ** self.t
            P.sub_(self._m, alpha=self.lr / b_corr)

        vector_to_parameters(P, self.trainable_params)
        self.model.zero_grad(set_to_none=True)

    def _get_grads_vec(self) -> torch.Tensor:
        return parameters_to_vector([p.grad for p in self.trainable_params])

    def train(self, train_loader: torch.utils.data.DataLoader, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> float:
        self.model.train()
        epoch_loss = 0.
        n_samples = 0
        for x, y in train_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            loss = criterion(self.model(x), y)
            bsz = y.shape[0]
            epoch_loss += loss.item() * bsz
            n_samples += bsz
            loss.backward()
            self._bounce()  # compares inter-batch gradients

        return epoch_loss / n_samples

    @torch.inference_mode()
    def test(self, test_loader: torch.utils.data.dataloader.DataLoader, ) -> float:
        self.model.eval()
        correct = 0
        n_samples = 0
        for x, y in test_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            pred = self.model(x).argmax(dim=1)
            correct += pred.eq(y).sum().item()
            n_samples += y.shape[0]

        accuracy = (100. * correct / n_samples)
        return accuracy


class BGDNew:
    r"""
    Args:
        model (callable): model (nn.Module) containing trainable parameters to optimize via BGD
        lr (float): base learning rate
        _eps (float): epsilon for convex weights (default: 1e-8)
    """

    def __init__(
            self,
            model: nn.Module,
            lr: float = 0.1,
            non_blocking: bool = False,
            _eps: float = 1e-8,
            beta: float = 0.9,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")

        self.model = model
        self.trainable_params: list[nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
        with torch.no_grad():
            params_vec = parameters_to_vector(self.trainable_params)

        self._prev_P: torch.Tensor = torch.empty_like(params_vec)
        self._v: torch.Tensor = torch.zeros_like(params_vec)
        del params_vec

        self.device = self.trainable_params[0].device
        self.lr = lr
        self.b = beta
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps

    @torch.no_grad()
    def _bounce1(self) -> None:
        P = parameters_to_vector(self.trainable_params)
        G = self._get_grads_vec()

        self._v.mul_(self.b).add_(G)

        if (self._v @ G) < 0.:
            w = G.abs_().sub_(self._v.abs_()).sigmoid_()
            self._v.zero_()
            P.lerp_(self._prev_P, weight=w)
        else:
            self._prev_P.copy_(P)
            P.sub_(self._v, alpha=self.lr)

        vector_to_parameters(P, self.trainable_params)
        self.model.zero_grad(set_to_none=True)

    @torch.no_grad()
    def _bounce2(self) -> None:
        P = parameters_to_vector(self.trainable_params)
        G = self._get_grads_vec()

        if (self._v @ G) < 0.:
            w = G.abs_().sub_(self._v.abs_()).sigmoid_()
            self._v.zero_()
            P.lerp_(self._prev_P, weight=w)
        else:
            self._prev_P.copy_(P)
            self._v.mul_(self.b).add_(G)
            P.sub_(self._v, alpha=self.lr)

        vector_to_parameters(P, self.trainable_params)
        self.model.zero_grad(set_to_none=True)

    def _get_grads_vec(self) -> torch.Tensor:
        return parameters_to_vector([p.grad for p in self.trainable_params])

    def train(
            self,
            train_loader: torch.utils.data.DataLoader,
            criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> float:
        self.model.train()
        epoch_loss = 0.0
        n_samples = 0

        for x, y in train_loader:
            x = x.to(self.device, non_blocking=self.non_blocking)
            y = y.to(self.device, non_blocking=self.non_blocking)

            loss = criterion(self.model(x), y)
            bsz = y.shape[0]
            epoch_loss += loss.item() * bsz
            n_samples += bsz

            loss.backward()
            self._bounce1()

            criterion(self.model(x), y).backward()
            self._bounce2()

        return epoch_loss / n_samples

    @torch.inference_mode()
    def test(self, test_loader: torch.utils.data.dataloader.DataLoader) -> float:
        self.model.eval()
        correct = 0
        n_samples = 0

        for x, y in test_loader:
            x = x.to(self.device, non_blocking=self.non_blocking)
            y = y.to(self.device, non_blocking=self.non_blocking)

            pred = self.model(x).argmax(dim=1)
            correct += pred.eq(y).sum().item()
            n_samples += y.shape[0]

        return 100.0 * correct / n_samples


from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from math import sqrt


class BGD_sq:
    r"""
    Args:
        model (callable): model (nn.Module) containing trainable parameters to optimize via BGD
        lr (float): base learning rate
        _eps (float): epsilon for convex weights (default: 1e-12)
    """

    def __init__(
            self,
            model: nn.Module,
            lr: float = 0.1,
            non_blocking: bool = False,
            _eps: float = 1e-8,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")

        self.model = model
        self.trainable_params: list[nn.Parameter] = []
        self._prev_params: list[torch.Tensor] = []
        self._prev_grads: list[torch.Tensor] = []
        for param in model.parameters():
            if param.requires_grad:
                self.trainable_params.append(param)  # stores a reference (pointer) to the actual model parameters
                self._prev_params.append(torch.empty_like(param))
                self._prev_grads.append(torch.empty_like(param))

        self.lr = lr
        self.device = param.device
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps
        self.epoch_losses: list[float] = []

    def _bounce_update(self, idx: int, curr_p: torch.Tensor) -> None:
        prev_grad = self._prev_grads[idx].view(-1)
        curr_grad = curr_p.grad.view(-1)
        if (prev_grad @ curr_grad).item() < 0.:
            # print("bounce!")
            f1, f2 = prev_grad @ prev_grad, curr_grad @ curr_grad
            s = f1 + f2 + self._eps
            d1 = f2 / s

            # curr_p.data = self._prev_params[idx] * d1 + curr_p * d2
            # curr_p.mul_(d2).add_(self._prev_params[idx], alpha=d1)
            curr_p.lerp_(self._prev_params[idx], weight=d1)

        else:
            # curr_p.data = curr_p - curr_grad * self.layer_lr[idx]
            curr_p.sub_(curr_grad.view(curr_p.shape), alpha=self.lr)

    def train(self, train_loader: torch.utils.data.DataLoader, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> float:
        self.model.train()
        epoch_loss = 0.
        n_samples = 0
        for x, y in train_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            loss = criterion(self.model(x), y)
            bsz = y.shape[0]
            epoch_loss += loss.item() * bsz
            n_samples += bsz
            loss.backward()

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    self._prev_params[i].copy_(p)
                    self._prev_grads[i].copy_(p.grad)
                    p.sub_(p.grad, alpha=self.lr)
                    p.grad = None

            criterion(self.model(x), y).backward()

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    self._bounce_update(i, p)
                    p.grad = None

        # self.epoch_losses.append(epoch_loss / n_samples)
        return epoch_loss / n_samples

    @torch.inference_mode()
    def test(self, test_loader: torch.utils.data.dataloader.DataLoader, ) -> float:
        self.model.eval()
        correct = 0
        n_samples = 0
        for x, y in test_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            pred = self.model(x).argmax(dim=1)
            correct += pred.eq(y).sum().item()
            n_samples += y.shape[0]

        accuracy = (100. * correct / n_samples)
        return accuracy


class BGD_soft:
    r"""
    Args:
        model (callable): model (nn.Module) containing trainable parameters to optimize via BGD
        lr (float): base learning rate
        _eps (float): epsilon for convex weights (default: 1e-8)
    """

    def __init__(
            self,
            model: nn.Module,
            lr: float = 0.1,
            non_blocking: bool = False,
            _eps: float = 1e-8,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")

        self.model = model
        self.trainable_params: list[nn.Parameter] = []
        self._prev_params: list[torch.Tensor] = []
        self._prev_grads: list[torch.Tensor] = []
        for param in model.parameters():
            if param.requires_grad:
                self.trainable_params.append(param)  # stores a reference (pointer) to the actual model parameters
                self._prev_params.append(torch.empty_like(param))
                self._prev_grads.append(torch.empty_like(param))

        self.device = param.device
        self.lr = lr
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps
        self.epoch_losses: list[float] = []

    def _bounce_update(self, idx: int, curr_p: torch.Tensor) -> None:
        prev_grad = self._prev_grads[idx]
        curr_grad = curr_p.grad
        if (prev_grad.view(-1) @ curr_grad.view(-1)).item() < 0.:
            # print("bounce!")
            w = curr_grad.sub_(prev_grad).sigmoid_()

            # curr_p.mul_(d2).add_(self._prev_params[idx], alpha=d1)
            curr_p.lerp_(self._prev_params[idx], weight=w)

        else:
            # curr_p.data = curr_p - curr_grad * self.lr
            curr_p.sub_(curr_grad, alpha=self.lr)

    def train(self, train_loader: torch.utils.data.DataLoader, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> float:
        self.model.train()
        epoch_loss = 0.
        n_samples = 0
        for x, y in train_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            loss = criterion(self.model(x), y)
            bsz = y.shape[0]
            epoch_loss += loss.item() * bsz
            n_samples += bsz
            loss.backward()

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    self._prev_params[i].copy_(p)
                    self._prev_grads[i].copy_(p.grad)
                    p.sub_(p.grad, alpha=self.lr)
                    p.grad = None

            criterion(self.model(x), y).backward()

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    self._bounce_update(i, p)
                    p.grad = None

        # self.epoch_losses.append(epoch_loss / n_samples)
        return epoch_loss / n_samples

    @torch.inference_mode()
    def test(self, test_loader: torch.utils.data.dataloader.DataLoader, ) -> float:
        self.model.eval()
        correct = 0
        n_samples = 0
        for x, y in test_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            pred = self.model(x).argmax(dim=1)
            correct += pred.eq(y).sum().item()
            n_samples += y.shape[0]

        accuracy = (100. * correct / n_samples)
        return accuracy


class BGD_soft_div2:
    r"""
    Args:
        model (callable): model (nn.Module) containing trainable parameters to optimize via BGD
        lr (float): base learning rate
        _eps (float): epsilon for convex weights (default: 1e-8)
    """

    def __init__(
            self,
            model: nn.Module,
            lr: float = 0.1,
            non_blocking: bool = False,
            _eps: float = 1e-8,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")

        self.model = model
        self.trainable_params: list[nn.Parameter] = []
        self._prev_params: list[torch.Tensor] = []
        self._prev_grads: list[torch.Tensor] = []
        for param in model.parameters():
            if param.requires_grad:
                self.trainable_params.append(param)  # stores a reference (pointer) to the actual model parameters
                self._prev_params.append(torch.empty_like(param))
                self._prev_grads.append(torch.empty_like(param))

        self.device = param.device
        self.lr = lr
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps
        self.epoch_losses: list[float] = []

    def _bounce_update(self, idx: int, curr_p: torch.Tensor) -> None:
        prev_grad = self._prev_grads[idx]
        curr_grad = curr_p.grad
        if (prev_grad.view(-1) @ curr_grad.view(-1)).item() < 0.:
            # print("bounce!")
            w = curr_grad.sub_(prev_grad).div_(2.).sigmoid_()

            # curr_p.mul_(d2).add_(self._prev_params[idx], alpha=d1)
            curr_p.lerp_(self._prev_params[idx], weight=w)

        else:
            # curr_p.data = curr_p - curr_grad * self.lr
            curr_p.sub_(curr_grad, alpha=self.lr)

    def train(self, train_loader: torch.utils.data.DataLoader, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> float:
        self.model.train()
        epoch_loss = 0.
        n_samples = 0
        for x, y in train_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            loss = criterion(self.model(x), y)
            bsz = y.shape[0]
            epoch_loss += loss.item() * bsz
            n_samples += bsz
            loss.backward()

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    self._prev_params[i].copy_(p)
                    self._prev_grads[i].copy_(p.grad)
                    p.sub_(p.grad, alpha=self.lr)
                    p.grad = None

            criterion(self.model(x), y).backward()

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    self._bounce_update(i, p)
                    p.grad = None

        # self.epoch_losses.append(epoch_loss / n_samples)
        return epoch_loss / n_samples

    @torch.inference_mode()
    def test(self, test_loader: torch.utils.data.dataloader.DataLoader, ) -> float:
        self.model.eval()
        correct = 0
        n_samples = 0
        for x, y in test_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            pred = self.model(x).argmax(dim=1)
            correct += pred.eq(y).sum().item()
            n_samples += y.shape[0]

        accuracy = (100. * correct / n_samples)
        return accuracy


class BGD_soft_div2_ema_m1_dual:
    r"""
    Args:
        model (callable): model (nn.Module) containing trainable parameters to optimize via BGD
        lr (float): base learning rate
        _eps (float): epsilon for convex weights (default: 1e-8)
    """

    def __init__(
            self,
            model: nn.Module,
            lr: float = 0.1,
            non_blocking: bool = False,
            _eps: float = 1e-8,
            beta: float = 0.9,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")

        self.model = model
        self.trainable_params: list[nn.Parameter] = []
        self._prev_params: list[torch.Tensor] = []
        self._prev_grads: list[torch.Tensor] = []
        self._first_moment: list[torch.Tensor] = []
        for param in model.parameters():
            if param.requires_grad:
                self.trainable_params.append(param)  # stores a reference (pointer) to the actual model parameters
                self._prev_params.append(torch.empty_like(param))
                self._prev_grads.append(torch.empty_like(param))
                self._first_moment.append(torch.zeros_like(param))

        self.device = param.device
        self.lr = lr
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps
        self.epoch_losses: list[float] = []
        self.b = beta
        self.t = 0

    def _bounce_update(self, idx: int, curr_p: torch.Tensor) -> None:
        prev_grad = self._prev_grads[idx]
        curr_grad = curr_p.grad
        if (prev_grad.view(-1) @ curr_grad.view(-1)).item() < 0.:
            # print("bounce!")
            w = curr_grad.sub(prev_grad).div_(2.).sigmoid_()

            # curr_p.mul_(d2).add_(self._prev_params[idx], alpha=d1)
            curr_p.lerp_(self._prev_params[idx], weight=w)

        else:
            # curr_p.data = curr_p - curr_grad * self.lr
            curr_p.sub_(curr_grad, alpha=self.lr*0.1)

    def train(self, train_loader: torch.utils.data.DataLoader, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> float:
        self.model.train()
        epoch_loss = 0.
        n_samples = 0
        for x, y in train_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            loss = criterion(self.model(x), y)
            bsz = y.shape[0]
            epoch_loss += loss.item() * bsz
            n_samples += bsz
            loss.backward()
            self.t += 1

            with torch.no_grad():
                b_corr = 1. - self.b ** self.t
                for i, p in enumerate(self.trainable_params):
                    self._prev_params[i].copy_(p)
                    self._prev_grads[i].copy_(p.grad)
                    m = self._first_moment[i].lerp_(p.grad, weight=(1. - self.b))
                    p.sub_(m, alpha=self.lr/b_corr)
                    p.grad = None

            criterion(self.model(x), y).backward()

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    self._bounce_update(i, p)
                    p.grad = None

        # self.epoch_losses.append(epoch_loss / n_samples)
        return epoch_loss / n_samples

    @torch.inference_mode()
    def test(self, test_loader: torch.utils.data.dataloader.DataLoader, ) -> float:
        self.model.eval()
        correct = 0
        n_samples = 0
        for x, y in test_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            pred = self.model(x).argmax(dim=1)
            correct += pred.eq(y).sum().item()
            n_samples += y.shape[0]

        accuracy = (100. * correct / n_samples)
        return accuracy


class BGD_soft_div2_ema_m1:
    r"""
    Args:
        model (callable): model (nn.Module) containing trainable parameters to optimize via BGD
        lr (float): base learning rate
        _eps (float): epsilon for convex weights (default: 1e-8)
    """

    def __init__(
            self,
            model: nn.Module,
            lr: float = 0.1,
            non_blocking: bool = False,
            _eps: float = 1e-8,
            beta: float = 0.9,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")

        self.model = model
        self.trainable_params: list[nn.Parameter] = []
        self._prev_params: list[torch.Tensor] = []
        self._prev_grads: list[torch.Tensor] = []
        self._first_moment: list[torch.Tensor] = []
        for param in model.parameters():
            if param.requires_grad:
                self.trainable_params.append(param)  # stores a reference (pointer) to the actual model parameters
                self._prev_params.append(torch.empty_like(param))
                self._prev_grads.append(torch.empty_like(param))
                self._first_moment.append(torch.zeros_like(param))

        self.device = param.device
        self.lr = lr
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps
        self.epoch_losses: list[float] = []
        self.b = beta
        self.t = 0

    def _bounce_update(self, idx: int, curr_p: torch.Tensor) -> None:
        prev_grad = self._prev_grads[idx]
        curr_grad = curr_p.grad
        if (prev_grad.view(-1) @ curr_grad.view(-1)).item() < 0.:
            # print("bounce!")
            w = curr_grad.sub(prev_grad).div_(2.).sigmoid_()

            # curr_p.mul_(d2).add_(self._prev_params[idx], alpha=d1)
            curr_p.lerp_(self._prev_params[idx], weight=w)

        else:
            # curr_p.data = curr_p - curr_grad * self.lr
            curr_p.sub_(curr_grad, alpha=self.lr)

    def train(self, train_loader: torch.utils.data.DataLoader, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> float:
        self.model.train()
        epoch_loss = 0.
        n_samples = 0
        for x, y in train_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            loss = criterion(self.model(x), y)
            bsz = y.shape[0]
            epoch_loss += loss.item() * bsz
            n_samples += bsz
            loss.backward()
            self.t += 1

            with torch.no_grad():
                b_corr = 1. - self.b ** self.t
                for i, p in enumerate(self.trainable_params):
                    self._prev_params[i].copy_(p)
                    self._prev_grads[i].copy_(p.grad)
                    m = self._first_moment[i].lerp_(p.grad, weight=(1. - self.b))
                    p.sub_(m, alpha=self.lr/b_corr)
                    p.grad = None

            criterion(self.model(x), y).backward()

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    self._bounce_update(i, p)
                    p.grad = None

        # self.epoch_losses.append(epoch_loss / n_samples)
        return epoch_loss / n_samples

    @torch.inference_mode()
    def test(self, test_loader: torch.utils.data.dataloader.DataLoader, ) -> float:
        self.model.eval()
        correct = 0
        n_samples = 0
        for x, y in test_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            pred = self.model(x).argmax(dim=1)
            correct += pred.eq(y).sum().item()
            n_samples += y.shape[0]

        accuracy = (100. * correct / n_samples)
        return accuracy


class BGD_soft_div2_ema_m1_margin:
    r"""
    Args:
        model (callable): model (nn.Module) containing trainable parameters to optimize via BGD
        lr (float): base learning rate
        _eps (float): epsilon for convex weights (default: 1e-8)
    """

    def __init__(
            self,
            model: nn.Module,
            lr: float = 0.1,
            non_blocking: bool = False,
            _eps: float = 1e-8,
            beta: float = 0.9,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")

        self.model = model
        self.trainable_params: list[nn.Parameter] = []
        self._prev_params: list[torch.Tensor] = []
        self._prev_grads: list[torch.Tensor] = []
        self._first_moment: list[torch.Tensor] = []
        for param in model.parameters():
            if param.requires_grad:
                self.trainable_params.append(param)  # stores a reference (pointer) to the actual model parameters
                self._prev_params.append(torch.empty_like(param))
                self._prev_grads.append(torch.empty_like(param))
                self._first_moment.append(torch.zeros_like(param))

        self.device = param.device
        self.lr = lr
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps
        self.epoch_losses: list[float] = []
        self.b = beta
        self.t = 0
        self.margin = -0.7

    def _bounce_update(self, idx: int, curr_p: torch.Tensor) -> None:
        prev_grad = self._prev_grads[idx]
        curr_grad = curr_p.grad
        if (prev_grad.view(-1) @ curr_grad.view(-1)).item() < (self.margin * prev_grad.norm() * curr_grad.norm()):
            # print("bounce!")
            w = curr_grad.sub(prev_grad).div_(2.).sigmoid_()

            # curr_p.mul_(d2).add_(self._prev_params[idx], alpha=d1)
            curr_p.lerp_(self._prev_params[idx], weight=w)

        else:
            # curr_p.data = curr_p - curr_grad * self.lr
            curr_p.sub_(curr_grad, alpha=self.lr)

    def train(self, train_loader: torch.utils.data.DataLoader, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> float:
        self.model.train()
        epoch_loss = 0.
        n_samples = 0
        for x, y in train_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            loss = criterion(self.model(x), y)
            bsz = y.shape[0]
            epoch_loss += loss.item() * bsz
            n_samples += bsz
            loss.backward()
            self.t += 1

            with torch.no_grad():
                b_corr = 1. - self.b ** self.t
                for i, p in enumerate(self.trainable_params):
                    self._prev_params[i].copy_(p)
                    self._prev_grads[i].copy_(p.grad)
                    m = self._first_moment[i].lerp_(p.grad, weight=(1. - self.b))
                    p.sub_(m, alpha=self.lr/b_corr)
                    p.grad = None

            criterion(self.model(x), y).backward()

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    self._bounce_update(i, p)
                    p.grad = None

        # self.epoch_losses.append(epoch_loss / n_samples)
        return epoch_loss / n_samples

    @torch.inference_mode()
    def test(self, test_loader: torch.utils.data.dataloader.DataLoader, ) -> float:
        self.model.eval()
        correct = 0
        n_samples = 0
        for x, y in test_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            pred = self.model(x).argmax(dim=1)
            correct += pred.eq(y).sum().item()
            n_samples += y.shape[0]

        accuracy = (100. * correct / n_samples)
        return accuracy


class BGD_soft_div2_ema_m1_rst:
    r"""
    Args:
        model (callable): model (nn.Module) containing trainable parameters to optimize via BGD
        lr (float): base learning rate
        _eps (float): epsilon for convex weights (default: 1e-8)
    """

    def __init__(
            self,
            model: nn.Module,
            lr: float = 0.1,
            non_blocking: bool = False,
            _eps: float = 1e-8,
            beta: float = 0.9,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")

        self.model = model
        self.trainable_params: list[nn.Parameter] = []
        self._prev_params: list[torch.Tensor] = []
        self._prev_grads: list[torch.Tensor] = []
        self._first_moment: list[torch.Tensor] = []
        for param in model.parameters():
            if param.requires_grad:
                self.trainable_params.append(param)  # stores a reference (pointer) to the actual model parameters
                self._prev_params.append(torch.empty_like(param))
                self._prev_grads.append(torch.empty_like(param))
                self._first_moment.append(torch.zeros_like(param))

        self.device = param.device
        self.lr = lr
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps
        self.epoch_losses: list[float] = []
        self.b = beta
        self.t = np.zeros(len(self.trainable_params))

    def _bounce_update(self, idx: int, curr_p: torch.Tensor) -> None:
        prev_grad = self._prev_grads[idx]
        curr_grad = curr_p.grad
        if (prev_grad.view(-1) @ curr_grad.view(-1)) < 0.:
            # print("bounce!")
            self._first_moment[idx].zero_()
            self._first_moment[idx].lerp_(curr_grad, weight=(1. - self.b))
            self.t[idx] = 1

            w = curr_grad.sub(prev_grad).div_(2.).sigmoid_()

            # curr_p.mul_(d2).add_(self._prev_params[idx], alpha=d1)
            curr_p.lerp_(self._prev_params[idx], weight=w)

        else:
            # curr_p.data = curr_p - curr_grad * self.lr
            curr_p.sub_(curr_grad, alpha=self.lr)

    def train(self, train_loader: torch.utils.data.DataLoader, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> float:
        self.model.train()
        epoch_loss = 0.
        n_samples = 0
        for x, y in train_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            loss = criterion(self.model(x), y)
            bsz = y.shape[0]
            epoch_loss += loss.item() * bsz
            n_samples += bsz
            loss.backward()
            self.t += 1

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    self._prev_params[i].copy_(p)
                    self._prev_grads[i].copy_(p.grad)
                    m = self._first_moment[i].lerp_(p.grad, weight=(1. - self.b))
                    b_corr = 1. - self.b ** self.t[i]
                    p.sub_(m, alpha=self.lr/b_corr)
                    p.grad = None

            criterion(self.model(x), y).backward()

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    self._bounce_update(i, p)
                    p.grad = None

        # self.epoch_losses.append(epoch_loss / n_samples)
        return epoch_loss / n_samples

    @torch.inference_mode()
    def test(self, test_loader: torch.utils.data.dataloader.DataLoader, ) -> float:
        self.model.eval()
        correct = 0
        n_samples = 0
        for x, y in test_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            pred = self.model(x).argmax(dim=1)
            correct += pred.eq(y).sum().item()
            n_samples += y.shape[0]

        accuracy = (100. * correct / n_samples)
        return accuracy


class BGD_soft_div2_ema_m15:
    r"""
    Args:
        model (callable): model (nn.Module) containing trainable parameters to optimize via BGD
        lr (float): base learning rate
        _eps (float): epsilon for convex weights (default: 1e-8)
    """

    def __init__(
            self,
            model: nn.Module,
            lr: float = 0.1,
            non_blocking: bool = False,
            _eps: float = 1e-8,
            beta: float = 0.9,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")

        self.model = model
        self.trainable_params: list[nn.Parameter] = []
        self._prev_params: list[torch.Tensor] = []
        self._prev_grads: list[torch.Tensor] = []
        self._first_moment: list[torch.Tensor] = []
        for param in model.parameters():
            if param.requires_grad:
                self.trainable_params.append(param)  # stores a reference (pointer) to the actual model parameters
                self._prev_params.append(torch.empty_like(param))
                self._prev_grads.append(torch.empty_like(param))
                self._first_moment.append(torch.zeros_like(param))

        self.device = param.device
        self.lr = lr
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps
        self.epoch_losses: list[float] = []
        self.b = beta
        self.t = np.zeros(len(self.trainable_params))

    def _bounce_update(self, idx: int, curr_p: torch.Tensor) -> None:
        prev_grad = self._prev_grads[idx]
        curr_grad = curr_p.grad
        if (prev_grad.view(-1) @ curr_grad.view(-1)).item() < 0.:
            # print("bounce!")
            w = curr_grad.sub(prev_grad).div_(2.).sigmoid_()

            # curr_p.mul_(d2).add_(self._prev_params[idx], alpha=d1)
            curr_p.lerp_(self._prev_params[idx], weight=w)

        else:
            # curr_p.data = curr_p - curr_grad * self.lr
            self.t[idx] += 1
            self._first_moment[idx].lerp_(curr_grad, weight=(1. - self.b))
            curr_p.sub_(curr_grad, alpha=self.lr)

    def train(self, train_loader: torch.utils.data.DataLoader, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> float:
        self.model.train()
        epoch_loss = 0.
        n_samples = 0
        for x, y in train_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            loss = criterion(self.model(x), y)
            bsz = y.shape[0]
            epoch_loss += loss.item() * bsz
            n_samples += bsz
            loss.backward()
            self.t += 1

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    self._prev_params[i].copy_(p)
                    self._prev_grads[i].copy_(p.grad)
                    m = self._first_moment[i].lerp_(p.grad, weight=(1. - self.b))
                    b_corr = 1. - self.b ** self.t[i]
                    p.sub_(m, alpha=self.lr/b_corr)
                    p.grad = None

            criterion(self.model(x), y).backward()

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    self._bounce_update(i, p)
                    p.grad = None

        # self.epoch_losses.append(epoch_loss / n_samples)
        return epoch_loss / n_samples

    @torch.inference_mode()
    def test(self, test_loader: torch.utils.data.dataloader.DataLoader, ) -> float:
        self.model.eval()
        correct = 0
        n_samples = 0
        for x, y in test_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            pred = self.model(x).argmax(dim=1)
            correct += pred.eq(y).sum().item()
            n_samples += y.shape[0]

        accuracy = (100. * correct / n_samples)
        return accuracy


class BGD_soft_div2_poly_m1:
    r"""
    Args:
        model (callable): model (nn.Module) containing trainable parameters to optimize via BGD
        lr (float): base learning rate
        _eps (float): epsilon for convex weights (default: 1e-8)
    """

    def __init__(
            self,
            model: nn.Module,
            lr: float = 0.1,
            non_blocking: bool = False,
            _eps: float = 1e-8,
            beta: float = 0.9
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")

        self.model = model
        self.trainable_params: list[nn.Parameter] = []
        self._prev_params: list[torch.Tensor] = []
        self._prev_grads: list[torch.Tensor] = []
        self._first_moment: list[torch.Tensor] = []
        for param in model.parameters():
            if param.requires_grad:
                self.trainable_params.append(param)  # stores a reference (pointer) to the actual model parameters
                self._prev_params.append(torch.empty_like(param))
                self._prev_grads.append(torch.empty_like(param))
                self._first_moment.append(torch.zeros_like(param))

        self.device = param.device
        self.lr = lr
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps
        self.epoch_losses: list[float] = []
        self.b = beta

    def _bounce_update(self, idx: int, curr_p: torch.Tensor) -> None:
        prev_grad = self._prev_grads[idx]
        curr_grad = curr_p.grad
        if (prev_grad.view(-1) @ curr_grad.view(-1)).item() < 0.:
            # print("bounce!")
            w = curr_grad.sub(prev_grad).div_(2.).sigmoid_()

            # curr_p.mul_(d2).add_(self._prev_params[idx], alpha=d1)
            curr_p.lerp_(self._prev_params[idx], weight=w)

        else:
            # curr_p.data = curr_p - curr_grad * self.lr
            curr_p.sub_(curr_grad, alpha=self.lr)

    def train(self, train_loader: torch.utils.data.DataLoader, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> float:
        self.model.train()
        epoch_loss = 0.
        n_samples = 0
        for x, y in train_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            loss = criterion(self.model(x), y)
            bsz = y.shape[0]
            epoch_loss += loss.item() * bsz
            n_samples += bsz
            loss.backward()

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    self._prev_params[i].copy_(p)
                    self._prev_grads[i].copy_(p.grad)
                    m = self._first_moment[i].mul_(self.b).add_(p.grad)
                    p.sub_(m, alpha=self.lr)
                    p.grad = None

            criterion(self.model(x), y).backward()

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    self._bounce_update(i, p)
                    p.grad = None

        # self.epoch_losses.append(epoch_loss / n_samples)
        return epoch_loss / n_samples

    @torch.inference_mode()
    def test(self, test_loader: torch.utils.data.dataloader.DataLoader, ) -> float:
        self.model.eval()
        correct = 0
        n_samples = 0
        for x, y in test_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            pred = self.model(x).argmax(dim=1)
            correct += pred.eq(y).sum().item()
            n_samples += y.shape[0]

        accuracy = (100. * correct / n_samples)
        return accuracy


class BGD_soft_div2_poly_m15:
    r"""
    Args:
        model (callable): model (nn.Module) containing trainable parameters to optimize via BGD
        lr (float): base learning rate
        _eps (float): epsilon for convex weights (default: 1e-8)
    """

    def __init__(
            self,
            model: nn.Module,
            lr: float = 0.1,
            non_blocking: bool = False,
            _eps: float = 1e-8,
            beta: float = 0.9,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")

        self.model = model
        self.trainable_params: list[nn.Parameter] = []
        self._prev_params: list[torch.Tensor] = []
        self._prev_grads: list[torch.Tensor] = []
        self._first_moment: list[torch.Tensor] = []
        for param in model.parameters():
            if param.requires_grad:
                self.trainable_params.append(param)  # stores a reference (pointer) to the actual model parameters
                self._prev_params.append(torch.empty_like(param))
                self._prev_grads.append(torch.empty_like(param))
                self._first_moment.append(torch.zeros_like(param))

        self.device = param.device
        self.lr = lr
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps
        self.epoch_losses: list[float] = []
        self.b = beta

    def _bounce_update(self, idx: int, curr_p: torch.Tensor) -> None:
        prev_grad = self._prev_grads[idx]
        curr_grad = curr_p.grad
        if (prev_grad.view(-1) @ curr_grad.view(-1)).item() < 0.:
            # print("bounce!")
            w = curr_grad.sub(prev_grad).div_(2.).sigmoid_()

            # curr_p.mul_(d2).add_(self._prev_params[idx], alpha=d1)
            curr_p.lerp_(self._prev_params[idx], weight=w)

        else:
            # curr_p.data = curr_p - curr_grad * self.lr
            self._first_moment[idx].mul_(self.b).add_(curr_grad)
            curr_p.sub_(curr_grad, alpha=self.lr)

    def train(self, train_loader: torch.utils.data.DataLoader, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> float:
        self.model.train()
        epoch_loss = 0.
        n_samples = 0
        for x, y in train_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            loss = criterion(self.model(x), y)
            bsz = y.shape[0]
            epoch_loss += loss.item() * bsz
            n_samples += bsz
            loss.backward()

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    self._prev_params[i].copy_(p)
                    self._prev_grads[i].copy_(p.grad)
                    m = self._first_moment[i].mul_(self.b).add_(p.grad)
                    p.sub_(m, alpha=self.lr)
                    p.grad = None

            criterion(self.model(x), y).backward()

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    self._bounce_update(i, p)
                    p.grad = None

        # self.epoch_losses.append(epoch_loss / n_samples)
        return epoch_loss / n_samples

    @torch.inference_mode()
    def test(self, test_loader: torch.utils.data.dataloader.DataLoader, ) -> float:
        self.model.eval()
        correct = 0
        n_samples = 0
        for x, y in test_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            pred = self.model(x).argmax(dim=1)
            correct += pred.eq(y).sum().item()
            n_samples += y.shape[0]

        accuracy = (100. * correct / n_samples)
        return accuracy


class BGD_loss_rel:
    r"""
    Args:
        model (callable): model (nn.Module) containing trainable parameters to optimize via BGD
        lr (float): base learning rate
        _eps (float): epsilon for convex weights (default: 1e-12)
    """

    def __init__(
            self,
            model: nn.Module,
            lr: float = 0.1,
            non_blocking: bool = False,
            _eps: float = 1e-8,
            beta = 0.9,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")

        self.model = model
        self.trainable_params: list[nn.Parameter] = []
        self._prev_params: list[torch.Tensor] = []
        self._prev_grads: list[torch.Tensor] = []
        self._first_moment: list[torch.Tensor] = []
        for param in model.parameters():
            if param.requires_grad:
                self.trainable_params.append(param)  # stores a reference (pointer) to the actual model parameters
                self._prev_params.append(torch.empty_like(param))
                self._prev_grads.append(torch.empty_like(param))
                self._first_moment.append(torch.zeros_like(param))

        self.lr = lr
        self.device = param.device
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps
        self.epoch_losses: list[float] = []
        self.b = beta
        self.t = 0

    def _bounce_update(self, idx: int, curr_p: torch.Tensor, prev_loss: torch.Tensor, curr_loss: torch.Tensor) -> None:
        prev_grad = self._prev_grads[idx]
        curr_grad = curr_p.grad
        if (prev_grad.view(-1) @ curr_grad.view(-1)).item() < 0.:
            # print("bounce!")
            s = prev_loss + curr_loss + self._eps
            w = curr_loss / s

            curr_p.lerp_(self._prev_params[idx], weight=w)

        else:
            curr_p.sub_(curr_grad, alpha=self.lr)

    def train(self, train_loader: torch.utils.data.DataLoader, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> float:
        self.model.train()
        epoch_loss = 0.
        n_samples = 0
        for x, y in train_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            loss_prev = criterion(self.model(x), y)
            bsz = y.shape[0]
            epoch_loss += loss_prev.item() * bsz
            n_samples += bsz
            loss_prev.backward()
            self.t += 1

            with torch.no_grad():
                b_corr = 1. - self.b ** self.t
                for i, p in enumerate(self.trainable_params):
                    self._prev_params[i].copy_(p)
                    self._prev_grads[i].copy_(p.grad)
                    m = self._first_moment[i].lerp_(p.grad, weight=(1. - self.b))
                    p.sub_(m, alpha=self.lr / b_corr)
                    p.grad = None

            loss_curr = criterion(self.model(x), y)
            loss_curr.backward()

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    self._bounce_update(i, p, loss_prev, loss_curr)
                    p.grad = None

        # self.epoch_losses.append(epoch_loss / n_samples)
        return epoch_loss / n_samples

    @torch.inference_mode()
    def test(self, test_loader: torch.utils.data.dataloader.DataLoader, ) -> float:
        self.model.eval()
        correct = 0
        n_samples = 0
        for x, y in test_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            pred = self.model(x).argmax(dim=1)
            correct += pred.eq(y).sum().item()
            n_samples += y.shape[0]

        accuracy = (100. * correct / n_samples)
        return accuracy


class BGD_adam:
    r"""
    Args:
        model (callable): model (nn.Module) containing trainable parameters to optimize via BGD
        lr (float): base learning rate
        _eps (float): epsilon for convex weights (default: 1e-8)
    """

    def __init__(
            self,
            model: nn.Module,
            lr: float = 0.1,
            non_blocking: bool = False,
            _eps: float = 1e-8,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")

        self.model = model
        self.trainable_params: list[nn.Parameter] = []
        self._prev_params: list[torch.Tensor] = []
        self._prev_grads: list[torch.Tensor] = []
        self._first_moment: list[torch.Tensor] = []
        self._second_moment: list[torch.Tensor] = []
        for param in model.parameters():
            if param.requires_grad:
                self.trainable_params.append(param)  # stores a reference (pointer) to the actual model parameters
                self._prev_params.append(torch.empty_like(param))
                self._prev_grads.append(torch.empty_like(param))
                self._first_moment.append(torch.zeros_like(param))
                self._second_moment.append(torch.zeros_like(param))

        self.device = param.device
        self.lr = lr
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps
        self.epoch_losses: list[float] = []
        self.b1 = 0.9
        self.b2 = 0.999
        self.t = 0

    def _bounce_update(self, idx: int, curr_p: torch.Tensor) -> None:
        prev_grad = self._prev_grads[idx]
        curr_grad = curr_p.grad
        if (prev_grad.view(-1) @ curr_grad.view(-1)).item() < 0.:
            # print("bounce!")
            w = curr_grad.sub(prev_grad).div_(2.).sigmoid_()

            # curr_p.mul_(d2).add_(self._prev_params[idx], alpha=d1)
            curr_p.lerp_(self._prev_params[idx], weight=w)

        else:
            # curr_p.data = curr_p - curr_grad * self.lr
            curr_p.sub_(curr_grad, alpha=self.lr)

    def train(self, train_loader: torch.utils.data.DataLoader, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> float:
        self.model.train()
        epoch_loss = 0.
        n_samples = 0
        for x, y in train_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            loss = criterion(self.model(x), y)
            bsz = y.shape[0]
            epoch_loss += loss.item() * bsz
            n_samples += bsz
            loss.backward()
            self.t += 1

            with torch.no_grad():
                b1_corr = 1. - self.b1 ** self.t
                b2_corr = 1. - self.b2 ** self.t

                for i, p in enumerate(self.trainable_params):
                    self._prev_params[i].copy_(p)
                    self._prev_grads[i].copy_(p.grad)
                    m = self._first_moment[i].lerp_(p.grad, weight=(1. - self.b1))
                    v = self._second_moment[i].lerp_(p.grad.square(), weight=(1. - self.b2))
                    denom = v.div(b2_corr).sqrt_().add_(self._eps)  # one temporary tensor creation (allocation)
                    p.addcdiv_(m, denom, value=-self.lr/b1_corr)  # we moved "m.div(b1_corr)" to lr to avoid temp tensor allocation
                    p.grad = None

            criterion(self.model(x), y).backward()

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    self._bounce_update(i, p)
                    p.grad = None

        # self.epoch_losses.append(epoch_loss / n_samples)
        return epoch_loss / n_samples

    @torch.inference_mode()
    def test(self, test_loader: torch.utils.data.dataloader.DataLoader, ) -> float:
        self.model.eval()
        correct = 0
        n_samples = 0
        for x, y in test_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            pred = self.model(x).argmax(dim=1)
            correct += pred.eq(y).sum().item()
            n_samples += y.shape[0]

        accuracy = (100. * correct / n_samples)
        return accuracy


class BGD_alt:
    r"""
    Args:
        model (callable): model (nn.Module) containing trainable parameters to optimize via BGD
        lr (float): base learning rate
        _eps (float): epsilon for convex weights (default: 1e-8)
    """

    def __init__(
            self,
            model: nn.Module,
            lr: float = 0.1,
            non_blocking: bool = False,
            _eps: float = 1e-8,
            beta: float = 0.9,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")

        self.model = model
        self.trainable_params: list[nn.Parameter] = []
        self._prev_params: list[torch.Tensor] = []
        self._prev_grads: list[torch.Tensor] = []
        self._first_moment: list[torch.Tensor] = []
        for param in model.parameters():
            if param.requires_grad:
                self.trainable_params.append(param)  # stores a reference/alias (pointer) to the actual model parameters
                self._prev_params.append(torch.empty_like(param))
                self._prev_grads.append(torch.zeros_like(param))
                self._first_moment.append(torch.zeros_like(param))

        self.device = param.device
        self.lr = lr
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps
        self.epoch_losses: list[float] = []
        self.b = beta
        self.t = np.zeros(len(self.trainable_params))


    def _bounce_update(self, idx: int, curr_p: torch.Tensor) -> None:
        prev_grad = self._prev_grads[idx]
        curr_grad = curr_p.grad
        if (prev_grad.view(-1) @ curr_grad.view(-1)) < 0.:
            # print("bounce!")
            self._first_moment[idx].zero_()
            self._first_moment[idx].lerp_(curr_grad, weight=(1. - self.b))
            self.t[idx] = 1

            w = curr_grad.sub(prev_grad).sigmoid_()

            # curr_p.mul_(d2).add_(self._prev_params[idx], alpha=d1)
            curr_p.lerp_(self._prev_params[idx], weight=w)

        else:
            curr_p.sub_(curr_grad, alpha=self.lr)


    def train(self, train_loader: torch.utils.data.DataLoader, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> float:
        self.model.train()
        epoch_loss = 0.
        n_samples = 0
        for x, y in train_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            loss = criterion(self.model(x), y)
            bsz = y.shape[0]
            epoch_loss += loss.item() * bsz
            n_samples += bsz
            loss.backward()

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    p_clone = p.clone()
                    self._bounce_update(i, p)
                    self._prev_params[i].copy_(p_clone)
                    self._prev_grads[i].copy_(p.grad)
                    p.grad = None

        # self.epoch_losses.append(epoch_loss / n_samples)
        return epoch_loss / n_samples

    @torch.inference_mode()
    def test(self, test_loader: torch.utils.data.dataloader.DataLoader, ) -> float:
        self.model.eval()
        correct = 0
        n_samples = 0
        for x, y in test_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            pred = self.model(x).argmax(dim=1)
            correct += pred.eq(y).sum().item()
            n_samples += y.shape[0]

        accuracy = (100. * correct / n_samples)
        return accuracy


class BGD_s:
    r"""
    Args:
        model (callable): model (nn.Module) containing trainable parameters to optimize via BGD
        lr (float): base learning rate
        _eps (float): epsilon for convex weights (default: 1e-8)
    """

    def __init__(
            self,
            model: nn.Module,
            lr: float = 0.1,
            non_blocking: bool = False,
            _eps: float = 1e-8,
            beta: float = 0.9,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")

        self.model = model
        self.trainable_params: list[nn.Parameter] = []
        self._prev_params: list[torch.Tensor] = []
        self._first_moment: list[torch.Tensor] = []
        for param in model.parameters():
            if param.requires_grad:
                self.trainable_params.append(param)  # stores a reference (pointer) to the actual model parameters
                self._prev_params.append(torch.empty_like(param))
                self._first_moment.append(torch.zeros_like(param))

        self.device = param.device
        self.lr = lr
        self.non_blocking: bool = non_blocking
        self._eps: float = _eps
        self.epoch_losses: list[float] = []
        self.b = beta
        self.t = np.zeros(len(self.trainable_params))

    def _bounce_update(self, idx: int, curr_p: torch.Tensor) -> None:
        prev_grad = self._first_moment[idx]
        curr_grad = curr_p.grad
        if (prev_grad.view(-1) @ curr_grad.view(-1)) < 0.:
            # print("bounce!")
            self._first_moment[idx].zero_()
            self._first_moment[idx].lerp_(curr_grad, weight=(1. - self.b))
            self.t[idx] = 1

            w = curr_grad.sub(prev_grad).sigmoid_()

            # curr_p.mul_(d2).add_(self._prev_params[idx], alpha=d1)
            curr_p.lerp_(self._prev_params[idx], weight=w)

        else:
            # curr_p.data = curr_p - curr_grad * self.lr
            curr_p.sub_(curr_grad, alpha=self.lr)

    def train(self, train_loader: torch.utils.data.DataLoader, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> float:
        self.model.train()
        epoch_loss = 0.
        n_samples = 0
        for x, y in train_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            loss = criterion(self.model(x), y)
            bsz = y.shape[0]
            epoch_loss += loss.item() * bsz
            n_samples += bsz
            loss.backward()
            self.t += 1

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    self._prev_params[i].copy_(p)
                    m = self._first_moment[i].lerp_(p.grad, weight=(1. - self.b))
                    b_corr = 1. - self.b ** self.t[i]
                    p.sub_(m, alpha=self.lr/b_corr)
                    p.grad = None

            criterion(self.model(x), y).backward()

            with torch.no_grad():
                for i, p in enumerate(self.trainable_params):
                    self._bounce_update(i, p)
                    p.grad = None

        # self.epoch_losses.append(epoch_loss / n_samples)
        return epoch_loss / n_samples

    @torch.inference_mode()
    def test(self, test_loader: torch.utils.data.dataloader.DataLoader, ) -> float:
        self.model.eval()
        correct = 0
        n_samples = 0
        for x, y in test_loader:
            x, y = x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)
            pred = self.model(x).argmax(dim=1)
            correct += pred.eq(y).sum().item()
            n_samples += y.shape[0]

        accuracy = (100. * correct / n_samples)
        return accuracy
