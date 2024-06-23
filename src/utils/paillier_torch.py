import functools

import numpy as np
import torch
import torch.nn as nn

HANDLED_FUNCTIONS = {}


def implements(torch_function):
    """Registers a torch function override for PaillierTensor"""

    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


def _ignore_overflow_decrypt(sk):
    def f(x):
        try:
            return sk.decrypt(x)
        except:
            return 0.0

    return f


class PaillierTensor(object):
    """torch.Tensor-like object for Paillier Encryption"""

    def __init__(self, paillier_array):
        if type(paillier_array) == list:
            self._paillier_np_array = np.array(paillier_array)
        elif type(paillier_array) == np.ndarray:
            self._paillier_np_array = paillier_array
        else:
            raise TypeError(f"{type(paillier_array)} is not supported.")
        self.device = torch.device("cpu")

    def __repr__(self):
        return "PaillierTensor"

    def decrypt(self, sk, device="cpu"):
        return torch.Tensor(
            np.vectorize(_ignore_overflow_decrypt(sk))(self._paillier_np_array)
        ).to(device)

    def tensor(self, sk=None):
        if sk is not None:
            return self.decypt(sk)
        else:
            return torch.zeros(self._paillier_np_array.shape)

    def numpy(self):
        return self._paillier_np_array

    def detach(self):
        return self

    def cpu(self):
        return self

    def size(self):
        return torch.Size(self._paillier_np_array.shape)

    @property
    def T(self):
        return PaillierTensor(self._paillier_np_array.T)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(
                issubclass(t, (torch.Tensor, PaillierTensor)) for t in types
        ):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    @implements(torch.add)
    def add(input, other):
        if type(other) in [int, float]:
            return PaillierTensor(input._paillier_np_array + other)
        elif type(other) in [
            torch.Tensor,
            torch.nn.parameter.Parameter,
            PaillierTensor,
        ]:
            return PaillierTensor(
                input._paillier_np_array + other.detach().cpu().numpy()
            )
        else:
            raise NotImplementedError(f"{type(other)} is not supported.")

    @implements(torch.sub)
    def sub(input, other):
        if type(other) in [int, float]:
            return PaillierTensor(input._paillier_np_array + (-1) * other)
        elif type(other) in [torch.Tensor, PaillierTensor]:
            return PaillierTensor(
                input._paillier_np_array + (-1) * other.detach().cpu().numpy()
            )
        else:
            raise NotImplementedError(f"{type(other)} is not supported.")

    @implements(torch.mul)
    def mul(input, other):
        if type(other) in [int, float]:
            return PaillierTensor(input._paillier_np_array * other)
        elif type(other) in [torch.Tensor, PaillierTensor]:
            return PaillierTensor(
                input._paillier_np_array * other.detach().cpu().numpy()
            )
        else:
            raise NotImplementedError(f"{type(other)} is not supported.")

    @implements(torch.matmul)
    def matmul(x, other):
        return PaillierTensor(
            np.matmul(x._paillier_np_array, other.detach().cpu().numpy())
        )

    @implements(torch.sum)
    def sum(input, dim=None):
        return PaillierTensor(np.array(np.sum(input._paillier_np_array, axis=dim)))

    @implements(torch.nn.functional.linear)
    def linear(x, w, bias):
        return torch.matmul(x, w.T) + bias

    def __add__(self, other):
        return torch.add(self, other)

    def __iadd__(self, other):
        self = torch.add(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return torch.sub(self, other)

    def __isub__(self, other):
        self = torch.sub(self, other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        return torch.mul(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __iter__(self):
        yield from self._paillier_np_array

class PaillierMSELoss(nn.Module):
    def __init__(self):
        super(PaillierMSELoss, self).__init__()
        self.ctx = None
        self.p_gradients = None

    def forward(self, y_pred, y_true):
        self.ctx = y_pred, y_true
        return None

    def p_backward(self):
        y_pred, y_true = self.ctx
        return (y_pred - y_true) * (1 / y_true.shape[0])
