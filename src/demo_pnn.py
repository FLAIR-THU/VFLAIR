import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import phe as paillier

from utils.paillier_torch import PaillierTensor, PaillierMSELoss

key_length = 128

keypair = paillier.generate_paillier_keypair(n_length=key_length)
pk, sk = keypair

ct_1 = pk.encrypt(13)
ct_2 = pk.encrypt(0.5)
ct_3 = ct_1 + ct_2

pt_1 = PaillierTensor([ct_1, ct_2, ct_3])
torch.testing.assert_close(
    pt_1.decrypt(sk), torch.Tensor([13, 0.5, 13.5]), atol=1e-5, rtol=1
)

pt_2 = pt_1 + torch.Tensor([0.4, 0.1, 0.2])
torch.testing.assert_close(
    pt_2.decrypt(sk), torch.Tensor([13.4, 0.6, 13.7]), atol=1e-5, rtol=1
)

pt_3 = pt_1 * torch.Tensor([1, 2.5, 0.5])
torch.testing.assert_close(
    pt_3.decrypt(sk), torch.Tensor([13, 1.25, 6.75]), atol=1e-5, rtol=1
)

pt_4 = pt_1 - torch.Tensor([0.7, 0.3, 0.6])
torch.testing.assert_close(
    pt_4.decrypt(sk), torch.Tensor([14.3, 0.2, 12.9]), atol=1e-5, rtol=1
)

pt_5 = pt_1 * 2
torch.testing.assert_close(
    pt_5.decrypt(sk), torch.Tensor([26, 1, 27]), atol=1e-5, rtol=1
)

print("basic test all passed")

torch.manual_seed(42)

plin = torch.nn.Linear(3, 1)
pcriterion = PaillierMSELoss()

x = torch.randn(10, 3, requires_grad=True)
y = x @ torch.Tensor([2, 6, 3]).reshape(3, 1) + 1
optimizer = torch.optim.SGD(plin.parameters(), lr=0.3)

print(plin.weight, plin.bias)

for i in range(100):
    optimizer.zero_grad()
    y_pred = plin(x)
    py_pred = PaillierTensor([[pk.encrypt(x) for x in xs] for xs in y_pred.tolist()])
    pcriterion(py_pred, y)
    torch.autograd.backward(y_pred, pcriterion.p_backward().decrypt(sk))
    optimizer.step()

print(i, plin.weight, plin.bias)
