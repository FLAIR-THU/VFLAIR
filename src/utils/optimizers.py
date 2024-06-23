import math
import torch
from torch.optim.optimizer import Optimizer
from time import time
from torch import Tensor
from typing import List
import matplotlib.pyplot as plt

############# most of the optimizer code are copied from torch v1.7.0 code #############

last_whole_model_params_list = []
new_whole_model_params_list = []
batch_cos_list = []
near_minimum = False


class MaliciousSGD(Optimizer):

    def __init__(self, params, lr=1e-2, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, gamma_lr_scale_up=1.0, min_grad_to_process=1e-4):

        self.last_parameters_grads = []
        self.gamma_lr_scale_up = gamma_lr_scale_up
        self.min_grad_to_process = min_grad_to_process
        self.min_ratio = 1.0
        self.max_ratio = 5.0

        self.certain_grad_ratios = torch.tensor([])

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(MaliciousSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MaliciousSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        id_group = 0
        for i in range(len(self.param_groups)):
            self.last_parameters_grads.append([])

        for group in self.param_groups:

            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            start = time()
            id_parameter = 0

            for p in group['params']:
                if p.grad is None:
                    continue

                if weight_decay != 0:
                    p.grad.data.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(p.grad.data).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, p.grad.data)
                    if nesterov:
                        p.grad.data = p.grad.data.add(momentum, buf)
                    else:
                        p.grad.data = buf

                if not near_minimum:
                    if len(self.last_parameters_grads[id_group]) <= id_parameter:
                        self.last_parameters_grads[id_group].append(p.grad.clone().detach())
                    else:
                        last_parameter_grad = self.last_parameters_grads[id_group][id_parameter]
                        current_parameter_grad = p.grad.clone().detach()
                        ratio_grad_scale_up = 1.0 + self.gamma_lr_scale_up * (
                                    current_parameter_grad / (last_parameter_grad + 1e-7))
                        ratio_grad_scale_up = torch.clamp(ratio_grad_scale_up, self.min_ratio, self.max_ratio)
                        p.grad.mul_(ratio_grad_scale_up)
                end = time()
                current_parameter_grad = p.grad.clone().detach()
                self.last_parameters_grads[id_group][id_parameter] = current_parameter_grad

                p.data.add_(-group['lr'], p.grad.data)

                id_parameter += 1
            id_group += 1

        return loss


class MaliciousSignSGD(Optimizer):

    def __init__(self, params, lr=1e-2, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, gamma_lr_scale_up=1.0, min_grad_to_process=1e-4):

        self.last_parameters_grads = []
        self.gamma_lr_scale_up = gamma_lr_scale_up
        self.min_grad_to_process = min_grad_to_process

        self.certain_grad_ratios = []

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(MaliciousSignSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MaliciousSignSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            # Indicate which module's paras we are processing
            id_parameter = 0

            for p in group['params']:
                if p.grad is None:
                    continue
                if len(self.last_parameters_grads) <= id_parameter:
                    self.last_parameters_grads.append(p.grad.clone().detach().numpy())
                else:
                    last_parameter_grad = self.last_parameters_grads[id_parameter]
                    current_parameter_grad = p.grad.clone().detach().numpy()
                    ratio_grad_scale_up = 1.0 + self.gamma_lr_scale_up * (
                                current_parameter_grad / (last_parameter_grad + 1e-7))
                    grad_shape = current_parameter_grad.shape
                    last_parameter_grad = last_parameter_grad.flatten()
                    grad_length = len(last_parameter_grad)
                    ratio_grad_scale_up = ratio_grad_scale_up.flatten()
                    for i in range(grad_length):
                        if abs(last_parameter_grad[i]) < self.min_grad_to_process:
                            ratio_grad_scale_up[i] = 1.0
                        ratio_grad_scale_up[i] = max(ratio_grad_scale_up[i], 1.0)
                        ratio_grad_scale_up[i] = min(ratio_grad_scale_up[i], 5.0)
                    ratio_grad_scale_up = ratio_grad_scale_up.reshape(grad_shape)

                    self.last_parameters_grads[id_parameter] = current_parameter_grad

                    p.grad = p.grad.mul(torch.tensor(ratio_grad_scale_up))

                id_parameter += 1

                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # sign SGD only uses sign of gradient to update model
                torch.sign(d_p, out=d_p)
                p.data.add_(-group['lr'], d_p)

        return loss


class SignSGD(Optimizer):

    def __init__(self, params, lr=1e-2, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SignSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SignSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                torch.sign(d_p, out=d_p)
                p.data.add_(-group['lr'], d_p)

        return loss


class MaliciousAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, gamma_lr_scale_up=1.0, min_grad_to_process=1e-4):

        # for malicious Adam updates
        self.last_parameters_grads = []
        self.gamma_lr_scale_up = gamma_lr_scale_up
        self.min_grad_to_process = min_grad_to_process
        self.min_ratio = 1.0
        self.max_ratio = 5.0
        self.certain_grad_ratios = torch.tensor([])

        # below are copied from normal Adam updates (torch v1.7.0)
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(MaliciousAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MaliciousAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # record of the history
        id_group = 0  # for each group of parameters in self.param_groups
        for i in range(len(self.param_groups)):
            self.last_parameters_grads.append([])
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []

            # id_parameter = 0 # for p, which representes each parameter in this group with id=id_group
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']
            self.adam(params_with_grad,
                      grads,
                      exp_avgs,
                      exp_avg_sqs,
                      max_exp_avg_sqs,
                      state_steps,
                      group['amsgrad'],
                      beta1,
                      beta2,
                      group['lr'],
                      group['weight_decay'],
                      group['eps'],
                      id_group
                      )
        return loss

    def adam(self,
             params: List[Tensor],
             grads: List[Tensor],
             exp_avgs: List[Tensor],
             exp_avg_sqs: List[Tensor],
             max_exp_avg_sqs: List[Tensor],
             state_steps: List[int],
             amsgrad: bool,
             beta1: float,
             beta2: float,
             lr: float,
             weight_decay: float,
             eps: float,
             id_group: int):
        r"""Functional API that performs Adam algorithm computation."""

        for i, param in enumerate(params):

            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]
            if amsgrad:
                max_exp_avg_sq = max_exp_avg_sqs[i]

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            # step_size = lr / bias_correction1

            # param.addcdiv_(exp_avg, denom, value=-step_size)

            adam_gradient_step = exp_avg / (bias_correction1 * denom)

            # begin malicious gradient scaling up
            if len(self.last_parameters_grads[id_group]) <= i:
                # no history to referred to, so self.last_parameters_grads[id_group][i]==scaled_up_grad==adam_gradient_step
                self.last_parameters_grads[id_group].append(adam_gradient_step.clone().detach())
            else:
                last_parameter_grad = self.last_parameters_grads[id_group][i]
                current_parameter_grad = adam_gradient_step.clone().detach()
                ratio_grad_scale_up = 1.0 + self.gamma_lr_scale_up * (
                            current_parameter_grad / (last_parameter_grad + 1e-7))
                ratio_grad_scale_up = torch.clamp(ratio_grad_scale_up, self.min_ratio, self.max_ratio)
                scaled_up_grad = current_parameter_grad * ratio_grad_scale_up
                # print(f"id_group={id_group}, id_param={i}, ratio_grad_scale_up={ratio_grad_scale_up}, scaled_up_grad={scaled_up_grad}")
                self.last_parameters_grads[id_group][i] = scaled_up_grad

            # print(f"param={param}")
            # print(f"adam_gradient_step={adam_gradient_step}")
            # print(f"scaled_up_grad={self.last_parameters_grads[id_group][i]}")
            # print(f"lr={lr}")
            param.add_((-1) * lr * self.last_parameters_grads[id_group][i])
            # print(f"param={param}")
