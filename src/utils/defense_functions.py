# import torch
# import numpy as np
# import random


# # Privacy Preserving Deep Learning
# def bound(grad, gamma):
#     if grad < -gamma:
#         return -gamma
#     elif grad > gamma:
#         return gamma
#     else:
#         return grad


# def generate_lap_noise(beta):
#     # beta = sensitivity / epsilon
#     u1 = np.random.random()
#     u2 = np.random.random()
#     if u1 <= 0.5:
#         n_value = -beta * np.log(1. - u2)
#     else:
#         n_value = beta * np.log(u2)
#     # print(n_value)
#     return n_value


# def sigma(x, c, sensitivity):
#     x = 2. * c * sensitivity / x
#     return x


# def get_grad_num(layer_grad_list):
#     num_grad = 0
#     num_grad_per_layer = []
#     for grad_tensor in layer_grad_list:
#         num_grad_this_layer = 0
#         if len(grad_tensor.shape) == 1:
#             num_grad_this_layer = grad_tensor.shape[0]
#         elif len(grad_tensor.shape) == 2:
#             num_grad_this_layer = grad_tensor.shape[0] * grad_tensor.shape[1]
#         num_grad += num_grad_this_layer
#         num_grad_per_layer.append(num_grad_this_layer)
#     return num_grad, num_grad_per_layer


# def get_grad_layer_id_by_grad_id(num_grad_per_layer, id):
#     id_layer = 0
#     id_temp = id
#     for num_grad_this_layer in num_grad_per_layer:
#         id_temp -= num_grad_this_layer
#         if id_temp >= 0:
#             id_layer += 1
#         else:
#             id_temp += num_grad_this_layer
#             break
#     return id_layer, id_temp


# def get_one_grad_by_grad_id(layer_grad_list, num_grad_per_layer, id):
#     id_layer, id_in_this_layer = get_grad_layer_id_by_grad_id(num_grad_per_layer, id)
#     grad_this_layer = layer_grad_list[id_layer]
#     if len(grad_this_layer.shape) == 1:
#         the_grad = grad_this_layer[id_in_this_layer]
#     else:
#         the_grad = grad_this_layer[id_in_this_layer // grad_this_layer.shape[1]][
#             id_in_this_layer % grad_this_layer.shape[1]]
#     return the_grad


# def set_one_grad_by_grad_id(layer_grad_list, num_grad_per_layer, id, set_value):
#     id_layer, id_in_this_layer = get_grad_layer_id_by_grad_id(num_grad_per_layer, id)
#     grad_this_layer = layer_grad_list[id_layer]
#     if len(grad_this_layer.shape) == 1:
#         layer_grad_list[id_layer][id_in_this_layer] = set_value
#     else:
#         layer_grad_list[id_layer][id_in_this_layer // grad_this_layer.shape[1]][
#             id_in_this_layer % grad_this_layer.shape[1]] = set_value


# def dp_gc_ppdl(epsilon, sensitivity, layer_grad_list, theta_u, gamma, tau):
#     grad_num, num_grad_per_layer = get_grad_num(layer_grad_list)
#     c = int(theta_u * grad_num)
#     # print("c:", c)
#     # exit()
#     epsilon1 = 8. / 9 * epsilon
#     epsilon2 = 2. / 9 * epsilon
#     used_grad_ids = []
#     really_useful_grad_ids = []
#     done_grad_count = 0
#     while 1:
#         r_tau = generate_lap_noise(sigma(epsilon1, c, sensitivity))
#         while 1:
#             while 1:
#                 grad_id = random.randint(0, grad_num - 1)
#                 if grad_id not in used_grad_ids:
#                     used_grad_ids.append(grad_id)
#                     break
#                 if len(used_grad_ids) == grad_num:
#                     return
#             grad = get_one_grad_by_grad_id(layer_grad_list, num_grad_per_layer, grad_id)
#             r_w = generate_lap_noise(2 * sigma(epsilon1, c, sensitivity))
#             if abs(bound(grad, gamma)) + r_w >= tau + r_tau:
#                 r_w_ = generate_lap_noise(sigma(epsilon2, c, sensitivity))
#                 set_one_grad_by_grad_id(layer_grad_list, num_grad_per_layer, grad_id, bound((grad + r_w_), gamma))
#                 really_useful_grad_ids.append(grad_id)
#                 done_grad_count += 1
#                 if done_grad_count >= c:
#                     for id in range(0, grad_num):
#                         if id not in really_useful_grad_ids:
#                             set_one_grad_by_grad_id(layer_grad_list, num_grad_per_layer, id, 0.)
#                     # print("really_useful_grad_ids:", really_useful_grad_ids)
#                     # print("len really_useful_grad_ids:", len(really_useful_grad_ids))
#                     # exit()
#                     return
#                 else:
#                     break


# # Multistep gradient
# def multistep_gradient(tensor, bound_abs, bins_num=12):
#     # Criteo 1e-3
#     max_min = 2 * bound_abs
#     interval = max_min / bins_num
#     tensor_ratio_interval = torch.div(tensor, interval)
#     tensor_ratio_interval_rounded = torch.round(tensor_ratio_interval)
#     tensor_multistep = tensor_ratio_interval_rounded * interval
#     return tensor_multistep


# # Gradient Compression
# class TensorPruner:
#     def __init__(self, zip_percent):
#         self.thresh_hold = 0.
#         self.zip_percent = zip_percent

#     def update_thresh_hold(self, tensor):
#         tensor_copy = tensor.clone().detach()
#         tensor_copy = torch.abs(tensor_copy)
#         survivial_values = torch.topk(tensor_copy.reshape(1, -1),
#                                       int(tensor_copy.reshape(1, -1).shape[1] * self.zip_percent))
#         self.thresh_hold = survivial_values[0][0][-1]

#     def prune_tensor(self, tensor):
#         # whether the tensor to process is on cuda devices
#         background_tensor = torch.zeros(tensor.shape).to(torch.float)
#         if 'cuda' in str(tensor.device):
#             background_tensor = background_tensor.cuda(device=tensor.device)
#         # print("background_tensor", background_tensor)
#         tensor = torch.where(abs(tensor) > self.thresh_hold, tensor, background_tensor)
#         # print("tensor:", tensor)
#         return tensor


# # Differential Privacy(Noisy Gradients)
# class DPLaplacianNoiseApplyer():
#     def __init__(self, beta):
#         self.beta = beta

#     def noisy_count(self):
#         # beta = sensitivity / epsilon
#         beta = self.beta
#         u1 = np.random.random()
#         u2 = np.random.random()
#         if u1 <= 0.5:
#             n_value = -beta * np.log(1. - u2)
#         else:
#             n_value = beta * np.log(u2)
#         n_value = torch.tensor(n_value)
#         # print(n_value)
#         return n_value

#     def laplace_mech(self, tensor):
#         # generate noisy mask
#         # whether the tensor to process is on cuda devices
#         noisy_mask = torch.zeros(tensor.shape).to(torch.float)
#         if 'cuda' in str(tensor.device):
#             noisy_mask = noisy_mask.to(device=tensor.device)
#         noisy_mask = noisy_mask.flatten()
#         for i in range(noisy_mask.shape[0]):
#             noisy_mask[i] = self.noisy_count()
#         noisy_mask = noisy_mask.reshape(tensor.shape)
#         # print("noisy_tensor:", noisy_mask)
#         tensor = tensor + noisy_mask
#         return tensor
