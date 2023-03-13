import torch
import numpy as np
import random


# Privacy Preserving Deep Learning
def bound(grad, gamma):
    if grad < -gamma:
        return -gamma
    elif grad > gamma:
        return gamma
    else:
        return grad

def generate_lap_noise(beta):
    # beta = sensitivity / epsilon
    u1 = np.random.random()
    u2 = np.random.random()
    if u1 <= 0.5:
        n_value = -beta * np.log(1. - u2)
    else:
        n_value = beta * np.log(u2)
    # print(n_value)
    return n_value

def sigma(x, c, sensitivity):
    x = 2. * c * sensitivity / x
    return x

def get_grad_num(layer_grad_list):
    num_grad = 0
    num_grad_per_layer = []
    for grad_tensor in layer_grad_list:
        num_grad_this_layer = 0
        if len(grad_tensor.shape) == 1:
            num_grad_this_layer = grad_tensor.shape[0]
        elif len(grad_tensor.shape) == 2:
            num_grad_this_layer = grad_tensor.shape[0] * grad_tensor.shape[1]
        num_grad += num_grad_this_layer
        num_grad_per_layer.append(num_grad_this_layer)
    return num_grad, num_grad_per_layer

def get_grad_layer_id_by_grad_id(num_grad_per_layer, id):
    id_layer = 0
    id_temp = id
    for num_grad_this_layer in num_grad_per_layer:
        id_temp -= num_grad_this_layer
        if id_temp >= 0:
            id_layer += 1
        else:
            id_temp += num_grad_this_layer
            break
    return id_layer, id_temp

def get_one_grad_by_grad_id(layer_grad_list, num_grad_per_layer, id):
    id_layer, id_in_this_layer = get_grad_layer_id_by_grad_id(num_grad_per_layer, id)
    grad_this_layer = layer_grad_list[id_layer]
    if len(grad_this_layer.shape) == 1:
        the_grad = grad_this_layer[id_in_this_layer]
    else:
        the_grad = grad_this_layer[id_in_this_layer // grad_this_layer.shape[1]][
            id_in_this_layer % grad_this_layer.shape[1]]
    return the_grad

def set_one_grad_by_grad_id(layer_grad_list, num_grad_per_layer, id, set_value):
    id_layer, id_in_this_layer = get_grad_layer_id_by_grad_id(num_grad_per_layer, id)
    grad_this_layer = layer_grad_list[id_layer]
    if len(grad_this_layer.shape) == 1:
        layer_grad_list[id_layer][id_in_this_layer] = set_value
    else:
        layer_grad_list[id_layer][id_in_this_layer // grad_this_layer.shape[1]][
            id_in_this_layer % grad_this_layer.shape[1]] = set_value

def dp_gc_ppdl(epsilon, sensitivity, layer_grad_list, theta_u, gamma, tau):
    grad_num, num_grad_per_layer = get_grad_num(layer_grad_list)
    c = int(theta_u * grad_num)
    epsilon1 = 8. / 9 * epsilon
    epsilon2 = 2. / 9 * epsilon
    used_grad_ids = []
    really_useful_grad_ids = []
    done_grad_count = 0
    while 1:
        r_tau = generate_lap_noise(sigma(epsilon1, c, sensitivity))
        while 1:
            while 1:
                grad_id = random.randint(0, grad_num - 1)
                if grad_id not in used_grad_ids:
                    used_grad_ids.append(grad_id)
                    break
                if len(used_grad_ids) == grad_num:
                    return
            grad = get_one_grad_by_grad_id(layer_grad_list, num_grad_per_layer, grad_id)
            r_w = generate_lap_noise(2 * sigma(epsilon1, c, sensitivity))
            if abs(bound(grad, gamma)) + r_w >= tau + r_tau:
                r_w_ = generate_lap_noise(sigma(epsilon2, c, sensitivity))
                set_one_grad_by_grad_id(layer_grad_list, num_grad_per_layer, grad_id, bound((grad + r_w_), gamma))
                really_useful_grad_ids.append(grad_id)
                done_grad_count += 1
                if done_grad_count >= c:
                    for id in range(0, grad_num):
                        if id not in really_useful_grad_ids:
                            set_one_grad_by_grad_id(layer_grad_list, num_grad_per_layer, id, 0.)
                    # print("really_useful_grad_ids:", really_useful_grad_ids)
                    # print("len really_useful_grad_ids:", len(really_useful_grad_ids))
                    # exit()
                    return
                else:
                    break


# Multistep gradient
def multistep_gradient(tensor, bound_abs, bins_num=12):
    # Criteo 1e-3
    max_min = 2 * bound_abs
    interval = max_min / bins_num
    tensor_ratio_interval = torch.div(tensor, interval)
    tensor_ratio_interval_rounded = torch.round(tensor_ratio_interval)
    tensor_multistep = tensor_ratio_interval_rounded * interval
    return tensor_multistep


# Gradient Compression
class TensorPruner:
    def __init__(self, zip_percent):
        self.thresh_hold = 0.
        self.zip_percent = zip_percent

    def update_thresh_hold(self, tensor):
        tensor_copy = tensor.clone().detach()
        tensor_copy = torch.abs(tensor_copy)
        survivial_values = torch.topk(tensor_copy.reshape(1, -1),
                                      int(tensor_copy.reshape(1, -1).shape[1] * self.zip_percent))
        self.thresh_hold = survivial_values[0][0][-1]

    def prune_tensor(self, tensor):
        # whether the tensor to process is on cuda devices
        background_tensor = torch.zeros(tensor.shape).to(torch.float)
        if 'cuda' in str(tensor.device):
            background_tensor = background_tensor.cuda(device=tensor.device)
        # print("background_tensor", background_tensor)
        tensor = torch.where(abs(tensor) > self.thresh_hold, tensor, background_tensor)
        # print("tensor:", tensor)
        return tensor


# Differential Privacy(Noisy Gradients)
class DPLaplacianNoiseApplyer():
    def __init__(self, beta):
        self.beta = beta

    def noisy_count(self):
        # beta = sensitivity / epsilon
        beta = self.beta
        u1 = np.random.random()
        u2 = np.random.random()
        if u1 <= 0.5:
            n_value = -beta * np.log(1. - u2)
        else:
            n_value = beta * np.log(u2)
        n_value = torch.tensor(n_value)
        # print(n_value)
        return n_value

    def laplace_mech(self, tensor):
        # generate noisy mask
        # whether the tensor to process is on cuda devices
        noisy_mask = torch.zeros(tensor.shape).to(torch.float)
        if 'cuda' in str(tensor.device):
            noisy_mask = noisy_mask.to(device=tensor.device)
        noisy_mask = noisy_mask.flatten()
        for i in range(noisy_mask.shape[0]):
            noisy_mask[i] = self.noisy_count()
        noisy_mask = noisy_mask.reshape(tensor.shape)
        # print("noisy_tensor:", noisy_mask)
        tensor = tensor + noisy_mask
        return tensor

def LaplaceDP(args, original_object):
    dp_strength = args.defense_configs['dp_strength']
    original_object = original_object[0]
    if dp_strength > 0.0:
        location = 0.0
        threshold = 0.2  # 1e9
        new_object = []
        with torch.no_grad():
            scale = dp_strength
            for ik in range(len(original_object)):
                # clip 2-norm per sample
                # print("norm of gradients:", torch.norm(original_object[ik], dim=1), torch.max(torch.norm(original_object[ik], dim=1)))
                norm_factor_a = torch.div(torch.max(torch.norm(original_object[ik], dim=1)),
                                            threshold + 1e-6).clamp(min=1.0)
                # add laplace noise
                dist_a = torch.distributions.laplace.Laplace(location, scale)
                new_object.append(torch.div(original_object[ik], norm_factor_a) + \
                                        dist_a.sample(original_object[ik].shape).to(args.device))
            # print("norm of gradients after laplace:", torch.norm(original_object, dim=1), torch.max(torch.norm(original_object, dim=1)))
        return new_object
    else:
        return original_object

def GaussianDP(args, original_object):
    original_object = original_object[0]
    dp_strength = args.defense_configs['dp_strength']
    if dp_strength > 0.0:
        location = 0.0
        threshold = 0.2  # 1e9
        new_object = []
        with torch.no_grad():
            scale = dp_strength
            for ik in range(len(original_object)):
                # print("norm of gradients:", torch.norm(original_object[ik], dim=1), torch.max(torch.norm(original_object[ik], dim=1)))
                norm_factor_a = torch.div(torch.max(torch.norm(original_object[ik], dim=1)),
                                        threshold + 1e-6).clamp(min=1.0)
                new_object.append(torch.div(original_object[ik], norm_factor_a) + \
                                        torch.normal(location, scale, original_object[ik].shape).to(args.device))
                # print("norm of gradients after gaussian:", torch.norm(original_object, dim=1), torch.max(torch.norm(original_object, dim=1)))
        return new_object
    else:
        return original_object

def GradientSparsification(args, original_object):
    original_object = original_object[0]
    grad_spars_ratio = args.defense_configs['gradient_sparse_rate']
    while grad_spars_ratio > 1.0:
        grad_spars_ratio = grad_spars_ratio / 100.0
    if grad_spars_ratio > 0.0:
        new_object = []
        with torch.no_grad():
        #     percent = grad_spars_ratio / 100.0 # percent to drop
        #     if self.gradients_res_a is not None and \
        #             pred_a_gradients_clone.shape[0] == self.gradients_res_a.shape[0]:
        #         pred_a_gradients_clone = pred_a_gradients_clone + self.gradients_res_a
            for ik in range(len(original_object)):
                a_thr = torch.quantile(torch.abs(original_object[ik]), grad_spars_ratio)
                gradients_res_a = torch.where(torch.abs(original_object[ik]).double() < a_thr.item(),
                                                        original_object[ik].double(), float(0.)).to(args.device)
                new_object.append(original_object[ik] - gradients_res_a)
        return new_object
    else:
        return original_object
    
def discrete(original_tensor,W):
    _mu = float(torch.mean(original_tensor).item())  #np.mean(original_object) 
    _sigma = torch.var(original_tensor).item()  #np.std(original_object)
    _sigma = float(_sigma**0.5)           
    A = np.linspace(_mu-_sigma, _mu+_sigma, num=W , endpoint=True, retstep=False, dtype=None)

    new_tensor = torch.empty(original_tensor.shape[0],original_tensor.shape[1])
    for i in range(original_tensor.shape[0]):
        for j in range(original_tensor.shape[1]):
            element = original_tensor[i][j].item()
            if element <= A[0]:
                new_tensor[i][j]=A[0]
            elif element >= A[-1]:
                new_tensor[i][j]=A[-1]
            else:
                for nodes in A:
                    if element > nodes:
                        new_tensor[i][j]=nodes
                        break
    return new_tensor

def DiscreteSGD(args, original_object):
    original_object = original_object[0] # list [tensor1 tensor2]
    W = args.defense_configs['bin_numbers']+1
    
    new_object = []
    if W >= 2:
        with torch.no_grad():
            for ik in range(len(original_object)):
                new_object.append(discrete(original_object[ik],W))
    else:
        print('Error: bin_numbers should be > 1')
        return original_object
    return new_object
    # TODO
    # ######################## defense start ############################
    # ######################## defense3: marvell ############################
    # elif self.apply_marvell and self.marvell_s != 0 and self.num_classes == 2:
    #     # for marvell, change label to [0,1]
    #     marvell_y = []
    #     for i in range(len(gt_one_hot_label)):
    #         marvell_y.append(int(gt_one_hot_label[i][1]))
    #     marvell_y = np.array(marvell_y)
    #     shared_var.batch_y = np.asarray(marvell_y)
    #     logdir = 'marvell_logs/main_task/{}_logs/{}'.format(self.dataset_name, time.strftime("%Y%m%d-%H%M%S"))
    #     writer = tf.summary.create_file_writer(logdir)
    #     shared_var.writer = writer
    #     with torch.no_grad():
    #         pred_a_gradients_clone = KL_gradient_perturb(pred_a_gradients_clone, self.marvell_s)
    #         pred_a_gradients_clone = pred_a_gradients_clone.to(self.device)
    # ######################## defense5: ppdl, GradientCompression, laplace_noise, DiscreteSGD ############################
    # # elif self.apply_ppdl:
    # #     dp_gc_ppdl(epsilon=1.8, sensitivity=1, layer_grad_list=[pred_a_gradients_clone], theta_u=self.ppdl_theta_u, gamma=0.001, tau=0.0001)
    # #     dp_gc_ppdl(epsilon=1.8, sensitivity=1, layer_grad_list=[pred_b_gradients_clone], theta_u=self.ppdl_theta_u, gamma=0.001, tau=0.0001)
    # elif self.apply_discrete_gradients:
    #     # print(pred_a_gradients_clone)
    #     pred_a_gradients_clone = multistep_gradient(pred_a_gradients_clone, bins_num=self.discrete_gradients_bins, bound_abs=self.discrete_gradients_bound)
    #     pred_b_gradients_clone = multistep_gradient(pred_b_gradients_clone, bins_num=self.discrete_gradients_bins, bound_abs=self.discrete_gradients_bound)
    # ######################## defense end ############################

    # return gradients
