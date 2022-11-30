
def apply_defense(args, gradients):
    # TODO
    # ######################## defense start ############################
    # ######################## defense1: dp ############################
    # if self.apply_laplace and self.dp_strength != 0.0 or self.apply_gaussian and self.dp_strength != 0.0:
    #     location = 0.0
    #     threshold = 0.2  # 1e9
    #     if self.apply_laplace:
    #         with torch.no_grad():
    #             scale = self.dp_strength
    #             # clip 2-norm per sample
    #             print("norm of gradients:", torch.norm(pred_a_gradients_clone, dim=1), torch.max(torch.norm(pred_a_gradients_clone, dim=1)))
    #             norm_factor_a = torch.div(torch.max(torch.norm(pred_a_gradients_clone, dim=1)),
    #                                       threshold + 1e-6).clamp(min=1.0)
    #             # add laplace noise
    #             dist_a = torch.distributions.laplace.Laplace(location, scale)
    #             pred_a_gradients_clone = torch.div(pred_a_gradients_clone, norm_factor_a) + \
    #                                      dist_a.sample(pred_a_gradients_clone.shape).to(self.device)
    #             print("norm of gradients after laplace:", torch.norm(pred_a_gradients_clone, dim=1), torch.max(torch.norm(pred_a_gradients_clone, dim=1)))
    #     elif self.apply_gaussian:
    #         with torch.no_grad():
    #             scale = self.dp_strength

    #             print("norm of gradients:", torch.norm(pred_a_gradients_clone, dim=1), torch.max(torch.norm(pred_a_gradients_clone, dim=1)))
    #             norm_factor_a = torch.div(torch.max(torch.norm(pred_a_gradients_clone, dim=1)),
    #                                       threshold + 1e-6).clamp(min=1.0)
    #             pred_a_gradients_clone = torch.div(pred_a_gradients_clone, norm_factor_a) + \
    #                                      torch.normal(location, scale, pred_a_gradients_clone.shape).to(self.device)
    #             print("norm of gradients after gaussian:", torch.norm(pred_a_gradients_clone, dim=1), torch.max(torch.norm(pred_a_gradients_clone, dim=1)))
    # ######################## defense2: gradient sparsification ############################
    # elif self.apply_grad_spar:
    #     with torch.no_grad():
    #         percent = self.grad_spars / 100.0
    #         if self.gradients_res_a is not None and \
    #                 pred_a_gradients_clone.shape[0] == self.gradients_res_a.shape[0]:
    #             pred_a_gradients_clone = pred_a_gradients_clone + self.gradients_res_a
    #         a_thr = torch.quantile(torch.abs(pred_a_gradients_clone), percent)
    #         self.gradients_res_a = torch.where(torch.abs(pred_a_gradients_clone).double() < a_thr.item(),
    #                                               pred_a_gradients_clone.double(), float(0.)).to(self.device)
    #         pred_a_gradients_clone = pred_a_gradients_clone - self.gradients_res_a
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
    # # elif self.apply_gc:
    # #     tensor_pruner = TensorPruner(zip_percent=self.gc_preserved_percent)
    # #     tensor_pruner.update_thresh_hold(pred_a_gradients_clone)
    # #     pred_a_gradients_clone = tensor_pruner.prune_tensor(pred_a_gradients_clone)
    # #     tensor_pruner.update_thresh_hold(pred_b_gradients_clone)
    # #     pred_b_gradients_clone = tensor_pruner.prune_tensor(pred_b_gradients_clone)
    # # elif self.apply_lap_noise:
    # #     dp = DPLaplacianNoiseApplyer(beta=self.noise_scale)
    # #     pred_a_gradients_clone = dp.laplace_mech(pred_a_gradients_clone)
    # #     pred_b_gradients_clone = dp.laplace_mech(pred_b_gradients_clone)
    # elif self.apply_discrete_gradients:
    #     # print(pred_a_gradients_clone)
    #     pred_a_gradients_clone = multistep_gradient(pred_a_gradients_clone, bins_num=self.discrete_gradients_bins, bound_abs=self.discrete_gradients_bound)
    #     pred_b_gradients_clone = multistep_gradient(pred_b_gradients_clone, bins_num=self.discrete_gradients_bins, bound_abs=self.discrete_gradients_bound)
    # ######################## defense end ############################

    return gradients