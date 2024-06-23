import sys, os

sys.path.append(os.pardir)

import torch
import torch.nn.functional as F
import time
import numpy as np
import copy
import pickle

from evaluates.attacks.attacker import Attacker
from models.reconstructors import Reconstructor
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res, MSE_PSNR


class DataReconstruction(Attacker):
    def __init__(self, top_vfl, args):
        super().__init__(args)
        self.args = args
        # get information for launching BLI attack
        self.vfl_info = top_vfl.first_epoch_state
        # prepare parameters
        self.device = args.device
        self.num_classes = args.num_classes
        self.k = args.k
        self.models_dict = args.model_list
        self.parties = args.parties
        self.lr = args.attack_configs['lr']
        self.epochs = args.attack_configs['epochs']
        self.early_stop = args.attack_configs['early_stop'] if 'early_stop' in args.attack_configs else 0
        self.early_stop_threshold = args.attack_configs[
            'early_stop_threshold'] if 'early_stop_threshold' in args.attack_configs else 1e-7
        self.label_size = args.num_classes
        self.dummy_active_top_trainable_model = None
        self.optimizer_trainable = None  # construct later
        self.dummy_active_top_non_trainable_model = None
        self.optimizer_non_trainable = None  # construct later
        self.criterion = cross_entropy_for_onehot
        self.file_name = 'attack_result.txt'
        self.exp_res_dir = f'exp_result/main/{args.dataset}/attack/DR/'
        self.exp_res_path = self.exp_res_dir + self.file_name

    def set_seed(self, seed=0):
        # random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    # def calc_label_recovery_rate(self, dummy_label, gt_label):
    #     success = torch.sum(torch.argmax(dummy_label, dim=-1) == torch.argmax(gt_label, dim=-1)).item()
    #     total = dummy_label.shape[0]
    #     return success / total

    def label_to_one_hot(self, target, num_classes=10):
        # print('label_to_one_hot:', target, type(target))
        try:
            _ = target.size()[1]
            # print("use target itself", target.size())
            onehot_target = target.type(torch.float32).to(self.device)
        except:
            target = torch.unsqueeze(target, 1).to(self.device)
            # print("use unsqueezed target", target.size())
            onehot_target = torch.zeros(target.size(0), num_classes, device=self.device)
            onehot_target.scatter_(1, target, 1)
        return onehot_target

    def attack(self):
        self.set_seed(123)

        # load models 
        parameter_path = self.args.exp_res_dir + f'trained_models/parties{self.k}_topmodel{self.args.apply_trainable_layer}_epoch{self.args.main_epochs}/'
        if self.args.apply_defense:
            file_path = parameter_path + f'{self.args.defense_name}_{self.args.defense_configs}.pkl'
        else:
            file_path = parameter_path + 'NoDefense.pkl'
        models_params = torch.load(file_path)
        for ik in range(self.k):
            self.parties[ik].local_model.load_state_dict(models_params[0][ik])
        if self.args.apply_trainable_layer == 1:
            self.parties[self.k - 1].global_model.load_state_dict(models_params[0][self.k])
        for ik in range(self.k):
            self.parties[ik].local_model.eval()
        self.parties[self.k - 1].global_model.eval()

        # collect transmit local model outputs
        transmitted_predicts = [[] for _ in range(self.k)]
        final_predicts = []
        data_loader_list = [self.parties[ik].train_loader for ik in range(self.k)]
        for parties_data in zip(*data_loader_list):
            self.parties_data = parties_data
            self.gt_one_hot_label = self.label_to_one_hot(parties_data[self.k - 1][1], self.num_classes)
            self.gt_one_hot_label = self.gt_one_hot_label.to(self.device)
            predicts = []
            for ik in range(self.k):
                transmitted_predicts[ik].append(self.parties[ik].local_model(parties_data[ik][0]))
                predicts.append(transmitted_predicts[ik][-1])
            final_predicts.append(F.softmax(self.parties[self.k - 1].global_model(predicts), dim=-1))
        print(len(transmitted_predicts[0]), type(transmitted_predicts[0][0]), transmitted_predicts[0][0].shape)
        for ik in range(self.k):
            transmitted_predicts[ik] = torch.cat(transmitted_predicts[ik], dim=0)
        # final_predicts = torch.cat(final_predicts,dim=0)
        means = [torch.mean(transmitted_predicts[ik]) for ik in range(self.k - 1)]
        stds = [torch.std(transmitted_predicts[ik]) ** 2 for ik in range(self.k - 1)]

        # prepare reconstructor
        noise_shape = [self.args.model_list[str(ik)]['input_dim'] if 'input_dim' in self.args.model_list[str(ik)] else
                       self.args.half_dim[ik] for ik in range(self.k - 1)]
        print(f"noise_shape={noise_shape}")
        self.reconstructor = [
            Reconstructor(noise_shape[ik]).to(self.device) for ik in range(self.k - 1)]
        self.reconstructor_optimizer = [
            torch.optim.Adam(self.reconstructor[ik].parameters(), lr=self.lr)
            for ik in range(self.k - 1)]

        start_time = time.time()

        for i_epoch in range(self.epochs):
            # train reconstrutor to reconstruct data
            for ik in range(self.k - 1):
                self.reconstructor[ik].train()
            kl_loss_criterion = torch.nn.KLDivLoss(reduction="batchmean")
            i = -1
            for parties_data in zip(*data_loader_list):
                i += 1
                self.parties_data = parties_data
                self.gt_one_hot_label = self.label_to_one_hot(parties_data[self.k - 1][1], self.num_classes)
                self.gt_one_hot_label = self.gt_one_hot_label.to(self.device)
                current_batch_size = parties_data[self.k - 1][0].size(0)
                predicts = []
                for ik in range(self.k - 1):
                    noise = torch.randn((current_batch_size, noise_shape[ik])).to(self.device)
                    predicts.append(self.parties[ik].local_model(self.reconstructor[ik](noise)))
                predicts.append(self.parties[self.k - 1].local_model(parties_data[self.k - 1][0]))
                final_predict, predict_loss = self.parties[self.k - 1].aggregate(predicts, self.gt_one_hot_label)
                final_predict = F.softmax(final_predict, dim=-1)
                # assert 1 == 0
                norm_loss = 0.0
                for ik in range(self.k - 1):
                    norm_loss += (torch.abs(means[ik] - torch.mean(predicts[ik])) + torch.abs(
                        stds[ik] - torch.std(predicts[ik]) ** 2))
                # try:
                #     final_predict = final_predict.log()
                # except:
                #     final_predict = final_predict
                # KL_loss = kl_loss_criterion(final_predict.log(), final_predicts[i])
                KL_loss = kl_loss_criterion(final_predict, final_predicts[i])
                loss = predict_loss + norm_loss - KL_loss

                for ik in range(self.k - 1):
                    self.reconstructor_optimizer[ik].zero_grad()
                loss.backward(retain_graph=True)
                for ik in range(self.k - 1):
                    self.reconstructor_optimizer[ik].step()

            if i_epoch % 10 == 9:
                print(f"i_epoch={i_epoch}, testing")
                # test reconstructor with test data
                for ik in range(self.k - 1):
                    self.reconstructor[ik].eval()
                test_data_loader_list = [self.parties[ik].test_loader for ik in range(self.k)]
                for test_parties_data in zip(*test_data_loader_list):
                    # gt_val_one_hot_label = self.label_to_one_hot(test_parties_data[self.k-1][1], self.num_classes)
                    # gt_val_one_hot_label = gt_val_one_hot_label.to(self.device)
                    current_batch_size = test_parties_data[self.k - 1][0].size(0)
                    reconstruct_data = []  # should be of length self.k-1
                    real_data = []
                    for ik in range(self.k - 1):
                        noise = torch.randn((current_batch_size, noise_shape[ik])).to(self.device)
                        reconstruct_data.append(self.reconstructor[ik](noise))
                        real_data.append(test_parties_data[ik][0])
                    # reconstruct_data.append(parties_data[self.k-1][0])
                    # real_data.append(parties_data[self.k-1][0])
                for ik in range(self.k - 1):
                    reconstruct_data[ik] = reconstruct_data[ik].reshape(real_data[ik].shape)
                reconstruct_data = torch.cat(reconstruct_data, dim=0)
                real_data = torch.cat(real_data, dim=0)

                mse, psnr = MSE_PSNR(real_data, reconstruct_data)

                end_time = time.time()
                print(f'batch_size=%d,class_num=%d,psnr=%lf,time_used=%lf' % (
                self.args.batch_size, self.label_size, psnr, end_time - start_time))

        end_time = time.time()
        print(f'batch_size=%d,class_num=%d,psnr=%lf,time_used=%lf' % (
        self.args.batch_size, self.label_size, psnr, end_time - start_time))

        print(f"DataReconstruction, if self.args.apply_defense={self.args.apply_defense}")
        if self.args.apply_defense == True:
            exp_result = f"bs|num_class|psnr,%d|%d|%lf (AttackConfig: %s) (Defense: %s %s)" % (
            self.args.batch_size, self.label_size, psnr, str(self.args.attack_configs), self.args.defense_name,
            str(self.args.defense_configs))
        else:
            exp_result = f"bs|num_class|psnr,%d|%d|%lf (AttackConfig: %s)" % (
            self.args.batch_size, self.label_size, psnr, str(self.args.attack_configs))
        append_exp_res(self.exp_res_path, exp_result)
