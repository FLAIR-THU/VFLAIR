import sys, os
sys.path.append(os.pardir)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np
import time
import copy

# from models.vision import resnet18, MLP2
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res
# from evaluates.attacks.attack_api import apply_attack
from evaluates.defenses.defense_api import apply_defense
from evaluates.defenses.defense_functions import *
from utils.constants import *
import utils.constants as shared_var
from utils.marvell_functions import KL_gradient_perturb
from utils.noisy_label_functions import add_noise
from utils.noisy_sample_functions import noisy_sample
from evaluates.attacks.attack_api import AttackerLoader


tf.compat.v1.enable_eager_execution() 

STOPPING_ACC = {'mnist': 0.977, 'cifar10': 0.80, 'cifar100': 0.40}  # add more about stopping accuracy for different datasets when calculating the #communication-rounds needed


class MainTaskVFL(object):

    def __init__(self, args):
        self.args = args
        self.k = args.k
        self.device = args.device
        self.dataset_name = args.dataset
        # self.train_dataset = args.train_dst
        # self.val_dataset = args.test_dst
        # self.half_dim = args.half_dim
        self.epochs = args.main_epochs
        self.lr = args.main_lr
        self.batch_size = args.batch_size
        self.models_dict = args.model_list
        # self.num_classes = args.num_classes
        # self.num_class_list = args.num_class_list
        self.num_classes = args.num_classes
        self.exp_res_dir = args.exp_res_dir

        self.exp_res_path = args.exp_res_path
        self.parties = args.parties
        
        self.Q = args.Q # FedBCD

        self.parties_data = None
        self.gt_one_hot_label = None
        self.clean_one_hot_label  = None
        self.pred_list = []
        self.pred_list_clone = []
        self.pred_gradients_list = []
        self.pred_gradients_list_clone = []
        
        # FedBCD related
        self.local_pred_list = []
        self.local_pred_list_clone = []
        self.local_pred_gradients_list = []
        self.local_pred_gradients_list_clone = []
        
        self.loss = None
        self.train_acc = None
        self.flag = 1

        # Early Stop
        self.early_stop_threshold = args.early_stop_threshold
        self.final_epoch = 0

        # some state of VFL throughout training process
        self.first_epoch_state = None
        self.middle_epoch_state = None
        self.final_state = None
        # self.final_epoch_state = None # <-- this is save in the above parameters
    
    def pred_transmit(self): # Active party gets pred from passive parties
        for ik in range(self.k):
            pred, pred_detach = self.parties[ik].give_pred()

            # defense applied on pred
            if self.args.apply_defense == True and self.args.apply_dp == True :
                # Only add noise to pred when launching FR attack(attaker_id=self.k-1)
                if (self.k-1 in self.args.attacker_id) and (ik != self.k-1): # attaker won't defend its own attack
                    pred_detach = self.launch_defense(pred_detach, "pred") 

            pred_clone = torch.autograd.Variable(pred_detach, requires_grad=True).to(self.args.device)

            if ik == (self.k-1): # Active party update local pred
                self.parties[ik].update_local_pred(pred_clone)
            if ik < (self.k-1): # Passive party sends pred for aggregation
                self.parties[self.k-1].receive_pred(pred_clone, ik) 
    
    def gradient_transmit(self):  # Active party sends gradient to passive parties
        gradient = self.parties[self.k-1].give_gradient() # gradient_clone
        
        # defense applied on gradients
        if self.args.apply_defense == True and self.args.apply_dcor == False and self.args.apply_mid == False and self.args.apply_cae == False:
            if (self.k-1) not in self.args.attacker_id:
                gradient = self.launch_defense(gradient, "gradients")   
        if self.args.apply_dcae == True:
            if (self.k-1) not in self.args.attacker_id:
                gradient = self.launch_defense(gradient, "gradients")  
            
        # active party update local gradient
        self.parties[self.k-1].update_local_gradient(gradient[self.k-1])
        # active party transfer gradient to passive parties
        for ik in range(self.k-1):
            self.parties[ik].receive_gradient(gradient[ik])
        return
    
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

    def LR_Decay(self,i_epoch):
        for ik in range(self.k):
            self.parties[ik].LR_decay(i_epoch)
        self.parties[self.k-1].global_LR_decay(i_epoch)
        
    def train_batch(self, parties_data, batch_label):
        '''
        batch_label: self.gt_one_hot_label   may be noisy
        '''
        encoder = self.args.encoder
        if self.args.apply_cae:
            assert encoder != None, "[error] encoder is None for CAE"
            _, gt_one_hot_label = encoder(batch_label) 
            # _, test_one_hot_label = encoder(torch.tensor([[0.0,1.0],[1.0,0.0]]).to(self.args.device))
            # print("one hot label for DCAE 1.0 of 2 class", test_one_hot_label)   
            # for DCAE-1.0-2class, <[0.0,1.0],[1.0,0.0]> ==> <[0.9403, 0.0597],[0.0568, 0.9432]>        
        else:
            gt_one_hot_label = batch_label
        
        self.parties[self.k-1].gt_one_hot_label = gt_one_hot_label
        # allocate data to each party
        for ik in range(self.k):
            self.parties[ik].obtain_local_data(parties_data[ik][0])

        # ====== normal vertical federated learning ======
        torch.autograd.set_detect_anomaly(True)
        # ======== FedBCD ============
        if self.args.BCD_type == 'p' or self.Q ==1 : # parallel FedBCD & noBCD situation
            for q in range(self.Q):
                if q == 0: 
                    # exchange info between parties
                    self.pred_transmit() 
                    self.gradient_transmit() 
                    # update parameters for all parties
                    for ik in range(self.k):
                        self.parties[ik].local_backward()
                    self.parties[self.k-1].global_backward()
                else: # FedBCD: additional iterations without info exchange
                    # for passive party, do local update without info exchange
                    for ik in range(self.k-1):
                        _pred, _pred_clone= self.parties[ik].give_pred() 
                        self.parties[ik].local_backward() 
                    # for active party, do local update without info exchange
                    _pred, _pred_clone = self.parties[self.k-1].give_pred() 
                    _gradient = self.parties[self.k-1].give_gradient()
                    self.parties[self.k-1].global_backward()
                    self.parties[self.k-1].local_backward()
        else: # Sequential FedBCD
            for q in range(self.Q):
                if q == 0: 
                    #first iteration, active party gets pred from passsive party
                    self.pred_transmit() 
                    _gradient = self.parties[self.k-1].give_gradient(self)
                    # active party: update parameters 
                    self.parties[self.k-1].local_backward()
                    self.parties[self.k-1].global_backward()
                else: 
                    # active party do additional iterations without info exchange
                    self.parties[self.k-1].give_pred(self)
                    _gradient = self.parties[self.k-1].give_gradient(self)
                    self.parties[self.k-1].local_backward()
                    self.parties[self.k-1].global_backward()

            # active party transmit grad to passive parties
            self.gradient_transmit() 

            # passive party do Q iterations
            for _q in range(self.Q):
                for ik in range(self.k-1): 
                    _pred, _pred_clone= self.parties[ik].give_pred() 
                    self.parties[ik].local_backward() 
        # ============= FedBCD ===================
        
        # ###### Noisy Label Attack #######
        # convert back to clean label to get true acc
        if self.args.apply_nl==True:
            real_batch_label = self.clean_one_hot_label
        else:
            real_batch_label = batch_label
        # ###### Noisy Label Attack #######

        pred = self.parties[self.k-1].global_pred
        loss = self.parties[self.k-1].global_loss
        predict_prob = F.softmax(pred, dim=-1)
        if self.args.apply_cae:
            predict_prob = encoder.decoder(predict_prob)
        suc_cnt = torch.sum(torch.argmax(predict_prob, dim=-1) == torch.argmax(real_batch_label, dim=-1)).item()
        train_acc = suc_cnt / predict_prob.shape[0]
        return loss.item(), train_acc

    def train(self):

        print_every = 1

        for ik in range(self.k):
            self.parties[ik].prepare_data_loader(batch_size=self.batch_size)

        # ####### missing feature attack ######
        # load attack features
        if self.args.apply_mf == True:
            assert 'missing_rate' in self.args.attack_configs, 'need parameter: missing_rate'
            assert 'party' in self.args.attack_configs, 'need parameter: party'
            attacker_id = self.args.attack_configs['party']
            missing_rate = self.args.attack_configs['missing_rate']
        # ####### missing feature attack ######

        test_acc = 0.0
        # Early Stop
        last_loss = 1000000
        early_stop_count = 0
        LR_passive_list = []
        LR_active_list = []
        for i_epoch in range(self.epochs):
            postfix = {'train_loss': 0.0, 'train_acc': 0.0, 'test_acc': 0.0}
            i = -1
            data_loader_list = [self.parties[ik].train_loader for ik in range(self.k)]
            for parties_data in zip(*data_loader_list):
                self.gt_one_hot_label = self.label_to_one_hot(parties_data[self.k-1][1], self.num_classes)
                self.gt_one_hot_label = self.gt_one_hot_label.to(self.device)
                
                # ###### Noisy Label Attack ######
                if self.args.apply_nl==True:
                    # noisy label
                    self.clean_one_hot_label = self.gt_one_hot_label
                    self.gt_one_hot_label = add_noise(self.args, self.gt_one_hot_label)
                    # if self.clean_one_hot_label.equal(self.gt_one_hot_label):
                    #     print('Noise not good')
                    # else:
                    #     print('Noise OK')
                # ###### Noisy Label Attack ######
                self.parties_data = parties_data

                i += 1
                for ik in range(self.k):
                    self.parties[ik].local_model.train()
                self.parties[self.k-1].global_model.train()
                # ====== train batch (start) ======
                if i == 0 and i_epoch == 0:
                    self.first_epoch_state = self.save_state(True)
                elif i_epoch == self.epochs//2 and i == 0:
                    self.middle_epoch_state = self.save_state(True)

                self.loss, self.train_acc = self.train_batch(self.parties_data,self.gt_one_hot_label)
                #  self.gt_one_hot_label

                if i == 0 and i_epoch == 0:
                    self.first_epoch_state.update(self.save_state(False))
                elif i_epoch == self.epochs//2 and i == 0:
                    self.middle_epoch_state.update(self.save_state(False))
                # ====== train batch (end) ======

            # if self.args.apply_attack == True:
            #     if (self.args.attack_name in LABEL_INFERENCE_LIST) and i_epoch==1:
            #         print('Launch Label Inference Attack, Only train 1 epoch')
            #         break    

            self.trained_models = self.save_state(True)
            if self.args.save_model == True:
                self.save_trained_models()

            # LR decay
            self.LR_Decay(i_epoch)
            # LR record
            LR_passive_list.append(self.parties[0].give_current_lr())
            LR_active_list.append(self.parties[1].give_current_lr())

            # validation
            if (i + 1) % print_every == 0:
                print("validate and test")
                for ik in range(self.k):
                    self.parties[ik].local_model.eval()
                self.parties[self.k-1].global_model.eval()
                
                suc_cnt = 0
                sample_cnt = 0

                with torch.no_grad():
                    data_loader_list = [self.parties[ik].test_loader for ik in range(self.k)]
                    for parties_data in zip(*data_loader_list):
                        # print("test", parties_data[0][0].size(),parties_data[self.k-1][0].size(),parties_data[self.k-1][1].size())

                        gt_val_one_hot_label = self.label_to_one_hot(parties_data[self.k-1][1], self.num_classes)
                        gt_val_one_hot_label = gt_val_one_hot_label.to(self.device)

                        pred_list = []
                        for ik in range(self.k):
                            # ####### missing feature attack ######
                            if (self.args.apply_mf == True) and (ik in attacker_id) and (np.random.random() < missing_rate):
                                #print('attacker:',ik)
                                pred_list.append(torch.zeros_like(self.parties[ik].local_model(parties_data[ik][0])))
                            # ####### missing feature attack ######
                            # ####### Noisy Sample #########
                            elif self.args.apply_ns == True and (ik in self.args.attack_configs['party']):
                                scale = self.args.attack_configs['noise_lambda']
                                pred_list.append( self.parties[ik].local_model( noisy_sample(parties_data[ik][0],scale) ) )
                            # ####### Noisy Sample #########
                            else:
                                pred_list.append(self.parties[ik].local_model(parties_data[ik][0]))
                        
                        test_logit, test_loss = self.parties[self.k-1].aggregate(pred_list, gt_val_one_hot_label, test="True")

                        enc_predict_prob = F.softmax(test_logit, dim=-1)
                        if self.args.apply_cae == True:
                            dec_predict_prob = self.args.encoder.decoder(enc_predict_prob)
                            predict_label = torch.argmax(dec_predict_prob, dim=-1)
                        else:
                            predict_label = torch.argmax(enc_predict_prob, dim=-1)

                        actual_label = torch.argmax(gt_val_one_hot_label, dim=-1)
                        sample_cnt += predict_label.shape[0]
                        suc_cnt += torch.sum(predict_label == actual_label).item()
                    self.test_acc = suc_cnt / float(sample_cnt)
                    postfix['train_loss'] = self.loss
                    postfix['train_acc'] = '{:.2f}%'.format(self.train_acc * 100)
                    postfix['test_acc'] = '{:.2f}%'.format(self.test_acc * 100)
                    # tqdm_train.set_postfix(postfix)
                    print('Epoch {}% \t train_loss:{:.2f} train_acc:{:.2f} test_acc:{:.2f}'.format(
                        i_epoch, self.loss, self.train_acc, self.test_acc))
                    
                    self.final_epoch = i_epoch
        self.final_state = self.save_state(True) 
        self.final_state.update(self.save_state(False)) 
        self.final_state.update(self.save_party_data()) 
           
        #LR scopes
        # xx = [i for i in range(len(LR_passive_list))]
        # plt.figure()
        # plt.plot(xx,LR_passive_list,'o-', color='b', alpha=0.8, linewidth=1, label='passive0')
        # plt.plot(xx,LR_active_list,'o-', color='r', alpha=0.8, linewidth=1, label='active1')
        # plt.legend()
        # plt.savefig('./exp_result/LR_Scope.png')

        return self.test_acc


    def train_graph(self):
        test_acc = 0.0
        # Early Stop
        last_loss = 1000000
        early_stop_count = 0

        for i_epoch in range(self.epochs):
            postfix = {'train_loss': 0.0, 'train_acc': 0.0, 'test_acc': 0.0}
            self.parties_data = [(self.parties[ik].train_data, self.parties[ik].train_label) for ik in range(self.k)]
            for ik in range(self.k):
                self.parties[ik].local_model.train()
            self.parties[self.k-1].global_model.train()

            self.gt_one_hot_label = self.label_to_one_hot(self.parties_data[self.k-1][1], self.num_classes)
            self.gt_one_hot_label = self.gt_one_hot_label.to(self.device)
            # ###### Noisy Label Attack ######
            if self.args.apply_nl==True:
                # noisy label
                self.clean_one_hot_label = self.gt_one_hot_label
                self.gt_one_hot_label = add_noise(self.args, self.gt_one_hot_label)
            # ###### Noisy Label Attack ######

            # ====== train batch (start) ======            
            if i_epoch == 0:
                self.first_epoch_state = self.save_state(True)
            elif i_epoch == self.epochs//2:
                self.middle_epoch_state = self.save_state(True)
            
            self.loss, self.train_acc = self.train_batch(self.parties_data, self.gt_one_hot_label)
        
            if  i_epoch == 0:
                self.first_epoch_state.update(self.save_state(False))
            elif i_epoch == self.epochs//2:
                self.middle_epoch_state.update(self.save_state(False))
            # ====== train batch (end) ======   

            # if self.args.apply_attack == True:
            #     if (self.args.attack_name in LABEL_INFERENCE_LIST) and i_epoch==1:
            #         print('Launch Label Inference Attack, Only train 1 epoch')
            #         break         

            # validation
            print("validate and test")
            for ik in range(self.k):
                self.parties[ik].local_model.eval()
            self.parties[self.k-1].global_model.eval()
            
            suc_cnt = 0
            sample_cnt = 0
            with torch.no_grad():
                parties_data = [(self.parties[ik].test_data, self.parties[ik].test_label) for ik in range(self.k)]
                
                gt_val_one_hot_label = self.label_to_one_hot(parties_data[self.k-1][1], self.num_classes)
                gt_val_one_hot_label = gt_val_one_hot_label.to(self.device)

                pred_list = []
                for ik in range(self.k):
                    pred_list.append(self.parties[ik].local_model(parties_data[ik][0]))
                test_logit, test_loss = self.parties[self.k-1].aggregate(pred_list, gt_val_one_hot_label)

                enc_predict_prob = F.softmax(test_logit, dim=-1)
                if self.args.apply_cae == True:
                    dec_predict_prob = self.args.encoder.decoder(enc_predict_prob)
                    predict_label = torch.argmax(dec_predict_prob, dim=-1)
                else:
                    predict_label = torch.argmax(enc_predict_prob, dim=-1)

                actual_label = torch.argmax(gt_val_one_hot_label, dim=-1)
                predict_label = predict_label[parties_data[self.k-1][0][2]]
                actual_label = actual_label[parties_data[self.k-1][0][2]]
                
                sample_cnt += predict_label.shape[0]
                suc_cnt += torch.sum(predict_label == actual_label).item()
                self.test_acc = suc_cnt / float(sample_cnt)
                postfix['train_loss'] = self.loss
                postfix['train_acc'] = '{:.2f}%'.format(self.train_acc * 100)
                postfix['test_acc'] = '{:.2f}%'.format(self.test_acc * 100)
                # tqdm_train.set_postfix(postfix)
                print('Epoch {}% \t train_loss:{:.2f} train_acc:{:.2f} test_acc:{:.2f}'.format(
                    i_epoch, self.loss, self.train_acc, self.test_acc))
                
                self.final_epoch = i_epoch
                # Early Stop Assessment
                # if self.loss < last_loss:       
                #     early_stop_count = 0
                # else:
                #     early_stop_count +=1
                # last_loss = self.loss
                
                # if early_stop_count >= self.early_stop_threshold:
                #     
                #     break
        self.final_state = self.save_state(True) 
        self.final_state.update(self.save_state(False)) 
        self.final_state.update(self.save_party_data()) 
        
        return self.test_acc


    def save_state(self, BEFORE_MODEL_UPDATE=True):
        if BEFORE_MODEL_UPDATE:
            return {
                "model": [copy.deepcopy(self.parties[ik].local_model) for ik in range(self.args.k)],
                "global_model":copy.deepcopy(self.parties[self.args.k-1].global_model),
                # type(model) = <class 'xxxx.ModelName'>
                "model_names": [str(type(self.parties[ik].local_model)).split('.')[-1].split('\'')[-2] for ik in range(self.args.k)]+[str(type(self.parties[self.args.k-1].global_model)).split('.')[-1].split('\'')[-2]]
            
            }
        else:
            return {
                # "model": [copy.deepcopy(self.parties[ik].local_model) for ik in range(self.args.k)]+[self.parties[self.args.k-1].global_model],
                "data": copy.deepcopy(self.parties_data), 
                "label": copy.deepcopy(self.gt_one_hot_label),
                "predict": [copy.deepcopy(self.parties[ik].local_pred_clone) for ik in range(self.k)],
                "gradient": [copy.deepcopy(self.parties[ik].local_gradient) for ik in range(self.k)],
                "local_model_gradient": [copy.deepcopy(self.parties[ik].weights_grad_a) for ik in range(self.k)],
                "train_acc": copy.deepcopy(self.train_acc),
                "loss": copy.deepcopy(self.loss),
                "global_pred":self.parties[self.k-1].global_pred,
                "final_model": [copy.deepcopy(self.parties[ik].local_model) for ik in range(self.args.k)],
                "final_global_model":copy.deepcopy(self.parties[self.args.k-1].global_model),
                
            }

    def save_party_data(self):
        return {
            "aux_data": [copy.deepcopy(self.parties[ik].aux_data) for ik in range(self.k)],
            "train_data": [copy.deepcopy(self.parties[ik].train_data) for ik in range(self.k)],
            "test_data": [copy.deepcopy(self.parties[ik].test_data) for ik in range(self.k)],
            "aux_label": [copy.deepcopy(self.parties[ik].aux_label) for ik in range(self.k)],
            "train_label": [copy.deepcopy(self.parties[ik].train_label) for ik in range(self.k)],
            "test_label": [copy.deepcopy(self.parties[ik].test_label) for ik in range(self.k)],
            "aux_loader": [copy.deepcopy(self.parties[ik].aux_loader) for ik in range(self.k)],
            "train_loader": [copy.deepcopy(self.parties[ik].train_loader) for ik in range(self.k)],
            "test_loader": [copy.deepcopy(self.parties[ik].test_loader) for ik in range(self.k)],
            "batchsize": self.args.batch_size,
            "num_classes": self.args.num_classes
        }
        
        
    def save_trained_models(self):
        dir_path = self.exp_res_dir + f'trained_models/parties{self.k}_topmodel{self.args.apply_trainable_layer}_epoch{self.epochs}/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if self.args.apply_defense:
            file_path = dir_path + f'{self.args.defense_name}_{self.args.defense_configs}.pkl'
        else:
            file_path = dir_path + 'NoDefense.pkl'
        torch.save(([self.trained_models["model"][i].state_dict() for i in range(len(self.trained_models["model"]))],
                    self.trained_models["model_names"]), 
                  file_path)

    def evaluate_attack(self):
        self.attacker = AttackerLoader(self, self.args)
        if self.attacker != None:
            attack_acc = self.attacker.attack()
        return attack_acc

    def launch_defense(self, gradients_list, _type):
        
        if _type == 'gradients':
            return apply_defense(self.args, _type, gradients_list)
        elif _type == 'pred':
            return apply_defense(self.args, _type, gradients_list)
        else:
            # further extention
            return gradients_list

    def calc_label_recovery_rate(self, dummy_label, gt_label):
        success = torch.sum(torch.argmax(dummy_label, dim=-1) == torch.argmax(gt_label, dim=-1)).item()
        total = dummy_label.shape[0]
        return success / total
