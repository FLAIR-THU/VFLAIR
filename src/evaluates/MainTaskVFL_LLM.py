import sys, os
sys.path.append(os.pardir)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np
import random
import time
import copy
from sklearn.metrics import roc_auc_score,matthews_corrcoef
import warnings

# from models.vision import resnet18, MLP2
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res, multiclass_auc
from utils.communication_protocol_funcs import get_size_of

# from evaluates.attacks.attack_api import apply_attack
from evaluates.defenses.defense_api import apply_defense
from evaluates.defenses.defense_functions import *
from utils.constants import *
import utils.constants as shared_var
from utils.marvell_functions import KL_gradient_perturb
from utils.noisy_label_functions import add_noise
from utils.noisy_sample_functions import noisy_sample
from utils.communication_protocol_funcs import compress_pred,Cache,ins_weight
from evaluates.attacks.attack_api import AttackerLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tf.compat.v1.enable_eager_execution() 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

MODEL_PATH = {'bert-base-uncased': "/home/DAIR/guzx/.cache/huggingface/hub/bert-base-uncased",
"bertweet-base-sentiment-analysis": "/home/DAIR/guzx/.cache/huggingface/hub/bertweet-base-sentiment-analysis",
"Bert-sequence-classification": "/home/DAIR/guzx/.cache/huggingface/hub/Bert-sequence-classification",
"toxic-bert": "/home/DAIR/guzx/.cache/huggingface/hub/toxic-bert",
"textattackbert-base-uncased-CoLA": "/home/DAIR/guzx/.cache/huggingface/hub/textattackbert-base-uncased-CoLA",
"geckosbert-base-uncased-finetuned-glue-cola": "/home/DAIR/guzx/.cache/huggingface/hub/geckosbert-base-uncased-finetuned-glue-cola",

}

STOPPING_ACC = {'mnist': 0.977, 'cifar10': 0.80, 'cifar100': 0.40,'diabetes':0.69,\
'nuswide': 0.88, 'breast_cancer_diagnose':0.88,'adult_income':0.84,'cora':0.72,\
'avazu':0.83,'criteo':0.74,'nursery':0.99,'credit':0.82, 'news20':0.8, 'cola_public':0.8}  # add more about stopping accuracy for different datasets when calculating the #communication-rounds needed


class MainTaskVFL_LLM(object):

    def __init__(self, args):
        self.args = args
        self.k = args.k
        # self.k_server = args.k_server
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
        # self.servers = args.servers

        
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
        self.stopping_iter = 0
        self.stopping_time = 0.0
        self.stopping_commu_cost = 0
        self.communication_cost = 0


        # Early Stop
        self.early_stop_threshold = args.early_stop_threshold
        self.final_epoch = 0
        self.current_epoch = 0
        self.current_step = 0

        # some state of VFL throughout training process
        self.first_epoch_state = None
        self.middle_epoch_state = None
        self.final_state = None
        # self.final_epoch_state = None # <-- this is save in the above parameters

        self.num_update_per_batch = args.num_update_per_batch
        self.num_batch_per_workset = args.Q #args.num_batch_per_workset
        self.max_staleness = self.num_update_per_batch*self.num_batch_per_workset 

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
        # for ik in range(self.k):
        #     self.parties[ik].LR_decay(i_epoch)
        self.parties[self.k-1].global_LR_decay(i_epoch)
    
    # def gradient_transmit(self):  # Active party sends gradient to passive parties
    #     gradient = self.parties[self.k-1].give_gradient() # gradient_clone

    #     if len(gradient)>1:
    #         for _i in range(len(gradient)-1):
    #             self.communication_cost += get_size_of(gradient[_i+1])#MB

    #     # defense applied on gradients
    #     if self.args.apply_defense == True and self.args.apply_dcor == False and self.args.apply_mid == False and self.args.apply_cae == False:
    #         if (self.k-1) in self.args.defense_configs['party']:
    #             gradient = self.launch_defense(gradient, "gradients")   
    #     if self.args.apply_dcae == True:
    #         if (self.k-1) in self.args.defense_configs['party']:
    #             gradient = self.launch_defense(gradient, "gradients")  
            
    #     # active party update local gradient
    #     self.parties[self.k-1].update_local_gradient(gradient[self.k-1])
        
    #     # active party transfer gradient to passive parties
    #     for ik in range(self.k-1):
    #         self.parties[ik].receive_gradient(gradient[ik])
    #     return

    def pred_transmit(self): 
        # Active party gets pred from passive parties
        for ik in range(self.k-1):
            pred, pred_detach, input_shape = self.parties[ik].give_pred()
            self.parties[ik].input_shape = input_shape

            # defense applied on pred
            if self.args.apply_defense == True and self.args.apply_dp == True :
                # Only add noise to pred when launching FR attack(attaker_id=self.k-1)
                if (ik in self.args.defense_configs['party']) and (ik != self.k-1): # attaker won't defend its own attack
                    pred_detach = torch.tensor(self.launch_defense(pred_detach, "pred")) 
            
            ########### communication_protocols ###########
            if self.args.communication_protocol in ['Quantization','Topk']:
                pred_detach = compress_pred( self.args ,pred_detach , self.parties[ik].local_gradient,\
                                self.current_epoch, self.current_step).to(self.args.device)
            ########### communication_protocols ###########
            pred_clone = torch.autograd.Variable(pred_detach, requires_grad=True).to(self.args.device)
            self.parties[ik].update_local_pred(pred_clone)

            # Party sends pred/attention_mask/input_shape for aggregation
            self.parties[self.k-1].input_shape = input_shape
            self.parties[self.k-1].receive_attention_mask(self.parties[ik].local_batch_attention_mask)
            self.parties[self.k-1].receive_pred(pred_clone, ik) 
            self.communication_cost += get_size_of(pred_clone)+get_size_of(self.parties[ik].local_batch_attention_mask)+\
                get_size_of( torch.tensor(input_shape) ) #MB
            
    def global_pred_transmit(self):
        # active party give global pred to passive party
        final_pred = self.parties[self.k-1].aggregate(self.parties[self.k-1].pred_received,self.parties[self.k-1].local_batch_attention_mask, test="True")
        
        for ik in range(self.k-1):
            self.communication_cost += get_size_of(final_pred)
            self.parties[ik].global_pred = final_pred
    
    def global_loss_transmit(self):
        # passive party give loss to active party -- used to update global model
        global_loss = self.parties[0].cal_loss() 
        self.communication_cost += get_size_of(global_loss)
        self.parties[self.k-1].global_loss = global_loss
    
    def inference(self, inference_data = 'test'):
        # current_model_type = self.args.model_list['0']['type']
        # full_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH[current_model_type]).to(self.args.device)

        print(' ========= Inference ==========')
        postfix = {'test_acc': 0.0}
        for ik in range(self.k):
            self.parties[ik].prepare_data_loader(batch_size=self.batch_size)
        self.parties[self.k-1].global_model.eval()
        
                
        suc_cnt = 0
        sample_cnt = 0

        test_preds = []
        test_targets = []
        test_predict_labels = []
        test_actual_labels = []


        scores = []
        targets = []
        with torch.no_grad():
            data_loader_list = [self.parties[ik].test_loader for ik in range(self.k-1)]
            for parties_data in zip(*data_loader_list):
                # parties_data[0]: (data, label, mask)
                # parties_data[k-1]: (None,None,None)  no data for active party

                parties_data = [ [_data[0].to(self.device),_data[1].to(self.device),_data[2].to(self.device)] for _data in parties_data]

                if self.args.dataset == 'jigsaw_toxic':
                    gt_val_one_hot_label = parties_data[0][1]
                else:
                    gt_val_one_hot_label = self.label_to_one_hot(parties_data[0][1], self.num_classes)
                # gt_val_one_hot_label = gt_val_one_hot_label.to(self.device)
                # parties_data = [ [_data[0].to(self.device),_data[1].to(self.device)] for _data in parties_data]
                

                pred_list = []
                for ik in range(self.k-1): # Passive data local predict
                    # _local_pred, input_shape = self.parties[ik].local_model(parties_data[ik][0])
                    _local_pred, input_shape = self.parties[ik].local_model(parties_data[ik][0],attention_mask=parties_data[ik][2])
                    self.parties[self.k-1].input_shape = input_shape
                    self.parties[self.k-1].local_batch_attention_mask = parties_data[0][2]
                    self.parties[ik].input_shape = input_shape
                    self.parties[ik].local_batch_attention_mask = parties_data[0][2]
                    pred_list.append(_local_pred)
                
                test_logit = self.parties[self.k-1].aggregate(pred_list,self.parties[self.k-1].local_batch_attention_mask, test="True")
                # print('test_logit:',type(test_logit),test_logit.shape)
                
                # val_pred = full_model(parties_data[0][0]) # ,return_dict = 'True'
                # print('val_pred:',val_pred)
                # print('test_logit:',test_logit)

                if self.args.dataset == 'jigsaw_toxic':
                    sm = torch.sigmoid(test_logit).cpu().detach().numpy()
                    print('sm:',type(sm),sm.shape)
                    scores.extend(sm) 
                    binary_scores = [s >= 0.5 for s in scores]
                    binary_scores = np.stack(binary_scores)
                    targets += gt_val_one_hot_label.cpu()
                else:  
                    enc_predict_prob = test_logit
                    enc_predict_prob = F.softmax(test_logit, dim=-1)

                    predict_label = torch.argmax(enc_predict_prob, dim=-1)
                    actual_label = torch.argmax(gt_val_one_hot_label, dim=-1)

                    test_preds.append(list(enc_predict_prob.detach().cpu().numpy()))
                    test_targets.append(list(gt_val_one_hot_label.detach().cpu().numpy()))

                    test_predict_labels.extend( list(predict_label.detach().cpu()) )
                    test_actual_labels.extend( list(actual_label.detach().cpu()) )
        
                    sample_cnt += predict_label.shape[0]
                    suc_cnt += torch.sum(predict_label == actual_label).item()

                del(parties_data) # remove from cuda
            

            if self.args.dataset == 'jigsaw_toxic':
                binary_scores = [s >= 0.5 for s in scores]
                binary_scores = np.stack(binary_scores)
                scores = np.stack(scores) #[instance_num,6]
                targets = np.stack(targets) #[instance_num,6]
                print('scores:',scores.shape)
                print('targets:',targets.shape)

                auc_scores = []

                for class_idx in range(scores.shape[1]): # 1-6
                    mask = targets[:, class_idx] != -1
                    print('===== class_idx:',class_idx)
                    print('mask:',type(mask), mask.shape )
                    target_binary = targets[mask, class_idx]
                    class_scores = scores[mask, class_idx]
                    try:
                        # print('target_binary:',type(target_binary),target_binary.shape,target_binary[:5])
                        # print('class_scores:',type(class_scores),class_scores.shape,class_scores[:5])
                        auc = roc_auc_score(target_binary, class_scores)
                        print('auc=',auc)
                        auc_scores.append(auc)
                    except Exception:
                        warnings.warn(
                            "Only one class present in y_true. ROC AUC score is not defined in that case. Set to nan for now."
                        )
                        auc_scores.append(np.nan)
                mean_auc = np.mean(auc_scores)
                self.test_auc = mean_auc

                postfix['test_auc'] = '{:.2f}%'.format(self.test_auc * 100)
                exp_result = 'test_auc:{:.2f}'.format(self.test_auc)
                return exp_result,self.test_auc

            else:
                self.test_acc = suc_cnt / float(sample_cnt) # ACC

                test_preds = np.vstack(test_preds)
                test_targets = np.vstack(test_targets)
                self.test_auc = np.mean(multiclass_auc(test_targets, test_preds)) # AUC

                # print('test_predict_labels:',np.array(test_predict_labels).shape,np.array(test_predict_labels)[:5])
                # print('test_actual_labels:',np.array(test_actual_labels).shape,np.array(test_actual_labels)[:5])

                self.test_mcc = matthews_corrcoef(np.array(test_predict_labels),np.array(test_actual_labels) ) # MCC

                postfix['test_acc'] = '{:.2f}%'.format(self.test_acc * 100)
                postfix['test_auc'] = '{:.2f}%'.format(self.test_auc * 100)
                postfix['test_mcc'] = '{:.2f}%'.format(self.test_mcc * 100)

                exp_result = 'test_acc:{:.2f} test_auc:{:.2f} test_mcc:{:.2f}'.format(self.test_acc, self.test_auc, self.test_mcc )
                print(exp_result)
                return exp_result , self.test_acc

    def train_batch(self, parties_data, batch_label):
        '''
        batch_label: self.gt_one_hot_label   may be noisy
        '''
        encoder = self.args.encoder
        if self.args.apply_cae:
            assert encoder != None, "[error] encoder is None for CAE"
            _, gt_one_hot_label = encoder(batch_label)      
        else:
            gt_one_hot_label = batch_label

        # allocate data/attention mask to passive party
        for ik in range(self.k-1):
            self.parties[ik].obtain_local_data(parties_data[ik][0])
            self.parties[ik].gt_one_hot_label = self.gt_one_hot_label
            self.parties[ik].local_batch_attention_mask = parties_data[ik][2]

        # ====== normal vertical federated learning ======
        torch.autograd.set_detect_anomaly(True)
        # ======== Commu ===========
        if self.args.communication_protocol in ['Vanilla','Quantization','Topk'] or self.Q ==1 : # parallel FedBCD & noBCD situation
            for q in range(self.Q):
                if q == 0: 
                    # exchange info between party: local_prd/global_pred
                    self.pred_transmit() 
                    self.global_pred_transmit() 
                    self.global_loss_transmit()
                    
                    # update parameters for global trainable part
                    self.parties[self.k-1].global_backward() 
                    # for ik in range(self.k):
                    #     self.parties[ik].local_backward()
        else:
            assert 1>2 , 'Communication Protocol not provided'
        # ============= Commu ===================
        
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
            predict_prob = encoder.decode(predict_prob)

        suc_cnt = torch.sum(torch.argmax(predict_prob, dim=-1) == torch.argmax(real_batch_label, dim=-1)).item()
        train_acc = suc_cnt / predict_prob.shape[0]
        
        return loss.item(), train_acc

    def train(self):

        print_every = 1

        for ik in range(self.k):
            self.parties[ik].prepare_data_loader(batch_size=self.batch_size)

        test_acc = 0.0
        # Early Stop
        last_loss = 1000000
        early_stop_count = 0
        LR_passive_list = []
        LR_active_list = []

        self.num_total_comms = 0
        total_time = 0.0
        flag = 0
        self.current_epoch = 0


        start_time = time.time()
        for i_epoch in range(self.epochs):
            self.current_epoch = i_epoch
            postfix = {'train_loss': 0.0, 'train_acc': 0.0, 'test_acc': 0.0}
            i = -1

            data_loader_list = [self.parties[ik].train_loader for ik in range(self.k-1)]

            self.current_step = 0
            for parties_data in zip(*data_loader_list):
                self.parties_data = [ [_data[0].to(self.device),_data[1].to(self.device),_data[2].to(self.device)] for _data in parties_data]

                self.gt_one_hot_label = self.label_to_one_hot(parties_data[0][1], self.num_classes)
                # ###### Noisy Label Attack ######
                if self.args.apply_nl==True:
                    # noisy label
                    self.clean_one_hot_label = self.gt_one_hot_label
                    self.gt_one_hot_label = add_noise(self.args, self.gt_one_hot_label)
                # ###### Noisy Label Attack ######

                i += 1
                self.parties[self.k-1].global_model.train()
                
                # ====== train batch (start) ======
                enter_time = time.time()
                self.loss, self.train_acc = self.train_batch(self.parties_data,self.gt_one_hot_label)
                exit_time = time.time()
                total_time += (exit_time-enter_time)
                self.num_total_comms = self.num_total_comms + 1
                # if self.num_total_comms % 10 == 0:
                #     print(f"total time for {self.num_total_comms} communication is {total_time}")
                if self.train_acc > STOPPING_ACC[str(self.args.dataset)] and flag == 0:
                    self.stopping_time = total_time
                    self.stopping_iter = self.num_total_comms
                    self.stopping_commu_cost = self.communication_cost
                    flag = 1
                # ====== train batch (end) ======

                self.current_step = self.current_step + 1

                del(self.parties_data) # remove from cuda
                del(parties_data)

            # if self.args.apply_attack == True:
            #     if (self.args.attack_name in LABEL_INFERENCE_LIST) and i_epoch==1:
            #         print('Launch Label Inference Attack, Only train 1 epoch')
            #         break    

            # self.trained_models = self.save_state(True)
            # if self.args.save_model == True:
            #     self.save_trained_models()

            # LR decay
            self.LR_Decay(i_epoch)
            # LR record
            # if self.args.k == 2:
            #     LR_passive_list.append(self.parties[0].give_current_lr())
                # LR_active_list.append(self.parties[1].give_current_lr())

            # validation
            if (i + 1) % print_every == 0:
                print("validate and test")
                self.parties[self.k-1].global_model.eval()
                
                suc_cnt = 0
                sample_cnt = 0

                test_preds = []
                test_targets = []
                test_predict_labels = []
                test_actual_labels = []

                scores = []
                targets = []
                with torch.no_grad():
                    data_loader_list = [self.parties[ik].test_loader for ik in range(self.k-1)]
                    
                    for parties_data in zip(*data_loader_list):
                        parties_data = [ [_data[0].to(self.device),_data[1].to(self.device),_data[2].to(self.device)] for _data in parties_data]
                        gt_val_one_hot_label = self.label_to_one_hot(parties_data[0][1], self.num_classes)
                        
                        pred_list = []
                        for ik in range(self.k-1): # Passive data local predict
                            # _local_pred, input_shape = self.parties[ik].local_model(parties_data[ik][0])
                            _local_pred, input_shape = self.parties[ik].local_model(parties_data[ik][0],attention_mask=parties_data[ik][2])
                            self.parties[self.k-1].input_shape = input_shape
                            self.parties[self.k-1].local_batch_attention_mask = parties_data[0][2]
                            self.parties[ik].input_shape = input_shape
                            self.parties[ik].local_batch_attention_mask = parties_data[0][2]
                            pred_list.append(_local_pred)
                        
                        test_logit = self.parties[self.k-1].aggregate(pred_list,self.parties[self.k-1].local_batch_attention_mask, test="True")
                        # print('test_logit:',type(test_logit),test_logit.shape)
                        
                        enc_predict_prob = F.softmax(test_logit, dim=-1)

                        predict_label = torch.argmax(enc_predict_prob, dim=-1)
                        actual_label = torch.argmax(gt_val_one_hot_label, dim=-1)

                        test_preds.append(list(enc_predict_prob.detach().cpu().numpy()))
                        test_targets.append(list(gt_val_one_hot_label.detach().cpu().numpy()))

                        test_predict_labels.extend( list(predict_label.detach().cpu()) )
                        test_actual_labels.extend( list(actual_label.detach().cpu()) )
            
                        sample_cnt += predict_label.shape[0]
                        suc_cnt += torch.sum(predict_label == actual_label).item()
                        del(parties_data) # remove from cuda

                    self.test_acc = suc_cnt / float(sample_cnt)
                    
                    test_preds = np.vstack(test_preds)
                    test_targets = np.vstack(test_targets)
                    self.test_auc = np.mean(multiclass_auc(test_targets, test_preds))

                    self.test_mcc = matthews_corrcoef(np.array(test_predict_labels),np.array(test_actual_labels) ) # MCC

                    postfix['train_loss'] = self.loss
                    postfix['train_acc'] = '{:.2f}%'.format(self.train_acc * 100)
                    postfix['test_acc'] = '{:.2f}%'.format(self.test_acc * 100)
                    postfix['test_auc'] = '{:.2f}%'.format(self.test_auc * 100)
                    postfix['test_mcc'] = '{:.2f}%'.format(self.test_mcc * 100)
                    
                    exp_result = 'Epoch {}% \t train_loss:{:.2f} train_acc:{:.2f} test_acc:{:.2f} test_auc:{:.2f} test_mcc:{:.2f}'.format(
                        i_epoch, self.loss, self.train_acc, self.test_acc, self.test_auc, self.test_mcc)
                    print(exp_result)
                    self.final_epoch = i_epoch
        
        exp_result = 'train_loss:{:.2f} train_acc:{:.2f} test_acc:{:.2f} test_auc:{:.2f} test_mcc:{:.2f}'.format(
                        self.loss, self.train_acc, self.test_acc, self.test_auc, self.test_mcc)
                    
        save_path = self.args.exp_res_dir + '/pretrained_trainable_layer/' 
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.parties[self.k-1].global_model.trainable_layer.state_dict(),\
            save_path + f'/model={self.args.model_list[str(0)]["type"]}_lr{self.args.main_lr}_bs{self.args.batch_size}_acc{str(self.test_acc)}.pth')
        print(save_path + f'/model={self.args.model_list[str(0)]["type"]}_lr{self.args.main_lr}_bs{self.args.batch_size}_acc{str(self.test_acc)}.pth')
        
        # self.final_state = self.save_state(True) 
        # self.final_state.update(self.save_state(False)) 
        # self.final_state.update(self.save_party_data()) 


        return exp_result, self.test_acc #, self.stopping_iter, self.stopping_time, self.stopping_commu_cost



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
            "aux_attribute": [copy.deepcopy(self.parties[ik].aux_attribute) for ik in range(self.k)],
            "train_attribute": [copy.deepcopy(self.parties[ik].train_attribute) for ik in range(self.k)],
            "test_attribute": [copy.deepcopy(self.parties[ik].test_attribute) for ik in range(self.k)],
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
