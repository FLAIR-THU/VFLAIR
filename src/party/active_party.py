import json
import sys, os
sys.path.append(os.pardir)
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from party.party import Party
from party.llm_party import Party as Party_LLM
from utils.basic_functions import cross_entropy_for_onehot, tf_distance_cov_cor,pairwise_dist
from dataset.party_dataset import ActiveDataset
from load.LoadModels import load_models_per_party, load_models_per_party_new

# class ActiveParty_LLM(Party_LLM):
#     def __init__(self, args, index):
#         super().__init__(args, index)
#         self.criterion = cross_entropy_for_onehot
#         self.encoder = args.encoder
#         # print(f"in active party, encoder=None? {self.encoder==None}, {self.encoder}")
#         self.train_index = args.idx_train
#         self.test_index = args.idx_test
        
#         self.gt_one_hot_label = None

#         self.pred_received = []
#         for _ in range(args.k):
#             self.pred_received.append([])
        
#         self.global_pred = None
#         self.global_loss = None


#     def prepare_data(self, args, index):
#         super().prepare_data(args, index)
#         # self.train_dst = TensorDataset(train_inputs, train_masks, train_labels) # the second label is just a place holder
#         # self.test_dst = TensorDataset(test_inputs, test_masks, test_labels) # the second label is just a place holder
        
#         self.train_dst = ActiveDataset_LLM(args, self.train_data, self.train_label)
#         self.test_dst = ActiveDataset_LLM(args, self.test_data, self.test_label)

#         print('Active self.train_dst:',len(self.train_dst),  type(self.train_dst[0]), type(self.train_dst[1]) )


#         # self.train_dst = ActiveDataset(self.train_data, self.train_label)
#         # self.test_dst = ActiveDataset(self.test_data, self.test_label)

            
#     def update_local_pred(self, pred):
#         self.pred_received[self.args.k-1] = pred
    
#     def receive_pred(self, pred, giver_index):
#         self.pred_received[giver_index] = pred

#     def cal_loss(self, test=False):
#         gt_one_hot_label = self.gt_one_hot_label
#         pred =  self.global_pred
#         if self.train_index != None: # for graph data
#             if test == False:
#                 loss = self.criterion(pred[self.train_index], gt_one_hot_label[self.train_index])
#             else:
#                 loss = self.criterion(pred[self.test_index], gt_one_hot_label[self.test_index])
#         else:
#             loss = self.criterion(pred, gt_one_hot_label)
#         # ########## for active mid model loss (start) ##########
#         if self.args.apply_mid == True and (self.index in self.args.defense_configs['party']):
#             # print(f"in active party mid, label={gt_one_hot_label}, global_model.mid_loss_list={self.global_model.mid_loss_list}")
#             assert len(pred_list)-1 == len(self.global_model.mid_loss_list)
#             for mid_loss in self.global_model.mid_loss_list:
#                 loss = loss + mid_loss
#             self.global_model.mid_loss_list = [torch.empty((1,1)).to(self.args.device) for _ in range(len(self.global_model.mid_loss_list))]
#         # ########## for active mid model loss (end) ##########
#         elif self.args.apply_dcor==True and (self.index in self.args.defense_configs['party']):
#             # print('dcor active defense')
#             self.distance_correlation_lambda = self.args.defense_configs['lambda']
#             # loss = criterion(pred, gt_one_hot_label) + self.distance_correlation_lambda * torch.mean(torch.cdist(pred_a, gt_one_hot_label, p=2))
#             for ik in range(self.args.k-1):
#                 loss += self.distance_correlation_lambda * torch.log(tf_distance_cov_cor(pred_list[ik], gt_one_hot_label)) # passive party's loss
#         return loss

#     def gradient_calculation(self, pred_list, loss):
#         pred_gradients_list = []
#         pred_gradients_list_clone = []
#         for ik in range(self.args.k):
#             pred_gradients_list.append(torch.autograd.grad(loss, pred_list[ik], retain_graph=True, create_graph=True))
#             # print(f"in gradient_calculation, party#{ik}, loss={loss}, pred_gradeints={pred_gradients_list[-1]}")
#             pred_gradients_list_clone.append(pred_gradients_list[ik][0].detach().clone())
#         # self.global_backward(pred, loss)
#         return pred_gradients_list, pred_gradients_list_clone
    
#     def give_gradient(self):
#         pred_list = self.pred_received 

#         if self.gt_one_hot_label == None:
#             print('give gradient:self.gt_one_hot_label == None')
#             assert 1>2
#         self.global_loss  = self.cal_loss()
#         pred_gradients_list, pred_gradients_list_clone = self.gradient_calculation(pred_list, self.global_loss)
#         # self.local_gradient = pred_gradients_list_clone[self.args.k-1] # update local gradient

#         if self.args.defense_name == "GradPerturb":
#             self.calculate_gradient_each_class(self.global_pred, pred_list)
        
#         self.update_local_gradient(pred_gradients_list_clone[0])

#         return pred_gradients_list_clone
    
#     def update_local_gradient(self, gradient):
#         self.local_gradient = gradient

#     def global_LR_decay(self,i_epoch):
#         if self.global_model_optimizer != None: 
#             eta_0 = self.args.main_lr
#             eta_t = eta_0/(np.sqrt(i_epoch+1))
#             for param_group in self.global_model_optimizer.param_groups:
#                 param_group['lr'] = eta_t
        
                
#     def global_backward(self):

#         if self.global_model_optimizer != None: 
#             # active party with trainable global layer
#             _gradients = torch.autograd.grad(self.global_loss, self.global_pred, retain_graph=True)
#             _gradients_clone = _gradients[0].detach().clone()
            
#             # if self.args.apply_mid == False and self.args.apply_trainable_layer == False:
#             #     return # no need to update

#             # update global model
#             self.global_model_optimizer.zero_grad()
#             parameters = []          
#             if (self.args.apply_mid == True) and (self.index in self.args.defense_configs['party']): 
#                 # mid parameters
#                 for mid_model in self.global_model.mid_model_list:
#                     parameters += list(mid_model.parameters())
#                 # trainable layer parameters
#                 if self.args.apply_trainable_layer == True:
#                     parameters += list(self.global_model.global_model.parameters())
                
#                 # load grads into parameters
#                 weights_grad_a = torch.autograd.grad(self.global_pred, parameters, grad_outputs=_gradients_clone, retain_graph=True)
#                 for w, g in zip(parameters, weights_grad_a):
#                     if w.requires_grad:
#                         w.grad = g.detach()
                        
#             else:
#                 # trainable layer parameters
#                 if self.args.apply_trainable_layer == True:
#                     # load grads into parameters
#                     weights_grad_a = torch.autograd.grad(self.global_pred, self.global_model.parameters(), grad_outputs=_gradients_clone, retain_graph=True)
#                     for w, g in zip(self.global_model.parameters(), weights_grad_a):
#                         if w.requires_grad:
#                             w.grad = g.detach()
#                 # non-trainabel layer: no need to update
#             self.global_model_optimizer.step()

#     def calculate_gradient_each_class(self, global_pred, local_pred_list, test=False):
#         # print(f"global_pred.shape={global_pred.size()}") # (batch_size, num_classes)
#         self.gradient_each_class = [[] for _ in range(global_pred.size(1))]
#         one_hot_label = torch.zeros(global_pred.size()).to(global_pred.device)
#         for ic in range(global_pred.size(1)):
#             one_hot_label *= 0.0
#             one_hot_label[:,ic] += 1.0
#             if self.train_index != None: # for graph data
#                 if test == False:
#                     loss = self.criterion(global_pred[self.train_index], one_hot_label[self.train_index])
#                 else:
#                     loss = self.criterion(global_pred[self.test_index], one_hot_label[self.test_index])
#             else:
#                 loss = self.criterion(global_pred, one_hot_label)
#             for ik in range(self.args.k):
#                 self.gradient_each_class[ic].append(torch.autograd.grad(loss, local_pred_list[ik], retain_graph=True, create_graph=True))
#         # end of calculate_gradient_each_class, return nothing
    


class ActiveParty_LLM(Party_LLM):
    def __init__(self, args, index):
        print('==== ActiveParty_LLM ======')
        if args.device == 'cuda':
            cuda_id = args.gpu
            torch.cuda.set_device(cuda_id)
            print(f'running on cuda{torch.cuda.current_device()}')

        super().__init__(args, index)
        self.name = "server#" + str(index + 1)
        self.criterion = cross_entropy_for_onehot
        # self.encoder = args.encoder

        self.train_index = None  #args.idx_train
        self.test_index = None #args.idx_test
        
        self.gt_one_hot_label = None

        self.pred_received = []
        for _ in range(args.k):
            self.pred_received.append([])
        
        self.global_pred = None # transmitted to passive party
        self.global_loss = None # transmitted from passive party
        self.global_gradient = None # transmitted from passive party

        self.weights_grad_a = None


    # def prepare_data_loader(self, **kwargs):
    #     super().prepare_data_loader(self.args.batch_size, self.args.need_auxiliary)

    def eval(self, **kwargs):
        self.global_model.eval()

    def prepare_model(self, args, index):
        current_model_type = args.model_list['1']['type']
        pretrained = args.pretrained
        task_type = args.task_type
        model_type = args.model_type
        current_output_dim = args.model_list['1']['output_dim']
        is_local = False
        device = args.device
        padding_side = args.padding_side
        model_path = args.model_path
        main_lr = args.main_lr
        pad_token = args.pad_token
        head_layer_trainable = args.head_layer_trainable
        # prepare model and optimizer
        (
            self.local_model,
            self.local_model_optimizer,
            self.global_model,
            self.global_model_optimizer,
            args.tokenizer,
            self.encoder
        ) = load_models_per_party_new(pretrained, task_type, model_type, current_model_type, current_output_dim, is_local, device, padding_side, model_path, main_lr, pad_token, head_layer_trainable)

    def prepare_data(self, args, index):
        print('Active Party has no data, only global model')

    def receive_pred(self, pred, giver_index):
        self.pred_received[giver_index] = pred
    
    def receive_attention_mask(self, attention_mask):
        self.local_batch_attention_mask = attention_mask
    
    def receive_token_type_ids(self, token_type_ids):
        self.local_batch_token_type_ids = token_type_ids

    def mean(self, last_task_result):
        result = json.loads(last_task_result)
        self.mean_local(result)

    def mean_local(self, result):
        exact_score_list, f1_list = result
        if self.args.task_type == "QuestionAnswering":
            exact_score = np.mean(exact_score_list)
            f1 = np.mean(f1_list)
            exp_result = 'exact_score:{:.4f} f1:{:.4f}'.format(exact_score, f1)
            return exp_result, exact_score
        elif self.args.task_type == "SequenceClassification":
            pass
            # if self.num_classes == 1:
            #     self.test_mse = torch.mean(
            #         (torch.tensor(test_predict_labels) - torch.tensor(test_actual_labels)) ** 2).item()
            #
            #     self.test_pearson_corr = \
            #     stats.pearsonr(torch.tensor(test_predict_labels), torch.tensor(test_actual_labels))[0]
            #     self.test_spearmanr_corr = \
            #     stats.spearmanr(torch.tensor(test_predict_labels), torch.tensor(test_actual_labels))[0]
            #
            #     exp_result = 'test_mse:{:.4f} test_pearson_corr:{:.4f} test_spearmanr_corr:{:.4f}'.format(self.test_mse,
            #                                                                                               self.test_pearson_corr,
            #                                                                                               self.test_spearmanr_corr)
            #     return exp_result, self.test_mse
            # else:
            #     self.test_acc = suc_cnt / float(sample_cnt)  # ACC
            #
            #     self.test_mcc = matthews_corrcoef(np.array(test_predict_labels), np.array(test_actual_labels))  # MCC
            #
            #     exp_result = 'test_acc:{:.2f} test_mcc:{:.2f}'.format(self.test_acc, self.test_mcc)
            #     return exp_result, self.test_acc

    def aggregate_remote(self, pred_list):
        self.global_model.eval()
        with torch.no_grad():
            t1 = torch.Tensor(pred_list[0])
            t2 = torch.Tensor(pred_list[1])
            t1 = t1.to(self.args.device)
            t2 = t2.to(self.args.device)
            return self.aggregate([[t1, t2]])

    def aggregate(self, pred_list, test=False):
        self.passive_pred_list = pred_list
        self.passive_pred_list[0][0].requires_grad = True

        if self.args.model_type == 'Bert': # passive_pred_list[0] = [intermediate, attention_mask]
            if self.args.task_type == 'SequenceClassification':# pred_list[0] = [intermediate, ,sequence_lengths, attention_mask]
                pred = self.global_model(self.passive_pred_list[0][0], attention_mask = self.passive_pred_list[0][1],return_dict=True).logits
            elif self.args.task_type == 'QuestionAnswering':# self.passive_pred_list[0] = [intermediate, attention_mask]
                pred = self.global_model(self.passive_pred_list[0][0], attention_mask = self.passive_pred_list[0][1], return_dict=True)
        
        elif self.args.model_type == 'GPT2': # self.passive_pred_list[0] = [intermediate, sequence_lengths, attention_mask]
            if self.args.task_type == 'CausalLM':# self.passive_pred_list[0] = [intermediate, attention_mask]
                pred = self.global_model(self.passive_pred_list[0][0],  attention_mask=self.passive_pred_list[0][1], return_dict=True).logits
            elif self.args.task_type == 'SequenceClassification':# self.passive_pred_list[0] = [intermediate, ,sequence_lengths, attention_mask]
                pred = self.global_model(self.passive_pred_list[0][0],  self.passive_pred_list[0][1], attention_mask=self.passive_pred_list[0][2], return_dict=True).logits
            elif self.args.task_type == 'QuestionAnswering':# self.passive_pred_list[0] = [intermediate, attention_mask]
                pred = self.global_model(self.passive_pred_list[0][0],  attention_mask=self.passive_pred_list[0][1], return_dict=True)
            else:
                assert 1>2 , 'Task type no supported'

        elif self.args.model_type == 'Llama': 
            if self.args.task_type == 'CausalLM':# self.passive_pred_list[0] = [intermediate, attention_mask]
                pred = self.global_model(self.passive_pred_list[0][0],  attention_mask=self.passive_pred_list[0][1], return_dict=True).logits
            elif self.args.task_type == 'SequenceClassification':# self.passive_pred_list[0] = [intermediate, ,sequence_lengths, attention_mask]
                pred = self.global_model(self.passive_pred_list[0][0],  self.passive_pred_list[0][1], attention_mask=self.passive_pred_list[0][2], return_dict=True).logits
            elif self.args.task_type == 'QuestionAnswering':# self.passive_pred_list[0] = [intermediate, attention_mask]
                pred = self.global_model(self.passive_pred_list[0][0],  attention_mask=self.passive_pred_list[0][1], return_dict=True)
            else:
                assert 1>2 , 'Task type no supported'

        self.global_pred = pred

        # passive_local_gradient = torch.autograd.grad(self.global_pred, self.passive_pred_list[0][0] , \
        #     retain_graph=True).detach().clone()
        return pred

    def receive_loss_and_gradients(self, loss, gradients):
        self.global_loss = loss
        self.global_gradients = gradients

    def generate(self, pred_list, test=False):
        # if self.args.model_type == 'Bert': # pred_list[0] = [intermediate, attention_mask]
        #     pred = self.global_model(pred_list[0][0], attention_mask = pred_list[0][1])
        
        if self.args.model_type == 'GPT2': # pred_list[0] = [intermediate, sequence_lengths, attention_mask]
            if self.args.task_type == 'CausalLM':# pred_list[0] = [intermediate, attention_mask]
                generated = self.global_model.transformer.greedy_search(intermediate = pred_list[0][0],  attention_mask=pred_list[0][1])
                print('generated:',generated)
                generate_text = self.args.tokenizer.decode(generated.tolist())
                print('generate_text:',generate_text)

        # elif self.args.model_type == 'Llama': 
        #     if self.args.task_type == 'CausalLM':# pred_list[0] = [intermediate, attention_mask]
        #         pred = self.global_model(pred_list[0][0],  attention_mask=pred_list[0][1])
        #     elif self.args.task_type == 'SequenceClassification':# pred_list[0] = [intermediate, ,sequence_lengths, attention_mask]
        #         pred = self.global_model(pred_list[0][0],  pred_list[0][1], attention_mask=pred_list[0][2])
        #     else:
        #         assert 1>2 , 'Task type no supported'

        return generated, generate_text

    def global_LR_decay(self,i_epoch):
        if self.global_model_optimizer != None: 
            eta_0 = self.args.main_lr
            eta_t = eta_0/(np.sqrt(i_epoch+1))
            for param_group in self.global_model_optimizer.param_groups:
                param_group['lr'] = eta_t

    def cal_passive_local_gradient(self, passive_party_id):
        passive_local_gradient = torch.autograd.grad(self.global_pred, self.passive_pred_list[passive_party_id][0] , \
            grad_outputs=self.global_gradient, retain_graph=True).detach().clone()
        return passive_local_gradient

    def global_backward(self):

        if self.global_model_optimizer != None: 
            if self.args.task_type == 'QuestionAnswering':
                # server with trainable global layer
                # print('self.global_loss:',type(self.global_loss), self.global_loss)
                # print('self.global_pred:',type(self.global_pred), self.global_pred) # .start_logits  end_logits
                
                _gradients_start = torch.autograd.grad(self.global_loss, self.global_pred.start_logits, retain_graph=True)
                _gradients_end = torch.autograd.grad(self.global_loss, self.global_pred.end_logits, retain_graph=True)
                _gradients = _gradients_end+_gradients_start
                _gradients_clone = _gradients[0].detach().clone()
                _gradients_clone = _gradients_clone/2
                # print('global_gradients_clone:',_gradients_clone)
                
                # update global model
                self.global_model_optimizer.zero_grad()
                parameters = []          
            
                # trainable layer parameters
                # load grads into parameters
                weights_grad_a_start = torch.autograd.grad(self.global_pred.start_logits, self.global_model.trainable_layer.parameters(), grad_outputs=_gradients_clone, retain_graph=True)
                weights_grad_a_end = torch.autograd.grad(self.global_pred.end_logits, self.global_model.trainable_layer.parameters(), grad_outputs=_gradients_clone, retain_graph=True)
                
                # print('weights_grad_a_start:',len(weights_grad_a_start),type(weights_grad_a_start)) # 2 tuple
                # print('weights_grad_a_end:',len(weights_grad_a_end),type(weights_grad_a_end))
                
                self.weights_grad_a = []
                for _i in range( len(weights_grad_a_start) ):
                    self.weights_grad_a.append(weights_grad_a_start[_i]+weights_grad_a_end[_i])
                self.weights_grad_a = tuple( self.weights_grad_a )
                # print('weights_grad_a:',len(self.weights_grad_a),type(self.weights_grad_a))
                    
                # self.weights_grad_a = weights_grad_a_start+weights_grad_a_end
                # print('weights_grad_a:',len(self.weights_grad_a),type(self.weights_grad_a))

                # self.weights_grad_a = self.weights_grad_a/2
                for w, g in zip(self.global_model.trainable_layer.parameters(), self.weights_grad_a):
                    if w.requires_grad:
                        w.grad = g.detach()
                # print('weights_grad_a:',weights_grad_a)
               
                self.global_model_optimizer.step()
            else:
                # # server with trainable global layer
                # _gradients = torch.autograd.grad(self.global_loss, self.global_pred, retain_graph=True)
                # _gradients_clone = _gradients[0].detach().clone()

                _gradients_clone = self.global_gradient
                
                # update global model
                self.global_model_optimizer.zero_grad()
                parameters = []          
            
                # trainable layer parameters
                weights_grad_a = torch.autograd.grad(self.global_pred, self.global_model.trainable_layer.parameters(), grad_outputs=_gradients_clone, retain_graph=True)
                self.weights_grad_a = weights_grad_a
                for w, g in zip(self.global_model.trainable_layer.parameters(), weights_grad_a):
                    if w.requires_grad:
                        w.grad = g.detach()
                # print('weights_grad_a:',weights_grad_a)
               
                self.global_model_optimizer.step()
          


class ActiveParty(Party):
    def __init__(self, args, index):
        super().__init__(args, index)
        self.criterion = cross_entropy_for_onehot
        self.encoder = args.encoder
        # print(f"in active party, encoder=None? {self.encoder==None}, {self.encoder}")
        self.train_index = args.idx_train
        self.test_index = args.idx_test
        
        self.gt_one_hot_label = None

        self.pred_received = []
        for _ in range(args.k):
            self.pred_received.append([])
        
        self.global_pred = None
        self.global_loss = None

    def prepare_data(self, args, index):
        super().prepare_data(args, index)
        self.train_dst = ActiveDataset(self.train_data, self.train_label)
        self.test_dst = ActiveDataset(self.test_data, self.test_label)
        if self.args.need_auxiliary == 1:
            self.aux_dst = ActiveDataset(self.aux_data, self.aux_label)
            # self.aux_loader = DataLoader(self.aux_dst, batch_size=batch_size,shuffle=True)

    def update_local_pred(self, pred):
        self.pred_received[self.args.k-1] = pred
    
    def receive_pred(self, pred, giver_index):
        self.pred_received[giver_index] = pred

    def aggregate(self, pred_list, gt_one_hot_label, test=False):
        if self.args.dataset == 'cora' and self.args.apply_trainable_layer == 1:
            pred = self.global_model(pred_list, self.local_batch_data)
        else:
            pred = self.global_model(pred_list)
        if self.train_index != None: # for graph data
            if test == False:
                loss = self.criterion(pred[self.train_index], gt_one_hot_label[self.train_index])
            else:
                loss = self.criterion(pred[self.test_index], gt_one_hot_label[self.test_index])
        else:
            loss = self.criterion(pred, gt_one_hot_label)
        # ########## for active mid model loss (start) ##########
        if self.args.apply_mid == True and (self.index in self.args.defense_configs['party']):
            # print(f"in active party mid, label={gt_one_hot_label}, global_model.mid_loss_list={self.global_model.mid_loss_list}")
            assert len(pred_list)-1 == len(self.global_model.mid_loss_list)
            for mid_loss in self.global_model.mid_loss_list:
                loss = loss + mid_loss
            self.global_model.mid_loss_list = [torch.empty((1,1)).to(self.args.device) for _ in range(len(self.global_model.mid_loss_list))]
        # ########## for active mid model loss (end) ##########
        elif self.args.apply_dcor==True and (self.index in self.args.defense_configs['party']):
            # print('dcor active defense')
            self.distance_correlation_lambda = self.args.defense_configs['lambda']
            # loss = criterion(pred, gt_one_hot_label) + self.distance_correlation_lambda * torch.mean(torch.cdist(pred_a, gt_one_hot_label, p=2))
            for ik in range(self.args.k-1):
                loss += self.distance_correlation_lambda * torch.log(tf_distance_cov_cor(pred_list[ik], gt_one_hot_label)) # passive party's loss
        return pred, loss

    def gradient_calculation(self, pred_list, loss):
        pred_gradients_list = []
        pred_gradients_list_clone = []
        for ik in range(self.args.k):
            pred_gradients_list.append(torch.autograd.grad(loss, pred_list[ik], retain_graph=True, create_graph=True))
            # print(f"in gradient_calculation, party#{ik}, loss={loss}, pred_gradeints={pred_gradients_list[-1]}")
            pred_gradients_list_clone.append(pred_gradients_list[ik][0].detach().clone())
        # self.global_backward(pred, loss)
        return pred_gradients_list, pred_gradients_list_clone
    
    def give_gradient(self):
        pred_list = self.pred_received 

        if self.gt_one_hot_label == None:
            print('give gradient:self.gt_one_hot_label == None')
            assert 1>2

        self.global_pred, self.global_loss = self.aggregate(pred_list, self.gt_one_hot_label)
        pred_gradients_list, pred_gradients_list_clone = self.gradient_calculation(pred_list, self.global_loss)
        # self.local_gradient = pred_gradients_list_clone[self.args.k-1] # update local gradient

        if self.args.defense_name == "GradPerturb":
            self.calculate_gradient_each_class(self.global_pred, pred_list)
        
        return pred_gradients_list_clone
    
    def update_local_gradient(self, gradient):
        self.local_gradient = gradient

    def global_LR_decay(self,i_epoch):
        if self.global_model_optimizer != None: 
            eta_0 = self.args.main_lr
            eta_t = eta_0/(np.sqrt(i_epoch+1))
            for param_group in self.global_model_optimizer.param_groups:
                param_group['lr'] = eta_t
        
                
    def global_backward(self):

        if self.global_model_optimizer != None: 
            # active party with trainable global layer
            _gradients = torch.autograd.grad(self.global_loss, self.global_pred, retain_graph=True)
            _gradients_clone = _gradients[0].detach().clone()
            
            # if self.args.apply_mid == False and self.args.apply_trainable_layer == False:
            #     return # no need to update

            # update global model
            self.global_model_optimizer.zero_grad()
            parameters = []          
            if (self.args.apply_mid == True) and (self.index in self.args.defense_configs['party']): 
                # mid parameters
                for mid_model in self.global_model.mid_model_list:
                    parameters += list(mid_model.parameters())
                # trainable layer parameters
                if self.args.apply_trainable_layer == True:
                    parameters += list(self.global_model.global_model.parameters())
                
                # load grads into parameters
                weights_grad_a = torch.autograd.grad(self.global_pred, parameters, grad_outputs=_gradients_clone, retain_graph=True)
                for w, g in zip(parameters, weights_grad_a):
                    if w.requires_grad:
                        w.grad = g.detach()
                        
            else:
                # trainable layer parameters
                if self.args.apply_trainable_layer == True:
                    # load grads into parameters
                    weights_grad_a = torch.autograd.grad(self.global_pred, self.global_model.parameters(), grad_outputs=_gradients_clone, retain_graph=True)
                    for w, g in zip(self.global_model.parameters(), weights_grad_a):
                        if w.requires_grad:
                            w.grad = g.detach()
                # non-trainabel layer: no need to update
            self.global_model_optimizer.step()

    def calculate_gradient_each_class(self, global_pred, local_pred_list, test=False):
        # print(f"global_pred.shape={global_pred.size()}") # (batch_size, num_classes)
        self.gradient_each_class = [[] for _ in range(global_pred.size(1))]
        one_hot_label = torch.zeros(global_pred.size()).to(global_pred.device)
        for ic in range(global_pred.size(1)):
            one_hot_label *= 0.0
            one_hot_label[:,ic] += 1.0
            if self.train_index != None: # for graph data
                if test == False:
                    loss = self.criterion(global_pred[self.train_index], one_hot_label[self.train_index])
                else:
                    loss = self.criterion(global_pred[self.test_index], one_hot_label[self.test_index])
            else:
                loss = self.criterion(global_pred, one_hot_label)
            for ik in range(self.args.k):
                self.gradient_each_class[ic].append(torch.autograd.grad(loss, local_pred_list[ik], retain_graph=True, create_graph=True))
        # end of calculate_gradient_each_class, return nothing
    