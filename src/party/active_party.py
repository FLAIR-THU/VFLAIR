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
    def __init__(self, args, index, need_data = True):
        print(f'==== initialize ActiveParty_LLM : party {index}======')
        if args.device == 'cuda':
            cuda_id = args.gpu
            torch.cuda.set_device(cuda_id)
            print(f'running on cuda{torch.cuda.current_device()}')

        super().__init__(args, index, need_data = need_data)
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
        self.global_gradients = None # transmitted from passive party

        self.weights_grad_a = None

        self.encoder_hidden_states = None
        self.encoder_attention_mask = None


    # def prepare_data_loader(self, **kwargs):
    #     super().prepare_data_loader(self.args.batch_size, self.args.need_auxiliary)

    def eval(self, **kwargs):
        self.global_model.eval()

    def prepare_data(self, args, index):
        print('Active Party has no data, only global model')

    def receive_pred(self, pred, giver_index):
        self.pred_received[giver_index] = pred
    
    def receive_attention_mask(self, attention_mask):
        self.local_batch_attention_mask = attention_mask
    
    def receive_token_type_ids(self, token_type_ids):
        self.local_batch_token_type_ids = token_type_ids

    def train_model(self):
        self.global_model.train()

    def _do_aggregate_remote(self, pred_list):
        t1 = torch.Tensor(pred_list[0])
        t2 = torch.Tensor(pred_list[1])
        t1 = t1.to(self.args.device)
        t2 = t2.to(self.args.device)
        result = self.aggregate([[t1, t2]])

        if self.args.model_type in ['Bert','Roberta']:
            if self.args.task_type == 'SequenceClassification':
                return {
                    "requires_grad": result.requires_grad,
                    "grad_fn": result.grad_fn.name(),
                    "logits": result.tolist()
                }
            elif self.args.task_type == 'QuestionAnswering':
                return {
                    # "loss": result.total_loss.float(),
                    "start_logits": result.start_logits.tolist(),
                    "end_logits": result.end_logits.tolist(),
                    # "hidden_states": result.outputs.hidden_states,
                    # "attentions": result.outputs.attentions,
                }

    def aggregate_remote(self, pred_list):
        if self.args.head_layer_trainable[1]:
            return self._do_aggregate_remote(pred_list)
        with torch.no_grad():
            return self._do_aggregate_remote(pred_list)

    def aggregate(self, pred_list, use_cache = False, test=False):
        # print(' == Active Aggregate == ')

        self.passive_pred_list = pred_list
        # print('self.passive_pred_list[0][0]:',self.passive_pred_list[0][0])

        self.passive_pred_list[0][0].requires_grad = True

        if self.args.model_type in ['Bert','Roberta']: # passive_pred_list[0] = [intermediate, attention_mask]
            if self.args.task_type == 'SequenceClassification':# pred_list[0] = [intermediate, ,sequence_lengths, attention_mask]
                self.global_output = self.global_model(self.passive_pred_list[0][0], attention_mask = self.passive_pred_list[0][1],return_dict=True)
                pred = self.global_output.logits
            elif self.args.task_type == 'QuestionAnswering':# self.passive_pred_list[0] = [intermediate, attention_mask]
                self.global_output = self.global_model(self.passive_pred_list[0][0], attention_mask = self.passive_pred_list[0][1], return_dict=True)
                pred = self.global_output
        
        elif self.args.model_type == 'GPT2': # self.passive_pred_list[0] = [intermediate, sequence_lengths, attention_mask]
            if self.args.task_type == 'CausalLM':# self.passive_pred_list[0] = [intermediate, attention_mask]
                self.global_output = self.global_model(self.passive_pred_list[0][0], \
                 attention_mask=self.passive_pred_list[0][1],\
                 use_cache = use_cache,\
                 return_dict=True)
                pred = self.global_output.logits
            elif self.args.task_type == 'Generation':# self.passive_pred_list[0] = [intermediate, attention_mask]
                self.global_output = self.global_model(self.passive_pred_list[0][0], \
                 attention_mask=self.passive_pred_list[0][1],\
                 local_past_key_values = self.passive_pred_list[0][3],\
                 past_key_values = self.past_key_values,\
                 use_cache = use_cache,\
                 return_dict=True)
                pred = self.global_output
            elif self.args.task_type == 'SequenceClassification':# self.passive_pred_list[0] = [intermediate,sequence_lengths, attention_mask]
                self.global_output = self.global_model(self.passive_pred_list[0][0],
                 attention_mask=self.passive_pred_list[0][1], \
                 sequence_lengths = self.passive_pred_list[0][2], \
                 use_cache = use_cache,\
                 return_dict=True)
                pred = self.global_output.logits
            elif self.args.task_type == 'QuestionAnswering':# self.passive_pred_list[0] = [intermediate, attention_mask]
                self.global_output = self.global_model(self.passive_pred_list[0][0],\
                 attention_mask=self.passive_pred_list[0][1],\
                 use_cache = use_cache,\
                 return_dict=True)
                pred = self.global_output
            else:
                assert 1>2 , 'Task type no supported'

        elif self.args.model_type == 'Llama': 
            # [0 intermediate, 1 attention_mask, 2 sequence_lengths, 3 past_key_values]
            if self.args.task_type == 'CausalLM':# self.passive_pred_list[0] = [intermediate, attention_mask]
                self.global_output = self.global_model(self.passive_pred_list[0][0],  attention_mask=self.passive_pred_list[0][1], return_dict=True)
                pred = self.global_output.logits
            elif self.args.task_type == 'Generation':# self.passive_pred_list[0] = [intermediate, attention_mask]
                self.global_output = self.global_model(self.passive_pred_list[0][0], attention_mask=self.passive_pred_list[0][1],\
                 local_past_key_values = self.passive_pred_list[0][2],\
                 past_key_values = self.past_key_values,\
                 use_cache = use_cache,\
                 return_dict=True)
                pred = self.global_output
            elif self.args.task_type == 'SequenceClassification':# self.passive_pred_list[0] = [intermediate, ,sequence_lengths, attention_mask]
                self.global_output = self.global_model(self.passive_pred_list[0][0], \
                    attention_mask=self.passive_pred_list[0][1], \
                    sequence_lengths = self.passive_pred_list[0][2], \
                    past_key_values = self.passive_pred_list[0][3],\
                    return_dict=True)
                pred = self.global_output.logits
            elif self.args.task_type == 'QuestionAnswering':# self.passive_pred_list[0] = [intermediate, attention_mask]
                self.global_output = self.global_model(self.passive_pred_list[0][0],  attention_mask=self.passive_pred_list[0][1], return_dict=True)
                pred = self.global_output
            else:
                assert 1>2 , 'Task type no supported'

        self.global_pred = pred
        # print('self.global_pred:',self.global_pred)

        return pred

    def receive_loss_and_gradients(self, gradients): # self , loss, gradients
        # self.global_loss = loss
        self.global_gradients = gradients
        # print('Active Party receive self.global_gradients:')
        # print(self.global_gradients)


    def global_LR_decay(self,i_epoch):
        if self.global_model_optimizer != None: 
            eta_0 = self.args.main_lr
            eta_t = eta_0/(np.sqrt(i_epoch+1))
            for param_group in self.global_model_optimizer.param_groups:
                param_group['lr'] = eta_t

    def cal_passive_local_gradient(self, ik):
        # print('self.global_pred:',type(self.global_pred))
        # print(self.global_pred.requires_grad)

        # print('self.passive_pred_list[ik][0]:',type(self.passive_pred_list[ik][0]))

        # print('self.global_gradients:',type(self.global_gradients))
        # print(self.global_gradients.requires_grad)

        if self.args.task_type == 'QuestionAnswering':
            passive_local_gradient = torch.autograd.grad(self.global_pred.start_logits+self.global_pred.end_logits, self.passive_pred_list[ik][0], \
            grad_outputs=self.global_gradients, retain_graph=True)[0].detach().clone()
        else:
            passive_local_gradient = torch.autograd.grad(self.global_pred, self.passive_pred_list[ik][0], \
            grad_outputs=self.global_gradients, retain_graph=True)[0].detach().clone()
            # print(f'Active Party cal passive pradient {ik}')
            # print(passive_local_gradient)

        return passive_local_gradient

    def global_backward(self):
        # print('=== Active Global Backward ===')

        if self.global_model_optimizer != None: 
            if self.args.task_type == 'QuestionAnswering':
                
                # update global model
                self.global_model_optimizer.zero_grad()
                parameters = []          
            
                # trainable layer parameters
                # load grads into parameters
                weights_grad_a_start = torch.autograd.grad(self.global_pred.start_logits, self.global_model.head_layer.parameters(), \
                    grad_outputs=self.global_gradients, retain_graph=True)
                weights_grad_a_end = torch.autograd.grad(self.global_pred.end_logits, self.global_model.head_layer.parameters(),\
                     grad_outputs=self.global_gradients, retain_graph=True)
                
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
                for w, g in zip(self.global_model.head_layer.parameters(), self.weights_grad_a):
                    if w.requires_grad:
                        w.grad = g.detach()
                # print('weights_grad_a:',weights_grad_a)
               
                self.global_model_optimizer.step()
            else:                
                # update global model
                self.global_model_optimizer.zero_grad()  
                weights_grad_a = torch.autograd.grad(self.global_pred, self.global_model.head_layer.parameters(), grad_outputs=self.global_gradients, retain_graph=True)
                self.weights_grad_a = weights_grad_a
                for w, g in zip(self.global_model.head_layer.parameters(), weights_grad_a):
                    if w.requires_grad:
                        w.grad = g.detach()
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
    