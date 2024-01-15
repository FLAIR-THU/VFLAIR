import sys, os
sys.path.append(os.pardir)
import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from utils.basic_functions import cross_entropy_for_onehot, tf_distance_cov_cor,pairwise_dist
from party.party import Party
from party.llm_party import Party as Party_LLM
from dataset.party_dataset import PassiveDataset, PassiveDataset_LLM
from dataset.party_dataset import ActiveDataset
from load.LoadModels import load_models_per_party

class PassiveParty(Party):
    def __init__(self, args, index):
        super().__init__(args, index)

    def prepare_data(self, args, index):
        super().prepare_data(args, index)
        # self.train_dst = TensorDataset(train_inputs, train_masks) # the second label is just a place holder
        # self.test_dst = TensorDataset(test_inputs, test_masks) # the second label is just a place holder
        
        self.train_dst = PassiveDataset(self.train_data)
        self.test_dst = PassiveDataset(self.test_data)
        if self.args.need_auxiliary == 1:
            self.aux_dst = ActiveDataset(self.aux_data, self.aux_label)


class PassiveParty_LLM(Party_LLM):
    def __init__(self, args, index):
        super().__init__(args, index)
        self.criterion = cross_entropy_for_onehot
        self.encoder = args.encoder
        self.train_index = args.idx_train
        self.test_index = args.idx_test
        
        self.gt_one_hot_label = None

        self.pred_received = []
        for _ in range(args.k):
            self.pred_received.append([])
        
        self.global_pred = None
        self.global_loss = None

        self.num_labels = args.num_classes
        self.weights_grad_a = None # no gradient for model in passive party(no model update)


    def prepare_data(self, args, index):
        super().prepare_data(args, index) # Party_llm's prepare_data
 
        self.train_dst = PassiveDataset_LLM(args, self.train_data, self.train_label)
        self.test_dst = PassiveDataset_LLM(args, self.test_data, self.test_label)

        # print('self.train_dst:',len(self.train_dst),  type(self.train_dst[0]), type(self.train_dst[1]) )

            
    def update_local_pred(self, pred):
        self.pred_received[self.args.k-1] = pred
    
    def receive_pred(self, pred, giver_index):
        self.pred_received[giver_index] = pred

    def cal_loss(self, test=False):
        gt_one_hot_label = self.gt_one_hot_label # label
        pred =  self.global_pred # logits

        # print('== In cal_loss ==')
        # print('gt_one_hot_label:',type(gt_one_hot_label),gt_one_hot_label )
        # print('pred:',type(pred),pred.shape )

        if self.args.task_type == 'SequenceClassification':
            # loss = self.criterion(pred, gt_one_hot_label)
            pooled_logits = pred
            labels = gt_one_hot_label
            # GPT2
            if self.num_labels == 1:
                self.problem_type = "regression"
            elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                self.problem_type = "single_label_classification"
            else:
                self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        elif self.args.task_type == 'CausalLM':
            #  GPT2
            labels = gt_one_hot_label
            # print('labels:',type(labels),labels)  # list of target_tokens
            label_id = [ self.args.tokenizer.convert_tokens_to_ids( label_text ) for label_text in labels ]
            label_id = torch.tensor(label_id).to(self.args.device)
            # print('label_id:', label_id.shape ) # torch.size([bs])
            
            lm_logits = pred # # [bs, seq_len, vocab_size]
            next_token_logits = lm_logits[:,-1,:]
            # print('next_token_logits:',next_token_logits.shape) # [bs, vocab_size]
            
            # Shift so that tokens < n predict n
            # shift_logits = lm_logits[..., :-1, :].contiguous()
            # print('shift_logits:',shift_logits.shape)
            # shift_labels = label_id #labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss_fct(next_token_logits, label_id)
            # print('loss:', loss)

        elif self.args.task_type == 'QuestionAnswering':
            # GPT2
            # print('gt_one_hot_label:',gt_one_hot_label)
            start_logits = pred.start_logits
            end_logits = pred.end_logits
            golden_start_positions, golden_end_positions = gt_one_hot_label[0] # bs *[start_id, end_id]  bs=1
            golden_start_positions = golden_start_positions.unsqueeze(0).long()
            golden_end_positions = golden_end_positions.unsqueeze(0).long()

            # print('logits:',start_logits.shape, end_logits.shape)
            # print('golden:',golden_start_positions, golden_end_positions)

            loss = None

            if len(golden_start_positions.size()) > 1:
                golden_start_positions = golden_start_positions.squeeze(-1).to(start_logits.device)
            if len(golden_end_positions.size()) > 1:
                golden_end_positions = golden_end_positions.squeeze(-1).to(end_logits.device)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            # print('ignored_index:',ignored_index)
            golden_start_positions = golden_start_positions.clamp(0, ignored_index)
            golden_end_positions = golden_end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

            # print('loss logits:',start_logits.shape, end_logits.shape)
            # print('loss golden:',golden_start_positions, golden_end_positions)

            start_loss = loss_fct(start_logits, golden_start_positions)
            end_loss = loss_fct(end_logits, golden_end_positions)
            loss = (start_loss + end_loss) / 2

        else:
            assert 1>2 , 'Task type not supported'
        

        # ########### Defense on Loss ###############
        # active mid model loss
        if self.args.apply_mid == True and (self.index in self.args.defense_configs['party']):
            # print(f"in active party mid, label={gt_one_hot_label}, global_model.mid_loss_list={self.global_model.mid_loss_list}")
            assert len(pred_list)-1 == len(self.global_model.mid_loss_list)
            for mid_loss in self.global_model.mid_loss_list:
                loss = loss + mid_loss
            self.global_model.mid_loss_list = [torch.empty((1,1)).to(self.args.device) for _ in range(len(self.global_model.mid_loss_list))]
        # active dcor loss
        elif self.args.apply_dcor==True and (self.index in self.args.defense_configs['party']):
            # print('dcor active defense')
            self.distance_correlation_lambda = self.args.defense_configs['lambda']
            # loss = criterion(pred, gt_one_hot_label) + self.distance_correlation_lambda * torch.mean(torch.cdist(pred_a, gt_one_hot_label, p=2))
            for ik in range(self.args.k-1):
                loss += self.distance_correlation_lambda * torch.log(tf_distance_cov_cor(pred_list[ik], gt_one_hot_label)) # passive party's loss
        # ########### Defense on Loss ###############
        # print('cal loss:',loss)
        return loss

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
        self.global_loss  = self.cal_loss()
        pred_gradients_list, pred_gradients_list_clone = self.gradient_calculation(pred_list, self.global_loss)
        # self.local_gradient = pred_gradients_list_clone[self.args.k-1] # update local gradient

        if self.args.defense_name == "GradPerturb":
            self.calculate_gradient_each_class(self.global_pred, pred_list)
        
        self.update_local_gradient(pred_gradients_list_clone[0])

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
    

# class PassiveParty_LLM(Party_LLM):
#     def __init__(self, args, index):
#         super().__init__(args, index)

#     def prepare_data(self, args, index):
#         super().prepare_data(args, index)
#         self.train_dst = PassiveDataset_LLM(args, self.train_data)

#         print('Passive self.train_dst:',len(self.train_dst), type(self.train_dst[0]), type(self.train_dst[1]) )

#         self.test_dst = PassiveDataset_LLM(args,self.test_data)

#         # self.train_dst = PassiveDataset(self.train_data)
#         # self.test_dst = PassiveDataset(self.test_data)

#         # if self.args.need_auxiliary == 1:
#         #     self.aux_dst = ActiveDataset(self.aux_data, self.aux_label)
#             # self.aux_loader = DataLoader(self.aux_dst, batch_size=batch_size,shuffle=True)