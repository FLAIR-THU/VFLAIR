import json
import sys, os
sys.path.append(os.pardir)
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from framework.ml.llm_party import Party as Party_LLM
from utils.basic_functions import cross_entropy_for_onehot, tf_distance_cov_cor,pairwise_dist
from framework.ml.LoadModels import load_models_per_party
import collections
from utils.squad_utils import normalize_answer
from sklearn.metrics import roc_auc_score,matthews_corrcoef

class ActiveParty_LLM(Party_LLM):
    def __init__(self, args):
        super().__init__()
        self.args = args
        print('==== ActiveParty_LLM ======')
        if args.device == 'cuda':
            cuda_id = args.gpu
            torch.cuda.set_device(cuda_id)
            print(f'running on cuda{torch.cuda.current_device()}')

        self.prepare_model(args)
        self.name = "server#active"
        self.criterion = cross_entropy_for_onehot
        # self.encoder = args.encoder

        self.train_index = None  #args.idx_train
        self.test_index = None #args.idx_test
        
        self.gt_one_hot_label = None

        self.pred_received = []
        for _ in range(2):
            self.pred_received.append([])
        
        self.global_pred = None
        self.global_loss = None
        self.need_auxiliary = 0

    def prepare_data_loader(self, **kwargs):
        super().prepare_data_loader(self.args.batch_size, self.need_auxiliary)

    def eval(self, **kwargs):
        self.global_model.eval()

    def prepare_model(self, args):
        current_model_type = args.model_list['1']['type']
        pretrained = args.pretrained
        task_type = args.task_type
        model_type = args.model_type
        current_output_dim = args.model_list['1']['output_dim']
        is_local = False
        device = args.device
        # prepare model and optimizer
        (
            self.local_model,
            self.local_model_optimizer,
            self.global_model,
            self.global_model_optimizer,
            args.tokenizer,
            self.encoder
        ) = load_models_per_party(pretrained, task_type, model_type, current_model_type, current_output_dim, is_local, device)

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

    def aggregate_local(self, pred_list):
        self.global_model.eval()
        with torch.no_grad():
            if self.args.model_type == 'Bert':  # pred_list[0] = [intermediate, attention_mask]
                # t1 = pred_list[0]
                # t2 = pred_list[1]
                # pred = self.global_model(t1, attention_mask=t2)

                pred_list_ = json.loads(pred_list)
                t1 = torch.Tensor(pred_list_[0])
                t2 = torch.Tensor(pred_list_[1])
                t1 = t1.to(self.args.device)
                t2 = t2.to(self.args.device)

                pred = self.global_model(t1, attention_mask=t2)

            elif self.args.model_type == 'GPT2':  # pred_list[0] = [intermediate, sequence_lengths, attention_mask]
                if self.args.task_type == 'CausalLM':  # pred_list[0] = [intermediate, attention_mask]
                    pred = self.global_model(pred_list[0][0], attention_mask=pred_list[0][1])
                elif self.args.task_type == 'SequenceClassification':  # pred_list[0] = [intermediate, ,sequence_lengths, attention_mask]
                    pred = self.global_model(pred_list[0][0], pred_list[0][1], attention_mask=pred_list[0][2])
                elif self.args.task_type == 'QuestionAnswering':  # pred_list[0] = [intermediate, attention_mask]
                    pred = self.global_model(pred_list[0][0], attention_mask=pred_list[0][1])
                else:
                    assert 1 > 2, 'Task type no supported'

            elif self.args.model_type == 'Llama':
                if self.args.task_type == 'CausalLM':  # pred_list[0] = [intermediate, attention_mask]
                    pred = self.global_model(pred_list[0][0], attention_mask=pred_list[0][1])
                elif self.args.task_type == 'SequenceClassification':  # pred_list[0] = [intermediate, ,sequence_lengths, attention_mask]
                    pred = self.global_model(pred_list[0][0], pred_list[0][1], attention_mask=pred_list[0][2])
                else:
                    assert 1 > 2, 'Task type no supported'

            self.global_pred = pred
            return pred

    def aggregate(self, pred_list):
        self.global_model.eval()
        with torch.no_grad():
            if self.args.model_type == 'Bert':  # pred_list[0] = [intermediate, attention_mask]
                t1 = torch.Tensor(pred_list[0])
                t2 = torch.Tensor(pred_list[1])
                t1 = t1.to(self.args.device)
                t2 = t2.to(self.args.device)
                print(t1.shape, t2.shape)

                pred = self.global_model(t1, attention_mask=t2)

            elif self.args.model_type == 'GPT2': # pred_list[0] = [intermediate, sequence_lengths, attention_mask]
                if self.args.task_type == 'CausalLM':# pred_list[0] = [intermediate, attention_mask]
                    pred = self.global_model(pred_list[0][0],  attention_mask=pred_list[0][1])
                elif self.args.task_type == 'SequenceClassification':# pred_list[0] = [intermediate, ,sequence_lengths, attention_mask]
                    pred = self.global_model(pred_list[0][0],  pred_list[0][1], attention_mask=pred_list[0][2])
                elif self.args.task_type == 'QuestionAnswering':# pred_list[0] = [intermediate, attention_mask]
                    pred = self.global_model(pred_list[0][0],  attention_mask=pred_list[0][1])
                else:
                    assert 1>2 , 'Task type no supported'

            elif self.args.model_type == 'Llama':
                if self.args.task_type == 'CausalLM':# pred_list[0] = [intermediate, attention_mask]
                    pred = self.global_model(pred_list[0][0],  attention_mask=pred_list[0][1])
                elif self.args.task_type == 'SequenceClassification':# pred_list[0] = [intermediate, ,sequence_lengths, attention_mask]
                    pred = self.global_model(pred_list[0][0],  pred_list[0][1], attention_mask=pred_list[0][2])
                else:
                    assert 1>2 , 'Task type no supported'

            self.global_pred = pred
            return pred
    
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
        
                
    def global_backward(self):

        if self.global_model_optimizer != None: 
            # server with trainable global layer
            _gradients = torch.autograd.grad(self.global_loss, self.global_pred, retain_graph=True)
            _gradients_clone = _gradients[0].detach().clone()

            # print('global_gradients_clone:',_gradients_clone)
            
            # update global model
            self.global_model_optimizer.zero_grad()
            parameters = []          
            if (self.args.apply_mid == True) and (self.index in self.args.defense_configs['party']): 
                # mid parameters
                for mid_model in self.global_model.mid_model_list:
                    parameters += list(mid_model.parameters())
                parameters += list(self.global_model.trainable_layer.parameters())
                # load grads into parameters
                weights_grad_a = torch.autograd.grad(self.global_pred, parameters, grad_outputs=_gradients_clone, retain_graph=True)
                for w, g in zip(parameters, weights_grad_a):
                    if w.requires_grad:
                        w.grad = g.detach()
            else:
                # trainable layer parameters
                # load grads into parameters
                weights_grad_a = torch.autograd.grad(self.global_pred, self.global_model.trainable_layer.parameters(), grad_outputs=_gradients_clone, retain_graph=True)
                for w, g in zip(self.global_model.trainable_layer.parameters(), weights_grad_a):
                    if w.requires_grad:
                        w.grad = g.detach()
                # print('weights_grad_a:',weights_grad_a)
                # non-trainabel layer: no need to update

            # for p in self.global_model.trainable_layer.parameters():#打印出每一层的参数的大小
            #     print(p)
            #     continue
            self.global_model_optimizer.step()
            # print('after==========')
            # for p in self.global_model.trainable_layer.parameters():#打印出每一层的参数的大小
            #     print(p)
            #     continue
            # assert 1>2

