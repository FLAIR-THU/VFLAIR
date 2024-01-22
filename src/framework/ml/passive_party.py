import sys, os
import torch
sys.path.append(os.pardir)
from utils.basic_functions import cross_entropy_for_onehot, tf_distance_cov_cor
from framework.ml.llm_party import Party as Party_LLM
from dataset.party_dataset import PassiveDataset_LLM
from framework.ml.LoadModels import load_models_per_party
from framework.ml.LoadDataset import load_dataset_per_party_llm
import framework.protos.message_pb2 as fpm
import framework.protos.node_pb2 as fpn
import framework.common.MessageUtil as mu
from framework.ml.LoadModels import QuestionAnsweringModelOutput
import json
import collections
from utils.squad_utils import normalize_answer

class PassiveParty_LLM(Party_LLM):
    _client = None

    def __init__(self, args, client):
        super().__init__()
        self._client = client
        args.need_auxiliary = 0
        args.dataset = args.dataset_split['dataset_name']
        self.args = args
        if args.device == 'cuda':
            cuda_id = args.gpu
            torch.cuda.set_device(cuda_id)
            print(f'running on cuda{torch.cuda.current_device()}')
        self.prepare_model(args)
        self.prepare_data(args)
        self.name = "server#passive"
        self.criterion = cross_entropy_for_onehot
        # self.encoder = args.encoder
        self.train_index = args.idx_train
        self.test_index = args.idx_test
        
        self.gt_one_hot_label = None

        self.pred_received = []
        for _ in range(args.k):
            self.pred_received.append([])
        
        self.global_pred = None
        self.global_loss = None

    def prepare_model(self, args):
        current_model_type = args.model_list['0']['type']
        pretrained = args.pretrained
        task_type = args.task_type
        model_type = args.model_type
        current_output_dim = args.model_list['0']['output_dim']
        is_local = True
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

    def prepare_data(self, args):
        (
            args,
            self.half_dim,
            train_dst,
            test_dst,
        ) = load_dataset_per_party_llm(args)

        self.train_data, self.train_label = train_dst
        self.test_data, self.test_label = test_dst
 
        self.train_dst = PassiveDataset_LLM(args, self.train_data, self.train_label)
        self.test_dst = PassiveDataset_LLM(args, self.test_data, self.test_label)

    def prepare_data_loader(self):
        super().prepare_data_loader(self.args.batch_size, self.args.need_auxiliary)
            
    def update_local_pred(self, pred):
        self.pred_received[self.args.k-1] = pred
    
    def receive_pred(self, pred, giver_index):
        self.pred_received[giver_index] = pred

    def cal_loss(self, test=False):
        gt_one_hot_label = self.gt_one_hot_label
        pred =  self.global_pred
        # print('== In cal_loss ==')
        # print('gt_one_hot_label:',gt_one_hot_label)
        # print('pred:',pred)

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

    def predict(self):
        data_loader_list = [self.test_loader]
        exact_score_list = []
        f1_list = []
        with torch.no_grad():
            for parties_data in zip(*data_loader_list):
                parties_data = [[_data[0], _data[1], _data[2], _data[3], _data[4]] for _data in parties_data]

                if parties_data[0][3] == []:
                    parties_data[0][3] = None
                if parties_data[0][4] == []:
                    parties_data[0][4] = None

                if self.args.task_type == "SequenceClassification" and self.num_classes > 1:  # regression
                    gt_val_one_hot_label = self.label_to_one_hot(parties_data[0][1], self.num_classes)
                else:
                    gt_val_one_hot_label = parties_data[0][1]

                pred_list = self.predict_batch(gt_val_one_hot_label, parties_data)
                # test_logit = self._send_message_local(pred_list)
                test_logit = self._send_message(pred_list)

                start_logits = torch.Tensor(test_logit['start_logits'])
                end_logits = torch.Tensor(test_logit['end_logits'])

                test_logit_output = QuestionAnsweringModelOutput(
                    loss=None,
                    start_logits=start_logits.to(self.args.device),
                    end_logits=end_logits.to(self.args.device),
                    hidden_states=None,
                    attentions=None,
                )

                exact_scores, f1s = self.output(test_logit_output, gt_val_one_hot_label, parties_data)
                exact_score_list.extend(exact_scores)
                f1_list.extend(f1s)
                del parties_data
        return exact_score_list, f1_list
    def predict_batch(self, gt_val_one_hot_label, parties_data):
        input_shape = parties_data[0][0].shape[:2]
        self.input_shape = input_shape
        self.obtain_local_data(parties_data[0][0])
        self.gt_one_hot_label = gt_val_one_hot_label
        self.local_batch_attention_mask = parties_data[0][2]
        self.local_batch_token_type_ids = parties_data[0][3]

        if self.args.model_type == 'Bert':
            _local_pred, _local_pred_detach, _local_attention_mask = self.give_pred()  # , _input_shape
            return [_local_pred, _local_attention_mask]

        elif self.args.model_type == 'GPT2':
            if self.args.task_type == 'SequenceClassification':
                _local_pred, _local_pred_detach, _local_sequence_lengths, _local_attention_mask = self.give_pred()  # , _input_shape
                return [_local_pred, _local_sequence_lengths, _local_attention_mask]
            elif self.args.task_type == 'CausalLM':
                _local_pred, _local_pred_detach, _local_attention_mask = self.give_pred()  # , _input_shape
                return [_local_pred, _local_attention_mask]
            elif self.args.task_type == 'QuestionAnswering':
                _local_pred, _local_pred_detach, _local_attention_mask = self.give_pred()  # , _input_shape
                return [_local_pred, _local_attention_mask]


        elif self.args.model_type == 'Llama':
            if self.args.task_type == 'SequenceClassification':
                _local_pred, _local_pred_detach, _local_sequence_lengths, _local_attention_mask = self.give_pred()  # , _input_shape
                return [_local_pred, _local_sequence_lengths, _local_attention_mask]
            elif self.args.task_type == 'CausalLM':
                _local_pred, _local_pred_detach, _local_attention_mask = self.give_pred()  # , _input_shape
                return [_local_pred, _local_attention_mask]
            elif self.args.task_type == 'QuestionAnswering':
                _local_pred, _local_pred_detach, _local_attention_mask = self.give_pred()  # , _input_shape
                return [_local_pred, _local_attention_mask]

    def output(self, test_logit, gt_val_one_hot_label, parties_data):
        if self.args.task_type == "SequenceClassification":
            if self.num_classes == 1:
                predict_label = test_logit.detach().cpu()
                actual_label = gt_val_one_hot_label.detach().cpu()

                predict_label = torch.tensor([_.item() for _ in predict_label])
                actual_label = torch.tensor([_.item() for _ in actual_label])

                # test_predict_labels.extend(list(predict_label))
                # test_actual_labels.extend(list(actual_label))
            else:  # Classification
                enc_predict_prob = test_logit

                predict_label = torch.argmax(enc_predict_prob, dim=-1)
                actual_label = torch.argmax(gt_val_one_hot_label, dim=-1)

                # test_preds.append(list(enc_predict_prob.detach().cpu().numpy()))
                # test_targets.append(list(gt_val_one_hot_label.detach().cpu().numpy()))
                #
                # test_predict_labels.extend(list(predict_label.detach().cpu()))
                # test_actual_labels.extend(list(actual_label.detach().cpu()))
                #
                # sample_cnt += predict_label.shape[0]
                # suc_cnt += torch.sum(predict_label == actual_label).item()

        elif self.args.task_type == "QuestionAnswering":

            def _get_best_indexes(logits, n_best_size=20):
                """Get the n-best logits from a list."""
                index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
                best_indexes = []
                for i in range(len(index_and_score)):
                    if i >= n_best_size:
                        break
                    best_indexes.append(index_and_score[i][0])
                return best_indexes

            def get_tokens(s):
                if not s: return []
                return normalize_answer(s).split()

            def compute_exact(a_gold, a_pred):
                return int(normalize_answer(a_gold) == normalize_answer(a_pred))

            def compute_f1(a_gold, a_pred):
                gold_toks = get_tokens(a_gold)
                pred_toks = get_tokens(a_pred)
                common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
                num_same = sum(common.values())
                if len(gold_toks) == 0 or len(pred_toks) == 0:
                    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
                    return int(gold_toks == pred_toks)
                if num_same == 0:
                    return 0
                precision = 1.0 * num_same / len(pred_toks)
                recall = 1.0 * num_same / len(gold_toks)
                f1 = (2 * precision * recall) / (precision + recall)
                return f1

            start_logits = test_logit.start_logits
            end_logits = test_logit.end_logits

            n_best_size = self.args.n_best_size
            start_indexes = [_get_best_indexes(_logits, n_best_size) for _logits in start_logits]
            end_indexes = [_get_best_indexes(_logits, n_best_size) for _logits in end_logits]

            exact_score_list = []
            f1_list = []

            for i in range(start_logits.shape[0]):
                # for each sample in this batch
                _start_logits = start_logits[i]
                _end_logits = end_logits[i]
                _start_indexes = start_indexes[i]
                _end_indexes = end_indexes[i]

                ############ Gold ################
                feature = parties_data[0][4]
                feature_tokens = [_token[0] for _token in feature["tokens"]]

                gold_start_indexs, gold_end_indexs = gt_val_one_hot_label[0]
                gold_ans = []
                for _i in range(len(gold_start_indexs)):
                    gold_start_index = int(gold_start_indexs[_i])
                    gold_end_index = int(gold_end_indexs[_i])

                    gold_ans_text = " ".join(feature_tokens[gold_start_index:(gold_end_index + 1)])
                    gold_ans_text = normalize_answer(gold_ans_text)
                    gold_ans.append(gold_ans_text)

                # print('gold_ans:',gold_ans,feature["orig_answer_text"])

                ############ Pred ################
                _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                    "PrelimPrediction",
                    ["start_index", "end_index", "start_logit", "end_logit"])
                _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                    "NbestPrediction", ["text", "start_logit", "end_logit"])

                # iterate through all possible start-end pairs
                prelim_predictions = []
                for start_index in _start_indexes:
                    for end_index in _end_indexes:
                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        if start_index >= len(feature["tokens"]):
                            continue
                        if end_index >= len(feature["tokens"]):
                            continue
                        if start_index not in feature["token_to_orig_map"]:
                            continue
                        if end_index not in feature["token_to_orig_map"]:
                            continue
                        if not feature["token_is_max_context"].get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > self.args.max_answer_length:
                            continue

                        prelim_predictions.append(
                            _PrelimPrediction(
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=_start_logits[start_index],
                                end_logit=_end_logits[end_index]))

                # Iterate through Sorted Predictions
                prelim_predictions = sorted(
                    prelim_predictions,
                    key=lambda x: (x.start_logit + x.end_logit),
                    reverse=True)
                _exact_score_list = []
                _f1_list = []
                exact_score = 0
                f1 = 0
                # Get n best prediction text
                nbest = []
                n_best_size = min(n_best_size, len(prelim_predictions))
                for _id in range(n_best_size):
                    start_index = prelim_predictions[_id].start_index
                    end_index = prelim_predictions[_id].end_index

                    pred_ans_text = " ".join(feature_tokens[start_index:(end_index + 1)])
                    pred_ans_text = normalize_answer(pred_ans_text)

                    nbest.append(
                        _NbestPrediction(
                            text=pred_ans_text,
                            start_logit=prelim_predictions[_id].start_logit,
                            end_logit=prelim_predictions[_id].end_logit))

                # Get best predicted answer
                total_scores = []
                best_non_null_entry = None

                if self.args.metric_type == "best_bred":
                    for entry in nbest:
                        total_scores.append(entry.start_logit + entry.end_logit)
                        if not best_non_null_entry:
                            if entry.text:
                                best_non_null_entry = entry
                        pred_ans_text = best_non_null_entry.text if (best_non_null_entry != None) else ""
                    # Calculate exact_score/f1
                    # print('best pred:',pred_ans_text)
                    exact_score = max(exact_score, max(compute_exact(a, pred_ans_text) for a in gold_ans))
                    f1 = max(f1, max(compute_f1(a, pred_ans_text) for a in gold_ans))
                    # print('this batch:', exact_score, f1)
                    exact_score_list.append(exact_score)
                    f1_list.append(f1)
                elif self.args.metric_type == "n_best":
                    for entry in nbest:
                        total_scores.append(entry.start_logit + entry.end_logit)
                        if not best_non_null_entry:
                            if entry.text:
                                best_non_null_entry = entry
                        pred_ans_text = entry.text
                        # Calculate exact_score/f1
                        # print('best pred:',pred_ans_text)
                        exact_score = max(exact_score, max(compute_exact(a, pred_ans_text) for a in gold_ans))
                        f1 = max(f1, max(compute_f1(a, pred_ans_text) for a in gold_ans))
                        # print('this batch:', exact_score, f1)
                        exact_score_list.append(exact_score)
                        f1_list.append(f1)

                return exact_score_list, f1_list

        else:
            assert 1 > 2, "task_type not supported"

    def _send_message_local(self, pred_list):
        # return self.args.parties[self.args.k - 1].aggregate_local(pred_list, test="True")
        new_list = [item.tolist() for item in pred_list]

        value = json.dumps(new_list)
        return self.args.parties[self.args.k - 1].aggregate_local(value)

    def _send_message(self, pred_list):
        value = fpm.Value()
        new_list = [item.tolist() for item in pred_list]

        value.string = json.dumps(new_list)
        node = fpn.Node(node_id=self._client.id)
        msg = mu.MessageUtil.create(node, {"pred_list": value}, 4)
        response = self._client.open_and_send(msg)
        result = response.named_values['test_logit'].string
        return json.loads(result)

