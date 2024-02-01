from party.ICommunication import ICommunication
import json
import framework.protos.message_pb2 as fpm
import framework.protos.node_pb2 as fpn
import framework.common.MessageUtil as mu
import torch
from load.LoadModels import QuestionAnsweringModelOutput

class DistributedCommunication(ICommunication):
    _client = None
    _device = None

    def __init__(self, client, device):
        self._client = client
        self._device = device

    def send_pred_message(self, pred_list):
        value = fpm.Value()
        new_list = [item.tolist() for item in pred_list[0]]

        value.string = json.dumps(new_list)
        node = fpn.Node(node_id=self._client.id)
        msg = mu.MessageUtil.create(node, {"pred_list": value}, 4)
        response = self._client.open_and_send(msg)
        result = response.named_values['test_logit'].string
        test_logit = json.loads(result)

        start_logits = torch.Tensor(test_logit['start_logits'])
        end_logits = torch.Tensor(test_logit['end_logits'])

        test_logit_output = QuestionAnsweringModelOutput(
            loss=None,
            start_logits=start_logits.to(self._device),
            end_logits=end_logits.to(self._device),
            hidden_states=None,
            attentions=None,
        )
        return test_logit_output

    def send_global_backward_message(self):
        pass

    def send_global_loss_and_gradients(self, loss, gradients):
        pass

    def send_cal_passive_local_gradient_message(self, pred):
        pass

    def send_global_lr_decay(self, i_epoch):
        pass

    def send_global_modal_train_message(self):
        pass

