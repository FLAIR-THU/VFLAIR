from party.ICommunication import ICommunication


class LocalCommunication(ICommunication):
    __active_party = None

    def __init__(self, active_party):
        self.__active_party = active_party

    def send_pred_message(self, pred_list, parse_result_fn=None, use_cache=False, test="True"):
        return self.__active_party.aggregate(pred_list, use_cache=use_cache, test=test)

    def send_global_backward_message(self):
        self.__active_party.global_backward()

    def send_global_loss_and_gradients(self, gradients):
        self.__active_party.receive_loss_and_gradients(gradients)

    def send_cal_passive_local_gradient_message(self, index):
        return self.__active_party.cal_passive_local_gradient(index, remote=False)

    def send_global_lr_decay(self, i_epoch):
        # for ik in range(self.k):
        #     self.parties[ik].LR_decay(i_epoch)
        self.__active_party.global_LR_decay(i_epoch)

    def send_global_model_train_message(self):
        self.__active_party.train()
