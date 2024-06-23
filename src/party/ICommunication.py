from abc import ABCMeta, abstractmethod


class ICommunication(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def send_pred_message(self, pred_list, parse_result_fn, use_cache=False):
        pass

    @abstractmethod
    def send_global_backward_message(self):
        pass

    @abstractmethod
    def send_global_loss_and_gradients(self, gradients):
        pass

    @abstractmethod
    def send_cal_passive_local_gradient_message(self, index):
        pass

    @abstractmethod
    def send_global_lr_decay(self, i_epoch):
        pass

    @abstractmethod
    def send_global_model_train_message(self):
        pass
