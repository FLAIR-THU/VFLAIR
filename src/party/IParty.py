from abc import ABCMeta, abstractmethod


class IParty(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def load_model(self, pred_list, parse_result_fn):
        pass

    @abstractmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def prepare_data_loader(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def train(self, i_epoch):
        pass

