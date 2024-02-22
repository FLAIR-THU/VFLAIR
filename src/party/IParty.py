from abc import ABCMeta, abstractmethod


class IParty(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def load_model(self, pred_list, parse_result_fn):
        pass

    @abstractmethod
    def load_dataset(self):
        pass

