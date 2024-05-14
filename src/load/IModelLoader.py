from abc import ABCMeta, abstractmethod


class IModelLoader(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def load(self, path: str, is_active_party: bool):
        pass
