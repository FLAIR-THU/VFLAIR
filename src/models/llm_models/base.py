import gc
from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Dict, Union
from transformers import PreTrainedModel
import os
from loguru import logger
import torch
from transformers import AutoTokenizer


class VFLModel(ABC):
    @abstractmethod
    def vfl_split(self, idx_of_layers: Iterable[int]) -> bool:
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def _clear_past_key_values(self):
        pass


class VFLPipeline(ABC):
    def __init__(self, split_index=Union[int, Tuple[int]], is_server=None):
        self.__split_index = split_index
        self.is_server = is_server

    @property
    def num_of_slices(self) -> int:
        return len(self.split_index) + 1

    @property
    def split_index(self) -> Tuple[int]:
        if isinstance(self.__split_index, Tuple):
            if len(self.__split_index) > 2:
                raise ValueError(f"Not supported split_index len:{self.__split_index}")
            return self.__split_index
        elif isinstance(self.__split_index, int):
            return (self.__split_index,)
        else:
            raise ValueError(f"Not supported split_index:{self.__split_index}")

    @property
    def _model_index(self) -> Iterable[int]:
        if self.is_server is None:
            return range(self.num_of_slices)
        elif self.is_server:
            return {1}
        else:
            idx = set(range(self.num_of_slices))
            idx.remove(1)
            return idx

    def _vfl_model_folder(self, model_path):
        return f"{model_path}_vfl_{self.__split_index}"

    def from_pretrained(self, model_name_or_path: str, **kwargs):
        try:
            return self.from_vfl(model_name_or_path, **kwargs)
        except Exception as e:
            logger.warning(f"{repr(e)}\nTry to load from raw model")
            return self._from_raw(model_name_or_path, **kwargs)

    @staticmethod
    def save_pretrained(model_name_or_path: str, models: Dict[int, PreTrainedModel], **kwargs):
        print(f'VFLPipeline save_pretrained')
        for i, m in models.items():
            m.save_pretrained(os.path.join(model_name_or_path, f"model_{i}"), **kwargs)

    def from_vfl(self, model_name_or_path, **kwargs) -> Dict[int, Union[PreTrainedModel, VFLModel]]:
        """
        try to load from local split model
        :param model_name_or_path:
        :param kwargs:
        :return:
        """
        _models = {}

        for i in self._model_index:
            model_path = os.path.join(model_name_or_path, f"model_{i}")
            if not os.path.exists(model_path):
                # check if vfl model exists
                if os.path.exists(self._vfl_model_folder(model_name_or_path)):
                    logger.info(f"Try existing vfl model: {self._vfl_model_folder(model_name_or_path)}")
                    return self.from_vfl(self._vfl_model_folder(model_name_or_path), **kwargs)
                else:
                    raise ValueError(f"Not found required vfl model in {model_name_or_path}")
            if i == 0:
                _model = self._load_model_head(model_path, **kwargs)
            elif i == self.num_of_slices - 1:
                _model = self._load_model_tail(model_path, **kwargs)
            else:
                _model = self._load_model_body(model_path, **kwargs)
            _models.update({i: _model})

        return _models

    def _from_raw(self, model_name_or_path, **kwargs) -> Dict[int, Union[PreTrainedModel, VFLModel]]:
        """
        try to load from raw model locally or remotely
        usually split, save and reload from split model
        :param model_name_or_path:
        :param kwargs:
        :return:
        """
        for i in self._model_index:
            if i == 0:
                _model = self._load_model_head(model_name_or_path, do_split=True, **kwargs)
            elif i == self.num_of_slices - 1:
                _model = self._load_model_tail(model_name_or_path, do_split=True, **kwargs)
            else:
                _model = self._load_model_body(model_name_or_path, do_split=True, **kwargs)
            self.save_pretrained(self._vfl_model_folder(model_name_or_path), models={i: _model})
            del _model
            gc.collect()
            torch.cuda.empty_cache()
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.save_pretrained(self._vfl_model_folder(model_name_or_path))
        return self.from_vfl(self._vfl_model_folder(model_name_or_path), **kwargs)

    # @abstractmethod
    # def _load_model_head(self, model_name_or_path, do_split=False, **kwargs) -> Union[PreTrainedModel, VFLModel]:
    #     pass

    # @abstractmethod
    # def _load_model_tail(self, model_name_or_path, do_split=False, **kwargs) -> Union[PreTrainedModel, VFLModel]:
    #     pass

    # @abstractmethod
    # def _load_model_body(self, model_name_or_path, do_split=False, **kwargs) -> Union[PreTrainedModel, VFLModel]:
    #     pass
