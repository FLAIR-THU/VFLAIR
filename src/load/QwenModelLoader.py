from .IModelLoader import IModelLoader
from transformers import PreTrainedModel, AutoTokenizer
from config import vfl_basic_config
from models.llm_models.qwen2 import VFLPipelineQwen


class QwenModelLoader(IModelLoader):
    _models = {}  # type:dict[int,PreTrainedModel]

    def load(self, model_path, is_active_party):
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        p = VFLPipelineQwen(vfl_basic_config.split_index, is_active_party)
        self._models.update(p.from_pretrained(model_path, **vfl_basic_config.kwargs_model_loading))
        config, generation_config = self._prepare_model_update_args()
        return {
            "tokenizer": tokenizer,
            "models": self._models,
            "config": config,
            "generation_config": generation_config
        }

    def _prepare_model_update_args(self):
        model = None
        for m in self._models.values():
            if m:
                model = m

        if model is not None:
            return model.config, model.generation_config
        return None, None
