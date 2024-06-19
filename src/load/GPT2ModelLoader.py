from .IModelLoader import IModelLoader
from transformers import PreTrainedModel, AutoTokenizer
from config import vfl_basic_config
from models.llm_models.gpt2_new import VFLPipelineGPT2
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftModelForCausalLM


class GPT2ModelLoader(IModelLoader):
    _models = {}  # type:dict[int,PreTrainedModel]

    def load(self, args, model_path, is_active_party):
        print(f'GPT2ModelLoader.load() {is_active_party}')
        # self.split_index = kwargs.get('split_index', (2,))
        print(f'VFLPipelineGPT2: {vfl_basic_config.split_index} {is_active_party}')
        p = VFLPipelineGPT2(vfl_basic_config.split_index, is_active_party)
        self._models.update(p.from_pretrained(model_path, **vfl_basic_config.kwargs_model_loading))
        print(f'{self._models.keys()}')
        
        config, generation_config = self._prepare_model_update_args()
        model_architectures = config.architectures
        model_embedded_dim = config.n_embd

        # print('origin trainable param:')
        # self._models[0].print_trainable_parameters()

        # for param in self._models[0].parameters():
        #     param.requires_grad = False

        if args.finetune_name == "LoRA":
            for i, m in self._models.items():
                peft_model = self._set_peft(m)
                self._models.update({i: peft_model})
        
        print('after lora trainable param:')
        self._models[0].print_trainable_parameters()


        if not is_active_party:
            encoder_trainable_ids = args.encoder_trainable_ids_list[0]
            print('encoder_trainable_ids = ', encoder_trainable_ids)
            for encoder_id in range(len(self._models[0].h)):
                if encoder_id not in encoder_trainable_ids: # freeze encoders that's not needed
                    for param in self._models[0].h.parameters():
                        param.requires_grad = False

            print('embedding_trainable = ', args.embedding_trainable[0])
            if not args.embedding_trainable[0]: # freeze embeddings that's not needed
                for param in self._models[0].wte.parameters():
                    param.requires_grad = False
                for param in self._models[0].wpe.parameters():
                    param.requires_grad = False
        else:
            encoder_trainable_ids = args.encoder_trainable_ids_list[1]
            print('encoder_trainable_ids = ', encoder_trainable_ids)
            for encoder_id in range(len(self._models[0].h)):
                if encoder_id not in encoder_trainable_ids: # freeze encoders that's not needed
                    for param in self._models[0].h.parameters():
                        param.requires_grad = False

        print('after config trainable param:')
        self._models[0].print_trainable_parameters()

        return {
            # "tokenizer": tokenizer,
            "models": self._models,
            "config": config,
            "generation_config": generation_config,
            "model_architectures": model_architectures,
            "model_embedded_dim": model_embedded_dim
        }

    def _set_peft(self, model):
        """
        peft training or load trained peft weights
        :return:
        """
        # print('args.finetune_detail_configs:',args.finetune_detail_configs)
        if args.finetune_detail_configs != None:
            lora_config = LoraConfig(
                **args.finetune_detail_configs
            )
        else:
            lora_config = LoraConfig(
                inference_mode=False,  
                r=4,  
                lora_alpha=32, 
                lora_dropout=0.1
            )

        def get_lora_model(model):
            model.enable_input_require_grads()
            peft_model = get_peft_model(model, lora_config)
            return peft_model

        model = get_lora_model(model)
        print('after lora')
        model.print_trainable_parameters()
        return model

    def _prepare_model_update_args(self):
        model = None
        for m in self._models.values():
            if m:
                model = m

        if model is not None:
            return model.config, model.generation_config
        return None, None
