'''
all configs shall not be changed during process
'''
import os
import datetime
import sys

from peft import PeftConfig, LoraConfig, TaskType
from transformers import TrainingArguments
import torch
from loguru import logger

# todo: test prod env

# indicator whether to use the new pipeline
_new_pipeline = False
is_test = False
if not is_test:
    logger.remove()
    logger.add(sys.stderr, level="INFO")


class VFLBasicConfig(object):
    num_of_slice = 2
    seed = 7
    kwargs_model_loading = {'device_map': 'auto',
                            'max_memory': {4: '20Gib'},
                            'torch_dtype': torch.bfloat16,
                            }

    def __init__(self, **kwargs):
        self.vfl_training_config = VFLTrainingConfig(self)
        # self.vfl_training_config = None
    @property
    def is_inference(self):
        return not self.vfl_training_config


class VFLTrainingConfig(object):
    def __init__(self, vbc: VFLBasicConfig):
        self.is_training = True
        self.vfl_basic_config = vbc
        self.__trainable_slice = (-1, 0)
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False,  # 训练模式
            r=4,  # Lora 秩
            lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
            lora_dropout=0.1,  # Dropout 比例
            # layers_to_transform=[0, 1]
        )  # type:PeftConfig
        self.training_args = TrainingArguments(
            # todo: save path
            output_dir=os.path.join('/mnt/data/projects/PlatForm/src/exp_result/dev'),
            logging_dir=os.path.join('/mnt/data/projects/PlatForm/src/exp_result/dev', 'logs',
                                     f'{datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")}_{"test" if is_test else "prod"}_{self.vfl_basic_config.num_of_slice}_{self.__trainable_slice}'),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            logging_steps=10,
            num_train_epochs=10,
            save_steps=50,
            learning_rate=1e-5,
            save_on_each_node=True,
            gradient_checkpointing=True,
            report_to=['tensorboard'],
            eval_steps=50
        )  # type:TrainingArguments

    def __bool__(self):
        return self.is_training

    @property
    def trainable_slice(self):
        """
        support positive and negative setting
        :return:
        """
        for i in self.__trainable_slice:
            yield range(self.vfl_basic_config.num_of_slice)[i]


vfl_basic_config = VFLBasicConfig()
