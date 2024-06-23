import datetime
import os
if not os.getenv('CUDA_VISIBLE_DEVICES'):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(6,7)])

import sys

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 定义模型训练函数
import torch
import random
from transformers import AutoTokenizer, AutoConfig, AutoModel, DataCollatorForSeq2Seq, TrainingArguments, Trainer, \
    GenerationConfig, PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset
import pandas as pd
from models.llm_models.qwen2 import Qwen2ModelHead, Qwen2TailForCausalLM, Qwen2DecoderLayerParam, \
    Qwen2ForCausalLM, Qwen2Config, E2EModel, VFLPipelineQwen
import pynvml
from loguru import logger
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
from safetensors.torch import load_file, save_file
import time
import json
import copy
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftModelForCausalLM
import numpy as np
from tqdm import tqdm

try:
    pynvml.nvmlInit()
except:
    pass
# device_count = pynvml.nvmlDeviceGetCount()


GPU_IDX = [int(i) for i in os.getenv('CUDA_VISIBLE_DEVICES').split(',')]

SEED = 7
BATCH_SIZE = 16
sample_rng = np.random.default_rng(SEED)

_qwen_72b_device_map = {'model.embed_tokens': 0,
                        'model.layers.0': 0,
                        'model.layers.1': 0,
                        'model.layers.2': 0,
                        'model.layers.3': 0,
                        'model.layers.4': 0,
                        'model.layers.5': 0,
                        'model.layers.6': 0,
                        'model.layers.7': 0,
                        'model.layers.8': 1,
                        'model.layers.9': 1,
                        'model.layers.10': 1,
                        'model.layers.11': 1,
                        'model.layers.12': 1,
                        'model.layers.13': 1,
                        'model.layers.14': 1,
                        'model.layers.15': 1,
                        'model.layers.16': 1,
                        'model.layers.17': 1,
                        'model.layers.18': 1,
                        'model.layers.19': 2,
                        'model.layers.20': 2,
                        'model.layers.21': 2,
                        'model.layers.22': 2,
                        'model.layers.23': 2,
                        'model.layers.24': 2,
                        'model.layers.25': 2,
                        'model.layers.26': 2,
                        'model.layers.27': 2,
                        'model.layers.28': 2,
                        'model.layers.29': 2,
                        'model.layers.30': 3,
                        'model.layers.31': 3,
                        'model.layers.32': 3,
                        'model.layers.33': 3,
                        'model.layers.34': 3,
                        'model.layers.35': 3,
                        'model.layers.36': 3,
                        'model.layers.37': 3,
                        'model.layers.38': 3,
                        'model.layers.39': 3,
                        'model.layers.40': 3,
                        'model.layers.41': 4,
                        'model.layers.42': 4,
                        'model.layers.43': 4,
                        'model.layers.44': 4,
                        'model.layers.45': 4,
                        'model.layers.46': 4,
                        'model.layers.47': 4,
                        'model.layers.48': 4,
                        'model.layers.49': 4,
                        'model.layers.50': 4,
                        'model.layers.51': 4,
                        'model.layers.52': 5,
                        'model.layers.53': 5,
                        'model.layers.54': 5,
                        'model.layers.55': 5,
                        'model.layers.56': 5,
                        'model.layers.57': 5,
                        'model.layers.58': 5,
                        'model.layers.59': 5,
                        'model.layers.60': 5,
                        'model.layers.61': 5,
                        'model.layers.62': 5,
                        'model.layers.63': 6,
                        'model.layers.64': 6,
                        'model.layers.65': 6,
                        'model.layers.66': 6,
                        'model.layers.67': 6,
                        'model.layers.68': 6,
                        'model.layers.69': 6,
                        'model.layers.70': 6,
                        'model.layers.71': 6,
                        'model.layers.72': 6,
                        'model.layers.73': 6,
                        'model.layers.74': 7,
                        'model.layers.75': 7,
                        'model.layers.76': 7,
                        'model.layers.77': 7,
                        'model.layers.78': 7,
                        'model.layers.79': 7,
                        'model.norm': 7,
                        'lm_head': 7}


def set_seed(seed=60):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def log_gpu_memory_usage():
    return None
    total_memery = len(GPU_IDX) * 24
    total_used = 0
    max_memery = {}
    for i in GPU_IDX:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        logger.info(f"GPU {i}: Used Memory: {info.used / 1024 ** 2} MB")
        max_memery.update({i: f"{int(22 - info.used / 1024 ** 3)}GB"})
        total_used += info.used / 1024 ** 3

    logger.info(f"GPU Memory\nTotal: {total_memery} GB\nUsed: {total_used} GB\nLeft: {total_memery - total_used} GB")
    return max_memery


def device_map_setter(kwargs: dict, add_model_prefix=False):
    new_kwargs = copy.deepcopy(kwargs)
    with open(os.path.join(model_path, 'config.json')) as f:
        model_config = json.loads(f.read())
    num_hidden_layers = model_config['num_hidden_layers']
    device_map = {'embed_tokens': 'cuda', 'norm': 'cuda', 'layers.0': 'cuda', 'layers.1': 'cuda', 'lm_head': 'meta'}
    for i in range(2, num_hidden_layers):
        device_map.update({f'layers.{i}': 'meta'})
    if add_model_prefix:
        for key in device_map.keys():
            if key != 'lm_head':
                device_map.update({f"model.{key}": device_map[key]})
                device_map.pop(key)
    new_kwargs.update({'device_map': device_map})
    return new_kwargs


# MODEL_PATH = '/dev/data/sunl0/model'
MODEL_PATH = '/mnt/data/model'
# MODEL_PATH='D:\project\PrivacyCompute\model'
# DATA_PATH = '/dev/data/sunl0/data'
DATA_PATH = '/mnt/data/data'
# DATA_PATH=r'D:\project\data'
SPLIT_INDEX = (2, -2)
IS_TEST = False
if not IS_TEST:
    logger.remove()
    logger.add(sys.stderr, level="INFO")
# model_path = '/dev/data/sunl0/model/test/qwen/Qwen1___5-72B-Chat'
model_path = os.path.join(MODEL_PATH, 'Qwen/Qwen1.5-0.5B-Chat')
# model_path = os.path.join(MODEL_PATH, 'Qwen/Qwen1___5-72B-Chat')

vfl_folder = model_path + f'_vfl_{SPLIT_INDEX}'
model_path_head = os.path.join(vfl_folder, 'model_head')
model_path_tail = os.path.join(vfl_folder, 'model_tail')
lora_path = '/mnt/data/model/Qwen/Qwen1.5-0.5B-Chat_3_test'
# lora_path = model_path + f'_lora_{SPLIT_INDEX}'

load_kwargs = {
    'device_map': 'auto',
    'torch_dtype': torch.bfloat16,
    'max_memory': {0: '20GB'}
    # 'max_memory': {0: '21GB', 1: '22GB', 2: '18GB', 3: '20GB', 4: '22GB', 5: '22GB', 6: '22GB', 7: '20GB'},
}


# def quantize_config():
#     from transformers import BitsAndBytesConfig
#     bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
#                                 bnb_4bit_compute_dtype=torch.bfloat16)
#     return bnb_config


class DevPipeline:
    def __init__(self, model_path=model_path):
        self.model_path = model_path
        self.tokenizer = None

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # tokenizer.save_pretrained('D:\project\PrivacyCompute\model\Qwen/Qwen1.5-0.5B-Chat_tokenizer')
        return self.tokenizer

    def tokenize_input(self, input="You are a python programmer, what can you do?"):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt")
        return model_inputs


def load_vfl_model(split_idx: [int, tuple] = SPLIT_INDEX, *args, **kwargs):
    if isinstance(split_idx, int) or len(split_idx) == 1:
        return load_vfl_model_2slice(split_idx, *args, **kwargs)
    elif isinstance(split_idx, tuple) and len(split_idx) == 2:
        return load_vfl_model_3slice(split_idx, *args, **kwargs)
    else:
        raise ValueError(f"Invalid split_idx: {split_idx}")


def load_vfl_model_2slice(split_idx=2, is_from_raw=True, save_model=False):
    logger.info(f"Loading VFL")
    log_gpu_memory_usage()
    p_server = VFLPipelineQwen(split_idx, True)
    p_client = VFLPipelineQwen(split_idx, False)
    # p_server = PipelineVFL2Slice(True)
    # p_client = PipelineVFL2Slice(False)
    # model_0, model_1,model_2 = None, None,None
    models = {}
    if is_from_raw:
        models.update(p_client.from_pretrained(model_path,
                                               **device_map_setter(load_kwargs)))
        models.update(p_server.from_pretrained(model_path,
                                               **load_kwargs))
        # models.update(p_client.from_pretrained(model_path, split_index=split_idx,
        #                                        **device_map_setter(load_kwargs)))
        # models.update(p_server.from_pretrained(model_path, split_index=split_idx,
        #                                        **load_kwargs))
        logger.info(f"finish loading models from raw")
        log_gpu_memory_usage()
    else:
        models.update(p_client.from_vfl(vfl_folder, **load_kwargs))
        models.update(p_server.from_vfl(vfl_folder, **load_kwargs))
        for i in range(len(models)):
            for param in models[i].parameters():
                param.requires_grad = False
        logger.info(f"finish loading models from vfl")
        log_gpu_memory_usage()
        # for i in range(len(models)):
        #     _p = os.path.join(vfl_folder, f"model_{i}")
        #     if os.path.exists(_p):
        #         if i == 0:
        #             __kwargs = copy.deepcopy(load_kwargs)
        #             __kwargs.update({'max_memory': {0: '20GiB'}})
        #             models.update(p_client.from_vfl(_p, **__kwargs))
        #         elif i == 1:
        #             __kwargs = copy.deepcopy(load_kwargs)
        #             __kwargs.update({'max_memory': {i: '14GiB' if i == 0 else '21GiB' for i in range(0, 8)}})
        #             models.update(p_server.from_vfl(_p, **__kwargs))
        #         else:
        #             continue
        #         for param in models[i].parameters():
        #             param.requires_grad = False
        #     else:
        #         break
        # logger.info(f"finish loading models from vfl")
        # log_gpu_memory_usage()
    if save_model:
        pass
        # logger.info(f"save!")
        # time.sleep(3)
        # logger.info(f"release resources and norm to cuda")
        # for i, m in models.items():
        #     if m:
        #         PipelineVFL2Slice.save_vfl(os.path.join(vfl_folder, f"model_{i}"), m)
        #     else:
        #         break

    return models


def load_vfl_model_3slice(split_idx=(2, -2), is_from_raw=True, save_model=False):
    logger.info(f"Loading VFL")
    log_gpu_memory_usage()
    p_server = VFLPipelineQwen(split_idx, True)
    p_client = VFLPipelineQwen(split_idx, False)
    # p_server = PipelineVFL3Slice(True)
    # p_client = PipelineVFL3Slice(False)
    models = {k: None for k in range(3)}
    if is_from_raw:
        models.update(p_client.from_pretrained(model_path,
                                               **load_kwargs))
        models.update(p_server.from_pretrained(model_path,
                                               **load_kwargs))
        # models.update(p_client.from_pretrained(model_path, split_index=split_idx,
        #                                        **load_kwargs))
        # models.update(p_server.from_pretrained(model_path, split_index=split_idx,
        #                                        **load_kwargs))
        logger.info(f"finish loading models from raw")
        log_gpu_memory_usage()
    else:
        models.update(p_client.from_vfl(vfl_folder, **load_kwargs))
        models.update(p_server.from_vfl(vfl_folder, **load_kwargs))
        for i in range(len(models)):
            for param in models[i].parameters():
                param.requires_grad = False
        logger.info(f"finish loading models from vfl")
        log_gpu_memory_usage()
    if save_model:
        pass
        # logger.info(f"save!")
        # time.sleep(3)
        # logger.info(f"release resources and norm to cuda")
        # for i, m in models.items():
        #     if m:
        #         PipelineVFL3Slice.save_vfl(os.path.join(vfl_folder, f"model_{i}"), m)
        #     else:
        #         continue

    return models


def get_trainable_parameters(model):
    for k, param in model.named_parameters():
        if param.requires_grad:
            # logger.debug(f"add trainable parameters: {k}")
            yield param


def check1():
    model_weight = load_file(os.path.join(model_path_tail, 'model.safetensors'))
    model_weight = load_file(os.path.join(lora_path, 'checkpoint-manual', 'adapter_model.safetensors'))

    return model_weight


def transformers_register():
    AutoConfig.register()


def process_func(example):
    MAX_LENGTH = 128  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []

    instruction_token = tokenizer(
        f"<|im_start|>system\n现在你要扮演皇帝身边的女人--甄嬛<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction_token["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction_token["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction_token["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    # labels = instruction_token["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    else:
        while len(input_ids) < MAX_LENGTH:
            input_ids.append(tokenizer.pad_token_id)
            attention_mask.append(0)
            labels.append(tokenizer.pad_token_id)

    # is_train = sample_rng.random() > 0.1
    # if device:
    #     return {
    #         "input_ids": torch.Tensor(input_ids).to(device),
    #         "attention_mask": torch.Tensor(attention_mask).to(device),
    #         "labels": torch.Tensor(labels).to(device),
    #         "is_train": is_train
    #     }
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        # "is_train": is_train
    }


def dataset_batch_processor(batch):
    res = {}
    for i, row in enumerate(batch):
        for k, v in row.items():
            if k not in res:
                res.update({k: [v]})
            else:
                res[k].append(v)
    return res


def fine_tune_dataset():
    # 将JSON文件转换为CSV文件
    df = pd.read_json(os.path.join(DATA_PATH, 'huanhuan.json'))
    ds = Dataset.from_pandas(df)
    ds.shuffle(seed=SEED)
    ds = ds.map(process_func, remove_columns=ds.column_names)
    # ds.split
    ans = ds.train_test_split(test_size=0.1, seed=SEED)
    return ans['train'], ans['test']


lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
    # layers_to_transform=[0, 1]
)


class PipelineLoraFinetune:
    def __init__(self, is_test=True, path_save=lora_path):
        if is_test:
            self.train_args = TrainingArguments(
                output_dir=os.path.join(path_save),
                logging_dir=os.path.join(path_save, 'logs',
                                         f'{datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")}_test'),
                per_device_train_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=1,
                logging_steps=10,
                num_train_epochs=2,
                save_steps=50,
                learning_rate=1e-5,
                save_on_each_node=True,
                gradient_checkpointing=True,
                report_to=['tensorboard'],
                eval_steps=210
            )
        else:
            self.train_args = TrainingArguments(
                output_dir=os.path.join(path_save),
                logging_dir=os.path.join(path_save, 'logs', f'{datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")}'),
                per_device_train_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=1,
                logging_steps=10,
                num_train_epochs=10,
                save_steps=210,
                # save_steps=50,
                learning_rate=1e-5,
                save_on_each_node=True,
                gradient_checkpointing=True,
                report_to=['tensorboard'],
                eval_steps=210
            )

    @staticmethod
    def get_lora_model(model):
        model.enable_input_require_grads()
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()
        return peft_model

    def train_model(self, model, dataset_train, dataset_validate, is_trainer=False):
        if is_trainer:
            self._train_model_use_trainer(model, dataset_train, dataset_validate)
        else:
            self.train_e2emodel(model, dataset_train, dataset_validate)

    def _train_model_use_trainer(self, model, dataset_train, dataset_validate):

        trainer = Trainer(
            model=model,
            args=self.train_args,
            train_dataset=dataset_train,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
            eval_dataset=dataset_validate
        )
        trainer.train()

    def train_e2emodel(self, model: PreTrainedModel, dataset_train, dataset_validate):
        training_args = self.train_args
        logger.info(f"Training saving to {training_args.output_dir}")
        # 创建 TensorBoard 摘要写入器
        tensorboard_writer = SummaryWriter(training_args.logging_dir)

        # 创建 DataLoader 加载训练和验证数据集
        # train_dataloader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=False)
        # eval_dataloader = DataLoader(eval_dataset, batch_size=training_args.per_device_eval_batch_size)
        if isinstance(model, E2EModel):
            trainable_params = filter(lambda x: x.requires_grad, model.parameters())
        else:
            # trainable_params = get_trainable_parameters(model)
            trainable_params = filter(lambda x: x.requires_grad, model.parameters())

        # 定义优化器和学习率调度器
        optimizer = torch.optim.AdamW(trainable_params, lr=training_args.learning_rate)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, end_factor=0.01, total_iters=10)

        # 开始训练
        forward_step = 0
        backward_step = 0
        # best_eval_loss = float('inf')
        dataset_train = DataLoader(dataset_train, batch_size=self.train_args.per_device_train_batch_size,
                                   collate_fn=dataset_batch_processor
                                   )
        eval_loss = self.validate_model(model, dataset_validate)
        tensorboard_writer.add_scalar('train/eval_loss', eval_loss, backward_step)
        tensorboard_writer.add_scalar('train/epoch', 0, backward_step)
        for epoch in range(training_args.num_train_epochs):
            for batch in tqdm(dataset_train, desc=f"Epoch {epoch + 1}/{training_args.num_train_epochs}", leave=False):
                # 将数据传递给设备
                # inputs = batch['input_ids'].to(training_args.device)
                # labels = batch['labels'].to(training_args.device)

                # 前向传播
                model.train()
                outputs = model(**self.__class__.data_collator(batch, model.device,
                                                               batch_size=self.train_args.per_device_train_batch_size))
                forward_step += 1
                # if not forward_step % self.train_args.gradient_accumulation_steps == 0:
                #     continue

                loss = outputs.loss

                # 反向传播
                loss.backward(retain_graph=IS_TEST)
                optimizer.step()
                optimizer.zero_grad()

                backward_step += 1

                # 记录训练损失
                tensorboard_writer.add_scalar('train/loss', loss, backward_step)
                tensorboard_writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], backward_step)
                if backward_step % training_args.eval_steps == 0:
                    # 在验证集上评估模型
                    eval_loss = self.validate_model(model, dataset_validate)
                    tensorboard_writer.add_scalar('train/eval_loss', eval_loss, backward_step)
                    optimizer.zero_grad()

                if backward_step % training_args.save_steps == 0:
                    model.save_pretrained(os.path.join(training_args.output_dir, f'checkpoint-{backward_step}'))

            scheduler.step()

            tensorboard_writer.add_scalar('train/epoch', epoch + 1, backward_step)

            # 如果当前模型性能优于之前的最佳性能，则保存当前模型
            # if eval_loss < best_eval_loss:
            #     best_eval_loss = eval_loss
            # torch.save(model.state_dict(), f"{training_args.output_dir}/best_model.pt")

            # 更新学习率

        tensorboard_writer.close()

    # 定义模型评估函数


    def validate_model(self,model, validation_dataloader):
        model.eval()
        count, loss = 0, 0
        _validation_dataloader = DataLoader(validation_dataloader, batch_size=self.train_args.per_device_train_batch_size,
                                   collate_fn=dataset_batch_processor
                                   )
        with torch.no_grad():
            for example in tqdm(_validation_dataloader, desc="Validating",leave=False):
                output = model(**self.data_collator(example, model.device,batch_size=self.train_args.per_device_train_batch_size))
                if output.loss.item() != output.loss.item():
                    continue
                    # return example
                loss += output.loss.item()
                count += 1
            return loss / count

    @staticmethod
    def data_collator(data, device, batch_size=1):
        new_data = {}
        new_data["input_ids"] = torch.Tensor(data["input_ids"]).int().to(device)
        new_data["attention_mask"] = torch.Tensor(data["attention_mask"]).int().to(device)
        new_data["labels"] = torch.Tensor(data["labels"]).long().to(device)
        return new_data


def gen(model: Qwen2ForCausalLM, tokenizer: PreTrainedTokenizer, prompt: str = '你是谁',
        prompt_system: str = "现在你要扮演皇帝身边的女人--甄嬛"):

    start_time = time.time()

    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=50,
        use_cache=False
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    end_time=time.time()
    logger.info(f"Took: {end_time-start_time} s\nGenerated: {response}")
    return response


if __name__ == '__main__':
    set_seed()
    log_gpu_memory_usage()
    # m=check1()
    # sd=torch.load(os.path.join(model_path_tail,'model.safetensors'))
    p = DevPipeline()
    tokenizer = p._load_tokenizer()
    model_inputs = p.tokenize_input()
    log_gpu_memory_usage()
    p_lora = PipelineLoraFinetune(is_test=IS_TEST)
    dataset_train, dataset_validate = fine_tune_dataset()

    if not "raw":
        model = Qwen2ForCausalLM.from_pretrained(model_path,
                                                 # quantization_config=bnb_config,
                                                 **load_kwargs
                                                 )
        # model.load_state_dict(torch.load(model_path, map_location=lambda storage))
        log_gpu_memory_usage()

        if not 'lora':
            if 'train':
                model_lora = p_lora.get_lora_model(model)
                p_lora.train_model(model_lora, dataset_train, dataset_validate, is_trainer=False)
            if not 'inference':
                model_lora = PeftModelForCausalLM.from_pretrained(model,
                                                                  model_id=os.path.join(lora_path, 'checkpoint_150'),
                                                                  config=lora_config, is_trainable=False)
                # model_lora.save_pretrained()
                prompt = "你是谁？"
                messages = [
                    {"role": "system", "content": "现在你要扮演皇帝身边的女人--甄嬛"},
                    {"role": "user", "content": prompt}
                ]

                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                model_inputs = tokenizer([text], return_tensors="pt").to(model_lora.device)

                generated_ids = model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=512
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                print(response)
        if not 'inference':
            model_inputs.to(model.device)
            mid_results = model.forward(model_inputs.input_ids, output_hidden_states=True)

            generated_ids = model.generate(
                model_inputs.input_ids.to(model.device),
                max_new_tokens=20, use_cache=False
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            logger.debug(f"Generated: {response}")
            # input_embed=model.model.embed_tokens(model_inputs.input_ids)
            # for layer in model.model.layers:
            #     new_input=layer.forward(input_embed)
            #     logger.debug('break')
    if not 'dev accelerate':
        logger.info("use accelerate")
        if 'raw':
            logger.info('loading raw model')
            model_config = AutoConfig.from_pretrained(model_path)
            with init_empty_weights():
                model = Qwen2ForCausalLM.from_pretrained(model_path, config=model_config)
            log_gpu_memory_usage()
            model = load_checkpoint_and_dispatch(model, model_path, no_split_module_classes='Qwen2DecoderLayer',
                                                 **load_kwargs)
            log_gpu_memory_usage()
        if 'vfl':
            if not 'head':
                logger.info("load model head")
                model_config = AutoConfig.from_pretrained(model_path_head)
                with init_empty_weights():
                    model = Qwen2ModelHead.from_pretrained(model_path_head, config=model_config)
                log_gpu_memory_usage()
                model = load_checkpoint_and_dispatch(model, model_path_head,
                                                     no_split_module_classes='Qwen2DecoderLayer',
                                                     **load_kwargs)
                log_gpu_memory_usage()
            if 'tail':
                logger.info("load model tail")
                model_config = AutoConfig.from_pretrained(model_path_tail)
                with init_empty_weights():
                    model = Qwen2TailForCausalLM._from_config(config=model_config)
                    # model.tie_weights()
                    # d_map= infer_auto_device_map(model,max_memory=load_kwargs.get('max_memory'),no_split_module_classes='Qwen2DecoderLayer')
                log_gpu_memory_usage()
                # time.sleep(10)
                model = load_checkpoint_and_dispatch(model, model_path_tail,
                                                     no_split_module_classes='Qwen2DecoderLayer',
                                                     # skip_keys=['model.embed_tokens.weight','lm_head.weight'],
                                                     # device_map=d_map,
                                                     **load_kwargs
                                                     )

                log_gpu_memory_usage()

    if "dev VFL":
        models = load_vfl_model(split_idx=SPLIT_INDEX, is_from_raw=True, save_model=False)
        local_model, global_model = models[0], models[1]
        logger.info("finish load model")
        model_inputs.to(local_model.device)

        if not "dev forward":
            if not 'dev train':
                local_model.train()
                optimizer = torch.optim.Adam(local_model.parameters(), lr=0.03)
                d, dataset_validate = fine_tune_dataset()
                a = {k: torch.Tensor(v).int().reshape([1, -1]).to(local_model.device) for k, v in d[0].items()}
                intermediate = local_model.forward(output_hidden_states=True,
                                                   output_attentions=True,
                                                   use_cache=False, **a)
                intermediate.to(global_model.device)
                output = global_model.forward(inputs_embeds=intermediate.hidden_states,
                                              attention_mask=intermediate.attention_mask,
                                              past_key_values=intermediate.past_key_values,
                                              output_hidden_states=intermediate.output_hidden_states,
                                              position_ids=intermediate.position_ids, use_cache=False,
                                              labels=a['labels'].long())
                loss = output.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            intermediate = local_model.forward(model_inputs.input_ids, output_hidden_states=True,
                                               output_attentions=True,
                                               use_cache=False)
            intermediate = intermediate  # type: Qwen2DecoderLayerParam
            intermediate.to(global_model.device)
            output = global_model.forward(inputs_embeds=intermediate.hidden_states,
                                          attention_mask=intermediate.attention_mask,
                                          past_key_values=intermediate.past_key_values,
                                          output_hidden_states=intermediate.output_hidden_states,
                                          position_ids=intermediate.position_ids, use_cache=False)
            log_gpu_memory_usage()
        if "e2e":
            if train_lora := 'lora':
                for p in models[0].parameters():
                    p.requires_grad = False
                if 2 in models:
                    models[2]=p_lora.get_lora_model(models[2])
                models[1] = p_lora.get_lora_model(models[1])
            elif not 'lora inference':
                for i,m in models:
                    _model_path=os.path.join(lora_path,f'model_{i}')
                    if os.path.exists(_model_path):
                        models[i]=PeftModel.from_pretrained(m,model_id=_model_path)
                # global_model = PeftModelForCausalLM.from_pretrained(global_model,
                #                                                     model_id=os.path.join(lora_path,'checkpoint-manual'),
                #                                                     config=lora_config)

            logger.info(f"start Loading e2e model")
            log_gpu_memory_usage()
            with init_empty_weights():
                # e2e_model = E2EModel(local_model.config, local_model, global_model)
                e2e_model = E2EModel(models[0].config, models)
            logger.info(f"finish loading e2e model")
            log_gpu_memory_usage()
            if train_lora:
                if not 'use trainer':
                    p_lora.train_model(e2e_model)
                elif 'use class':
                    p_lora.train_model(e2e_model, dataset_train, dataset_validate, is_trainer=False)
                else:
                    dataset_train_new = DataLoader(dataset_train, batch_size=BATCH_SIZE,
                                                   collate_fn=dataset_batch_processor)

                    trainable_params = get_trainable_parameters(global_model)
                    optimizer = torch.optim.Adam(trainable_params, lr=0.0001)
                    for data in tqdm(dataset_train_new):
                        a = {k: torch.Tensor(v).int().reshape([BATCH_SIZE, -1]).to(local_model.device) for k, v in
                             data.items()}
                        a['labels'] = a['labels'].long()
                        output = e2e_model.forward(**a)
                        output.loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    e2e_model.eval()
            # p_lora.validate_model(e2e_model,dataset_validate)
            gen(e2e_model,tokenizer)
            ans = gen(e2e_model, tokenizer=tokenizer, prompt='You are a python programmer, what can you do?',
                      prompt_system='You are a helpful assistant.')
    pynvml.nvmlShutdown()
