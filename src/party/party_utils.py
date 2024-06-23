# from party.llm_party import Party
# from typing import Dict
#
# from loguru import logger
# from .LocalCommunication import LocalCommunication
# import torch
#
#
# class ProxyModel:
#     def __init__(self, party: Party, model_index: int):
#         self.party = party
#         self.model_index = model_index
#         self.model = self.party.models[model_index]
#         self.optimizer = self.party.optimizers[model_index]
#         self.lr_scheduler = self.party.lr_schedulers[model_index]
#         self.__input_tensor = None  # type: torch.Tensor
#         self.__output_tensor = None  # type: torch.Tensor
#
#     def forward(self, **kwargs):
#         if self.model_index == 0:
#             params = self._format_forward_kwargs(**kwargs)
#         else:
#             params = kwargs
#         for k in params:
#             if isinstance(params[k], list):
#                 # params[k] = torch.tensor(params[k]).to(self.args.device)
#                 params[k] = torch.tensor(params[k])
#             if isinstance(params[k], torch.Tensor):
#                 params[k] = params[k].to(self.model.device)
#         self.input_tensor = self._extract_input_tensor(params)
#         outputs = self.model.forward(**params)
#         self.output_tensor = self._extract_output_tensor(outputs)
#         logger.debug(f'finish model forward {self.model_index}')
#         logger.debug(self.output_tensor)
#         if self.model_index == 0:
#             if isinstance(self.party._communication, LocalCommunication):
#                 return outputs
#             outputs.to_json()
#             return outputs.prepare_for_forward()
#         else:
#             return outputs
#         # return {
#         #     "hidden_states": intermediate.hidden_states.tolist(),
#         #     "attention_mask": intermediate.attention_mask.tolist(),
#         #     "past_key_values": None,
#         #     "output_hidden_states": intermediate.output_hidden_states,
#         #     "position_ids": intermediate.position_ids.tolist()
#         # }
#
#     def backward(self, grad_output=None):
#         self.output_tensor.backward(grad_output=grad_output)
#         return self.input_tensor.grad
#
#     def optimizer_step(self):
#         self.optimizer.step()
#
#     def optimizer_zero_grad(self):
#         self.optimizer.zero_grad()
#
#     def lr_scheduler_step(self):
#         self.lr_scheduler.step()
#
#     def save_pretrained(self, **kwargs):
#         self.model.save_pretrained(**kwargs)
#
#     def __call__(self, *args, **kwargs):
#         return self.forward(**kwargs)
#
#     def _extract_input_tensor(self, param: Dict) -> torch.Tensor:
#         ans = param.get('inputs_embeds')
#         if ans is None:
#             ans = param.get('inputs_ids')
#         return ans
#
#     def _extract_output_tensor(self, param: Dict) -> torch.Tensor:
#         ans = param.get('hidden_states')
#         if ans is None:
#             ans = param.get('loss')
#         return ans
#
#     def _format_forward_kwargs(self, **kwargs):
#         if not kwargs:
#             tokenizer = self.party.args.tokenizer
#             prompt = "You are a python programmer, what can you do?"
#             messages = [
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": prompt}
#             ]
#             text = tokenizer.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=True
#             )
#             model_inputs = tokenizer([text], return_tensors="pt")
#             kwargs.update({'input_ids': model_inputs.input_ids,
#                            'output_hidden_states': True})
#             logger.debug(f"default inference, kwargs.keys: {kwargs.keys()}")
#         base_dict = {'input_ids': None,
#                      'attention_mask': None,
#                      'position_ids': None,
#                      'past_key_values': None,
#                      'inputs_embeds': None,
#                      'use_cache': False,
#                      'output_attentions': None,
#                      'output_hidden_states': True,
#                      'return_dict': None, }
#         for k in base_dict:
#             if k in kwargs:
#                 base_dict.update({k: kwargs.get(k)})
#         return base_dict
#
#     def eval(self):
#         self.model.eval()
#
#     def train(self, **kwargs):
#         self.model.train(**kwargs)
#
#     @property
#     def device(self):
#         return self.model.device
#
#     @property
#     def input_tensor(self):
#         return self.__input_tensor
#
#     @input_tensor.setter
#     def input_tensor(self, input_tensor: torch.Tensor):
#         self.__input_tensor = input_tensor
#
#     @property
#     def output_tensor(self):
#         return self.__output_tensor
#
#     @output_tensor.setter
#     def output_tensor(self, output_tensor):
#         self.__output_tensor = output_tensor
#
#     @property
#     def is_continue_backward(self):
#         return self.input_tensor.requires_grad
