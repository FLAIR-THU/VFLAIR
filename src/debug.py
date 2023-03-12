# import pickle
# import torch
# import matplotlib.pyplot as plt

# vfl_info = pickle.load(open('./vfl_info.pkl','rb'))
# self_data = [vfl_info['data'][0][0],vfl_info['data'][1][0]]
# self_label = vfl_info['label']
# data = torch.cat(self_data,dim=2)

# fig = plt.figure(figsize=(10,10))
# for i in range(1,26):
#     plt.subplot(5,5,i)
#     plt.imshow(data[i][0].cpu(), cmap='gray')
#     plt.subplots_adjust(wspace=0.15,hspace=0.15)
#     plt.axis('off')
#     plt.title(torch.argmax(self_label[i]).item())
#     plt.xticks([])
#     plt.yticks([])
# plt.tight_layout()
# # plt.savefig('mei_first100.png')
# plt.savefig('mei_last100.png')

# import sys
# def main():
#     return 23.6, 45.8
# # if __name__ == "__main__":
# #     main()
# #     sys.exit(0)

# main()


import torch
import torch.nn as nn
import torch.nn.functional as F # nn.functional.py中存放激活函数等的实现
 
@torch.no_grad()
def init_weights(m):
    # print("xxxx:", m)
    if type(m) == nn.Linear:
         m.weight.fill_(1.0)
        #  print("yyyy:", m.weight)
 
class MyModel(nn.Module):
    def __init__(self):
        # 在实现自己的__init__函数时,为了正确初始化自定义的神经网络模块,一定要先调用super().__init__
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5) # submodule(child module)
        self.conv2 = nn.Conv2d(20, 20, 5)
        self.add_module("conv3", nn.Conv2d(10, 40, 5)) # 添加一个submodule到当前module,等价于self.conv3 = nn.Conv2d(10, 40, 5)
        self.register_buffer("buffer", torch.randn([2,3])) # 给module添加一个presistent(持久的) buffer
        self.param1 = nn.Parameter(torch.rand([1])) # module参数的tensor
        self.register_parameter("param2", nn.Parameter(torch.rand([1]))) # 向module添加参数
 
        # nn.Sequential: 顺序容器,module将按照它们在构造函数中传递的顺序添加,它允许将整个容器视为单个module
        self.feature = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        self.feature.apply(init_weights) # 将fn递归应用于每个submodule,典型用途为初始化模型参数
        self.feature.to(torch.double) # 将参数数据类型转换为double
        cpu = torch.device("cpu")
        self.feature.to(cpu) # 将参数数据转换到cpu设备上
 
    def forward(self, x):
       x = F.relu(self.conv1(x))
       return F.relu(self.conv2(x))
 
model = MyModel()
print("## Model:", model)
print(type(model))
print(str(type(model)).split('.')[-1].split('\'')[-2])
# print(str(type(model)).split('\''))
 
model.cpu() # 将所有模型参数和buffers移动到CPU上
model.float() # 将所有浮点参数和buffers转换为float数据类型
model.zero_grad() # 将所有模型参数的梯度设置为零
 
# # state_dict:返回一个字典,保存着module的所有状态,参数和persistent buffers都会包含在字典中,字典的key就是参数和buffer的names
# print("## state_dict:", model.state_dict().keys())
 
# for name, parameters in model.named_parameters(): # 返回module的参数(weight and bias)的迭代器,产生(yield)参数的名称以及参数本身
#     print(f"## named_parameters: name: {name}; parameters size: {parameters.size()}")
 
# for name, buffers in model.named_buffers(): # 返回module的buffers的迭代器,产生(yield)buffer的名称以及buffer本身
#     print(f"## named_buffers: name: {name}; buffers size: {buffers.size()}")
 
# # 注:children和modules中重复的module只被返回一次
# for children in model.children(): # 返回当前module的child module(submodule)的迭代器
#     print("## children:", children)
 
# for name, children in model.named_children(): # 返回直接submodule的迭代器,产生(yield) submodule的名称以及submodule本身
#     print(f"## named_children: name: {name}; children: {children}")
 
# for modules in model.modules(): # 返回当前模型所有module的迭代器,注意与children的区别
#     print("## modules:", modules)
 
# for name, modules in model.named_modules(): # 返回网络中所有modules的迭代器,产生(yield)module的名称以及module本身,注意与named_children的区别
#     print(f"## named_modules: name: {name}; module: {modules}")
 
# model.train() # 将module设置为训练模式
# model.eval() # 将module设置为评估模式
 
# print("test finish")