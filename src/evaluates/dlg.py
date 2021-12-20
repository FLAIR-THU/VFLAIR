import os
import torch
import csv
import json
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import grad
import torchvision
from attacks.utils import *
import attacks.dlg_config as args
import matplotlib.pyplot as plt
from torchvision import models, datasets, transforms

torch.manual_seed(1234)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, 10)
        )
        
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out
        
def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)

class DLGAttacker(object): 
  #The attacker object 
  def __init__(self, args):
    #self.num_iters = args.num_iters
    self.data_size = args.data_size
    self.label_size = args.label_size
    self.verbose = args.verbose
    self.save_path = args.save_path
    self.save_per_iter = args.save_per_iter
    self.batch_size = args.batch_size
    self.device = args.device
    self.tt = transforms.ToPILImage()
  
  def train(self, model, origin_grad, criterion, num_iters=300):
    dummy_data = torch.randn(args.data_size).to(self.device).requires_grad_(True)
    dummy_label =  torch.randn([1,10]).to(self.device).requires_grad_(True)
    #model = model.to(self.device)

    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    history = []
    for iters in range(num_iters):
        def closure():
            optimizer.zero_grad()

            dummy_pred = model(dummy_data) 
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            #print(dummy_label)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
            
            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, origin_grad): 
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            
            return grad_diff
        
        optimizer.step(closure)
        if iters % 10 == 0: 
            current_loss = closure()
            print(iters, "%.4f" % current_loss.item())
            history.append(self.tt(dummy_data[0].cpu()))

            if self.verbose == 1:
                self.save_dummy(dummy_data, dummy_label, iters)
     
    self.dummy_data = dummy_data.detach().clone()
    self.dummy_label = dummy_label.detach().clone()

    fig = plt.figure(figsize=(12, 8))
    for i in range(30):
        plt.subplot(3, 10, i + 1)
        plt.imshow(history[i])
        plt.title("iter=%d" % (i * 10))
        plt.axis('off')
    plt.savefig('./visulization.png')
    
    
  def get_original_grad(self, gt_data, gt_label, criterion, model):
      
        self.save_dummy(gt_data, gt_label, 'real')
        
        model = model.to(self.device)
        gt_data = gt_data.to(self.device)
        gt_label = gt_label.long().to(self.device)
        gt_onehot_label = label_to_onehot(gt_label)
      
        pred = net(gt_data)

        y = criterion(pred, gt_onehot_label)

        print(gt_onehot_label.shape)
        dy_dx = torch.autograd.grad(y, net.parameters())

        original_dy_dx = list((_.detach().clone() for _ in dy_dx))
      
        return original_dy_dx
  
  def get_original_grad_from_weights(self, parameter_new, parameter_old, num_rounds):
      return torch.div(parameter_old - parameter_new, num_rounds)
  
  
  def save_dummy(self, dummy_data, dummy_label, iters):
      save_path = os.path.join(self.save_path, 'saved_dummys/iters_{}/'.format(iters))
      if not os.path.exists(save_path):
        os.makedirs(save_path)
      
      torchvision.utils.save_image(dummy_data.detach().clone().cpu(), 
                                   os.path.join(save_path, 'batch_images.png'), 
                                   normalize=True,
                                   nrow=10)
      with open(os.path.join(save_path, 'batch_labels.txt'), 'w') as f:
          f.write(str(dummy_label.detach().clone().cpu()))
      
        
if __name__ == '__main__': 
    #from utils.model_utils import create_model
    from torchvision import models, datasets, transforms
    
    img_index = 1
    
    transform = transforms.Compose(
       [transforms.ToTensor()])

    dataset = datasets.CIFAR10(root='./data', download=True)
    
    net = LeNet().to(args.device) #create_model('cnn', 'Mnist-alpha0.1-ratio0.5', 'FedGen')[0]
    net.apply(weights_init)
    #print(model)
    #model_path = os.path.join("models", 'Mnist-alpha0.1-ratio0.5')
    #model = torch.load(os.path.join(model_path, "server" + ".pt"))
    
    gt_data = transform(dataset[img_index][0]).to(args.device)
    gt_data = gt_data.view(args.batch_size, *gt_data.size())
    gt_label = torch.Tensor([dataset[img_index][1]]).long().to(args.device)
    gt_label = gt_label.view(args.batch_size, )
    criterion = cross_entropy_for_onehot
    
    gld_attacker =  DLGAttacker(args)
    
    origin_grad = gld_attacker.get_original_grad(gt_data, gt_label, criterion, net)
    gld_attacker.train(net, origin_grad,  criterion)