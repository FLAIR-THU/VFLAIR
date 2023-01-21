import pickle
import torch
import matplotlib.pyplot as plt

vfl_info = pickle.load(open('./vfl_info.pkl','rb'))
self_data = [vfl_info['data'][0][0],vfl_info['data'][1][0]]
self_label = vfl_info['label']
data = torch.cat(self_data,dim=2)

fig = plt.figure(figsize=(10,10))
for i in range(1,26):
    plt.subplot(5,5,i)
    plt.imshow(data[i][0].cpu(), cmap='gray')
    plt.subplots_adjust(wspace=0.15,hspace=0.15)
    plt.axis('off')
    plt.title(torch.argmax(self_label[i]).item())
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
# plt.savefig('mei_first100.png')
plt.savefig('mei_last100.png')