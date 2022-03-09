import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torchsummary import summary
from models import *

import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

path = "./saved/parms/resnet18/"

net = ResNet18()
net = net.to(device)
net.eval()

if device == 'cuda':
    print('==> Using ', device)
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint = torch.load('./saved/ResNet18.pth')
_dict = checkpoint['net']
net.load_state_dict(_dict)
acc = checkpoint['acc']
best_loss = checkpoint['loss']
start_epoch = checkpoint['epoch']

os.makedirs(path, exist_ok=True)

for key, val in _dict.items():
    if 'num_batches' in key:
        continue
    w = val.cpu().detach().numpy()
    w_reshaped = w.reshape(w.shape[0],-1)
    w_reshaped = w_reshaped.flatten()
 
    with open(path+key+".dat", "wb") as fp:
        w_reshaped.tofile(fp, format='f4')
    fp.close()

# Layer list (weight O)
f_ = open(path+"names.txt", "w")

name_list = []
for name, _ in net.named_parameters():
    name = name.replace(".weight", "")
    name = name.replace(".bias", "")
    if name not in name_list:
        name_list.append(name)
    else:
        continue
    f_.write(name)
    f_.write("\n")

f_.close()

#model structure
f = open(path+"meta.txt", "w")
print(net, file=f)
f.close()