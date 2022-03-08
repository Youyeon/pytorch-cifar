'''Train CIFAR10 with PyTorch.'''
from functools import total_ordering
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_loss = 100000  # best test loss
patience = 10
_patience = 0 # to check
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# train-valid split
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
_trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=_transform_train)
trainset = torch.utils.data.ConcatDataset([trainset, _trainset])


split_ = 0.2
shuffle = True
random_seed = 111
dataset_size = trainset.__len__()#len(trainset)
indices = list(range(dataset_size))
split = int(np.floor(split_ * dataset_size))
if shuffle :
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]
trainsampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
validsampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, sampler=trainsampler, num_workers=2)
validloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, sampler=validsampler, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = ResNet34()
net = ResNet50()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    print('==> Using ', device)
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

summary(net, input_size=(3,32,32))

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    acc = checkpoint['acc']
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.6f | Acc: %.4f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_loss
    global _patience
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(validloader), 'Loss: %.6f | Acc: %.4f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    loss = 100.*test_loss/total
    acc = 100.*correct/total
    if loss < best_loss:
        _patience = 0
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'loss': loss,
            'epoch': epoch,
        }
        if not os.path.isdir('./saved/checkpoint'):
            os.mkdir('./saved/checkpoint')
        torch.save(state, './saved/checkpoint/ckpt.pth') # model
        best_loss = loss
    else:
        _patience += 1

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    if (patience==_patience):
        print("==> epoch ", epoch, " Training END" )
        break
    scheduler.step()


os.system("cp ./saved/checkpoint/ckpt.pth ./saved/ResNet50.pth")