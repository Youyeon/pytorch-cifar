import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torchsummary import summary
from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

net = ResNet50()
net = net.to(device)

if device == 'cuda':
    print('==> Using ', device)
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint = torch.load('./saved/ResNet50.pth')
net.load_state_dict(checkpoint['net'])
acc = checkpoint['acc']
best_loss = checkpoint['loss']
start_epoch = checkpoint['epoch']

summary(net, input_size=(3,32,32))

def eval(testloader):
    net.eval()
    total = 0
    correct = 0
    test_acc = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    print("Accuracy: ", acc, "(", correct, "/", total, ")")

eval(testloader)
