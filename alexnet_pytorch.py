import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.cuda as cuda
import torch.optim as optim
import torch.autograd as autograd

# local response normalization
class LocalResponseNormalization(nn.Module):
    def __init__(self, size=5, alpha=1e-4, beta=0.75, k=2):
        super(LocalResponseNormalization, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.average_3d = nn.AvgPool3d(kernel_size=(size, 1, 1),
                                        stride=1,
                                        padding=((size-1)//2, 0, 0))
        self.average_2d = nn.AvgPool2d(kernel_size=size,
                                        stride=1,
                                        padding=(size-1)//2)          
        
    def forward(self, a):
        r"""
        Applies local response normalization over an input signal composed of
        several input planes, where channels occupy the second dimension.
        """
        dim = a.dim()
        # occupy the second dimension
        if(dim == 3):
            div = a.pow(2).unsqueeze(1)
            div = self.average_2d(div).squeeze(1)
        else:
            div = a.pow(2)
            div = self.average_3d(div)
        div = div.mul(self.alpha).add(self.k).pow(self.beta)
        b = a.div(div) 
        return b
    
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LocalResponseNormalization()
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LocalResponseNormalization()
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
            )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
            )
        self.layer6 = nn.Sequential(
            nn.Linear(in_features=6*6*256, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
            )
        self.layer7 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
            )
        self.layer8 = nn.Linear(in_features=4096, out_features=num_classes)
        
    def forward(self, x):
        # print(x.size())
        x = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        # print(x.size())
        # change view
        x = x.view(-1, 6*6*256)
        # print(x.size())
        x = self.layer8(self.layer7(self.layer6(x)))
        #   print(x.size())
        return x

# define dataset paper version
transform_paper = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])
dataset = datasets.CIFAR10(root="datasets/", train=True, transform=transform_paper, download=True)
dataset.num_classes=10
dataset.name = "cifar10"
train_loader = data.DataLoader(dataset, batch_size=128, shuffle=True)


def weight_init(m):
    if(isinstance(m, nn.Conv2d)):
        # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # import math
        # m.weight.data.normal_(0, math.sqrt(2. / n))
        # paper version below 
        m.weight.data.normal_(0, 0.01)
    elif(isinstance(m, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif(isinstance(m, nn.Linear)):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.fill_(1)
        
model = AlexNet(dataset.num_classes)
model_info = 'use customized model with LRN'
print(model_info)
model.apply(weight_init)

use_gpu = cuda.is_available()
if(use_gpu):
    model = model.cuda()
    print('USE GPU')
else:
    print('USE CPU')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if(use_gpu):
            data, target = data.cuda(), target.cuda()
        data, target = autograd.Variable(data), autograd.Variable(target)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if(batch_idx%10 == 0):
            print("Train Epoch:{} [{}/{}  ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100.*batch_idx/len(train_loader.dataset), loss.data[0]))
    
for epoch in range(3):
    train(epoch)
print("Training Finished!")