'''
    accuracy: 91%
'''
import math
from visdom import Visdom
import torch
import numpy as np
from torch.nn import functional as F  # Function-style
from torchvision import datasets, transforms
print(torch.__version__)
print(torch.cuda.is_available())


# viz = Visdom()
# viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
batch_size = 128
learning_rate = 0.002
epochs = 50
device = torch.device('cuda:0')


train_db = datasets.CIFAR10('../data', train=True, download=True,
                   transform=transforms.Compose([
                        # transforms.Resize([32, 32]),
                        # transforms.RandomHorizontalFlip(),
                        # transforms.RandomVerticalFlip(),
                        # transforms.RandomRotation(15),  # -15 ~ 15度旋转
                        # transforms.RandomRotation([90, 180, 270]),  # 随机挑选一个角度
                        transforms.RandomCrop([32, 32], padding=4),  # 先四周填充0，再把图像随机剪裁成32*32
                        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # feature scaling 均值 方差
                   ]))
test_db = datasets.CIFAR10('../data', train=False, download=True,
                   transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                   ]))

print('train:', len(train_db), 'test:', len(test_db))
train_db, val_db = torch.utils.data.random_split(train_db, [45000, 5000])  # 手动划分val
print('divide train to train and val    train_db:', len(train_db), 'val_db:', len(val_db))
train_loader = torch.utils.data.DataLoader(train_db, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_db, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_db, batch_size=batch_size, shuffle=True)

print("train:  batch_num {}  total_samples_num {}".format(len(train_loader), len(train_loader.dataset)))
print("validation:  batch_num {}  total_samples_num {}".format(len(val_loader), len(val_loader.dataset)))








class Reshape(torch.nn.Module):
    def __init__(self, c, h, w):
        super(Reshape, self).__init__()
        self.c = c
        self.h = h
        self.w = w

    def forward(self, x):  # x is input
        return x.view(x.size(0), self.c, self.h, self.w)


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):  # x is input
        return x.view(x.size(0), -1)


class ResBlk(torch.nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        '''
            b*ch_in*h*w  =>
            b*ch_out*h*w (可能下采样)  =>
            b*ch_out*h*w
        '''
        super(ResBlk, self).__init__()
        self.left = torch.nn.Sequential(
            torch.nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(ch_out),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(ch_out)
        )
        self.shortcut = torch.nn.Sequential()
        # short cut
        if stride != 1 or ch_out != ch_in:  # change input channels to ch_out using 1*1 Conv  (diff from original version)
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        '''
            :param x: input
            :return:
        '''
        return F.relu(self.left(x)+self.shortcut(x), inplace=True)


class ResNet(torch.nn.Module):  # ResNet18   1 + 8*2 + 1

    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.inchannel = 64  # tmp help variable
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )
        self.model = torch.nn.Sequential(
            self.conv1,
            self.make_layer(ResBlk, 64, 2, stride=1),  # 64 32 32
            self.make_layer(ResBlk, 128, 2, stride=2),  # 128 16 16
            self.make_layer(ResBlk, 256, 2, stride=2),  # 256 8 8
            self.make_layer(ResBlk, 512, 2, stride=2),  # 512 4 4
            torch.nn.AvgPool2d(kernel_size=4),  # 512 1 1
            Flatten(),  # 512
            torch.nn.Linear(512, num_classes)  # 10
        )

    def make_layer(self, block, ch_out, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)  # strides=[1,1] or [2,1]
        layers = []
        for one_stride in strides:
            layers.append(block(self.inchannel, ch_out, one_stride))
            self.inchannel = ch_out
        return torch.nn.Sequential(*layers)

    def forward(self, x):  # 重写
        x = self.model(x)  # 会触发__call__  然后调用forward
        return x



def train():
    net = ResNet(10).to(device)  # MLP().to(device) 直接这样也行 因为是引用
    # SGD可以手动添加momentum
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  #
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=10, threshold=0.000001)
    # regularization weight_decay=0.001
    # momentum       momentum=0.78
    criteon = torch.nn.CrossEntropyLoss().to(device)
    for epoch in range(epochs):
        net.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):

            data = data.view(-1, 3, 32, 32)
            data, target = data.to(device), target.cuda()  # .cuda()是老版本 不再推荐 这里只是师范有这么一个方法

            logits = net(data)
            ''' L1 regularization '''
            regularization_loss = 0
            for param in net.parameters():
                regularization_loss += torch.sum(torch.abs(param))
            ''''''

            loss = criteon(logits, target) + 0.000001*regularization_loss

            optimizer.zero_grad()
            loss.backward()
            # scheduler.step(loss)
            optimizer.step()

            if(batch_idx % 50 == 0):  # 每50*batchsize 个sample打印一次
                print('Train Epoch: {} [{}/{}]\t  Average Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data),
                    len(train_loader.dataset), loss.item()  # loss.item()/len(data) ?
                ))
                # viz.line([loss.item()], [epoch*len(train_loader)+batch_idx], win='train_loss', update='append')
                # viz.images(data[0]*0.3081+0.1307, win='image')
                # print(data[0, 234:245])
                # viz.images(torch.randn(3, 200, 200), win='image')


        net.model.eval() # 关闭dropout
        # 在val集上测试
        val_loss = 0
        correct = 0
        for data, target in val_loader:
            data = data.view(-1, 3, 32, 32)
            data, target = data.to(device), target.to(device)
            logits = net(data)  # shape: num*10
            val_loss += criteon(logits, target).item()
            pred = logits.data.max(dim=1)[1]  #  shape:1*num
            # print(type(logits), type(logits.data), logits.data.shape, pred.shape)
            correct += pred.eq(target.data).sum()  # 有多少相等的
        val_loss /= len(val_loader)  #
        print('\nVal set: Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)
        ))

    return net


def test(net):
    net.model.eval()
    criteon = torch.nn.CrossEntropyLoss().to(device)

    # 在整个测试集上
    test_loss = 0
    correct = 0
    # print(len(test_loader.dataset))
    for data, target in test_loader:
        data = data.view(-1, 3, 32, 32)
        data, target = data.to(device), target.to(device)
        logits = net(data)  # shape: num*10
        test_loss += criteon(logits, target).item()
        pred = logits.data.max(dim=1)[1]  # shape:1*num
        # print(type(logits), type(logits.data), logits.data.shape, pred.shape)
        correct += pred.eq(target.data).sum()  # 有多少相等的

    test_loss /= len(test_loader)
    print('\nTest set: Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))


if __name__ == '__main__':
    net = train()
    test(net)