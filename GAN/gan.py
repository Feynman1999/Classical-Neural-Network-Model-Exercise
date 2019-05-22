import torch
from torch import nn, optim, autograd
import numpy as np
import visdom
import random


h_dim = 400
batchsize = 512
viz = visdom.Visdom()


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            # z: [ batch, 2] => [batch, 2]
            nn.Linear(2, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 2),
        )

    def forward(self, z):
        output = self.net(z)
        return output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.net(x)
        return output.view(-1)  # 2 -> 1



def make_data():
    '''
    8-gaussian mixture models
    :return:
    '''
    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1./np.sqrt(2)),
        (1. / np.sqrt(2), -1./np.sqrt(2)),
        (-1. / np.sqrt(2), 1./np.sqrt(2)),
        (-1. / np.sqrt(2), -1./np.sqrt(2)),
    ]
    centers = [(scale*x, scale*y) for x, y in centers]

    while True:
        dataset = []

        for i in range(batchsize):
            point = np.random.randn(2) * 0.02
            center = random.choice(centers)
            # N(0, 0.02) + center
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)

        dataset = np.array(dataset).astype(np.float32)
        dataset /= 1.414
        yield dataset  # 每一次到这记录并返回


def gradient_penalty(D, xr, xf):
    '''

    :param D:
    :param xr: [b,2]
    :param xf: [b,2]
    :return:
    '''
    t = torch.rand(batchsize, 1).cuda()
    t = t.expand_as(xr)
    # interpolation
    mid = t * xr + (1-t) *xf
    # set require gradient
    pred = D(mid)
    grads = autograd.grad(outputs=pred, inputs=mid,
                          grad_outputs=torch.ones_like(pred),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]

    gp = torch.pow(grads.norm(2, dim=1)-1, 2).mean()

    return gp


def main():

    torch.manual_seed(23)
    np.random.seed(23)

    data_iter = make_data()
    x = next(data_iter)
    # [b,2]
    # print(x.shape)

    G = Generator().cuda()
    D = Discriminator().cuda()
    #print(G)
    #print(D)
    optim_G = optim.Adam(G.parameters(), lr=5e-4, betas=(0.5, 0.9))
    optim_D = optim.Adam(D.parameters(), lr=5e-4, betas=(0.5, 0.9))

    for epoch in range(50000):
        # 1. train D
        for _ in range(5):
            # train on real data
            xr = next(data_iter)
            xr = torch.from_numpy(xr).cuda()
            # [b, 2] => [b]
            predr = D(xr)
            lossr = -predr.mean()

            # train on fake data
            z = torch.randn(batchsize, 2).cuda()
            xf = G(z).detach()  # don't train G
            predf = D(xf)
            lossf = predf.mean()

            # gradient penalty
            gp = gradient_penalty(D, xr, xf.detach())

            # Loss
            loss_D = lossr + lossf + 0.2 * gp

            # optimize
            optim_D.zero_grad()  # 清空  之前在train G时会留下
            loss_D.backward()
            optim_D.step()


        # 2. train G
        z = torch.randn(batchsize, 2).cuda()
        xf = G(z)
        predf = D(xf)
        # max predf
        loss_G = -predf.mean()

        # optimize
        optim_G.zero_grad()
        loss_G.backward()  # 还会计算D的
        optim_G.step()  #只更新G的

        if epoch % 100 ==0:
            print(loss_D.item(), loss_G.item())


if __name__ == '__main__':
    main()