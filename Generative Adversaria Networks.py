import torch
from torch import nn, optim, autograd
import numpy as np
import visdom
import random
from torch.nn import functional as F
from matplotlib import pyplot as plt

h_dim = 400
batchsz = 512
viz = visdom.Visdom()
USE_CUDA = True


class Generator(nn.Module):
    def __init__(self, hide_layer=2):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(hide_layer, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 2),
            # 最后这个2是设定输出的维度
        )

    def forward(self, z):
        output = self.net(z)
        return output


class Discriminator(nn.Module):

    def __init__(self, hide_layer=2):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(hide_layer, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.net(x)
        return output.view(-1)


def data_generator():
    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2)),
    ]
    centers = [(scale * x, scale * y) for x, y in centers]

    while True:
        dataset = []

        for i in range(batchsz):
            point = np.random.randn(2) * 0.02
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)

        dataset = np.array(dataset).astype(np.float32)
        dataset /= 1.414
        yield dataset


def generate_image(D, G, x_r, epoch):
    N_POINTS = 128
    RANGE = 3
    plt.clf()
    x_r = x_r.cpu()


    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))
    # (16384, 2)
    # print('p:', points.shape)

    # draw contour
    with torch.no_grad():
        if USE_CUDA:
            points = torch.Tensor(points).cuda() # [16384, 2]
        else:
            points = torch.Tensor(points)
        disc_map = D(points).cpu().numpy() # [16384]
    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    cs = plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
    plt.clabel(cs, inline=1, fontsize=10)
    # plt.colorbar()


    # draw samples
    with torch.no_grad():
        if USE_CUDA:
            z = torch.randn(batchsz, 2).cuda() # [b, 2]
        else:
            z = torch.randn(batchsz, 2)
        samples = G(z).cpu().numpy() # [b, 2]
    plt.scatter(x_r[:, 0], x_r[:, 1], c='orange', marker='.')
    plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')

    viz.matplot(plt, win='contour', opts=dict(title='p(x):%d' % epoch))


def main():
    torch.manual_seed(23)
    np.random.seed(23)

    data_iter = data_generator()
    x = next(data_iter)
    if USE_CUDA:
        G = Generator().cuda()
    else:
        G = Generator()
    if USE_CUDA:
        D = Discriminator().cuda()
    else:
        D = Discriminator()
    optim_G = optim.Adam(G.parameters(), lr=5e-6, betas=(0.5, 0.9))
    optim_D = optim.Adam(D.parameters(), lr=5e-6, betas=(0.5, 0.9))

    print('batch:', next(data_iter).shape)

    viz.line([[0, 0]], [0], win='loss', opts=dict(title='loss', legend=['D', 'G']))

    for epoch in range(1500):

        for _ in range(5):
            x_r = next(data_iter)
            if USE_CUDA:
                x_r = torch.from_numpy(x_r).cuda()
            else:
                x_r = torch.from_numpy(x_r)
            pred_r = D(x_r)  # to maximize
            loss_r = -pred_r.mean()

            if USE_CUDA:
                z = torch.randn(batchsz, 2).cuda()
            else:
                z = torch.randn(batchsz, 2)
            x_f = G(z).detach()
            pred_f = D(x_f)
            loss_f = pred_f.mean()

            loss_D = loss_r + loss_f

            optim_D.zero_grad()  # clear to zero
            loss_D.backward()
            optim_D.step()

        if USE_CUDA:
            z = torch.randn(batchsz, 2).cuda()
        else:
            z = torch.randn(batchsz, 2)
        x_fake = G(z)
        pred_fake = D(x_fake)
        loss_G = -pred_fake.mean()

        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if epoch % 5 == 0:
            viz.line([[loss_D.item(), loss_G.item()]], [epoch], win='loss', update='append')
            generate_image(D, G, x_r, epoch)
            print(loss_D.item(), loss_G.item())


if __name__ == '__main__':
    main()