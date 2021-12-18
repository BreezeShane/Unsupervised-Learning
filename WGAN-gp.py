import os
import torch
import argparse
from torch import nn, optim, autograd
import numpy as np
import visdom
import random
from torch.nn import functional as F
from matplotlib import pyplot as plt


class Generator(nn.Module):
    def __init__(self, hide_layer=2):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(hide_layer, opt.h_dim),
            nn.ReLU(True),
            nn.Linear(opt.h_dim, opt.h_dim),
            nn.ReLU(True),
            nn.Linear(opt.h_dim, opt.h_dim),
            nn.ReLU(True),
            nn.Linear(opt.h_dim, 2),
            # 最后这个2是设定输出的维度
        )

    def forward(self, z):
        output = self.net(z)
        return output


class Discriminator(nn.Module):
    def __init__(self, hide_layer=2):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(hide_layer, opt.h_dim),
            nn.ReLU(True),
            nn.Linear(opt.h_dim, opt.h_dim),
            nn.ReLU(True),
            nn.Linear(opt.h_dim, opt.h_dim),
            nn.ReLU(True),
            nn.Linear(opt.h_dim, 1),
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

        for i in range(opt.batchsz):
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
        if opt.use_cuda:
            points = torch.Tensor(points).cuda() # [16384, 2]
        else:
            points = torch.Tensor(points) # [16384, 2]
        disc_map = D(points).cpu().numpy() # [16384]
    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    cs = plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
    plt.clabel(cs, inline=1, fontsize=10)
    # plt.colorbar()


    # draw samples
    with torch.no_grad():
        if opt.use_cuda:
            z = torch.randn(opt.batchsz, 2).cuda() # [b, 2]
        else:
            z = torch.randn(opt.batchsz, 2)
        samples = G(z).cpu().numpy() # [b, 2]
    plt.scatter(x_r[:, 0], x_r[:, 1], c='orange', marker='.')
    plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')

    viz.matplot(plt, win='contour', opts=dict(title='p(x):%d' % epoch))


def weights_init(m):
    if isinstance(m, nn.Linear):
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)


def gradient_penalty(D, x_r, x_f):
    if opt.use_cuda:
        t = torch.rand(opt.batchsz, 1).cuda()
    else:
        t = torch.rand(opt.batchsz, 1)
    t = t.expand_as(x_r)
    mid = t * x_r + (1 - t) * x_f
    mid.requires_grad_()

    pred = D(mid)
    grads = autograd.grad(outputs=pred,
                          inputs=mid,
                          grad_outputs=torch.ones_like(pred),
                          create_graph=True,
                          retain_graph=True,
                          only_inputs=True)[0]
    gp = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()
    return gp


def save_module(D, G):
    if not os.path.exists("WGAN_gp_Model"):
        os.mkdir("WGAN_gp_Model")
    D_path = os.path.join("WGAN_gp_Model", "Discriminator.pt")
    G_path = os.path.join("WGAN_gp_Model", "Generator.pt")
    torch.save(D, D_path)
    torch.save(G, G_path)


def load_module():
    if os.path.exists("WGAN_gp_Model"):
        D_path = os.path.join("WGAN_gp_Model", "Discriminator.pt")
        G_path = os.path.join("WGAN_gp_Model", "Generator.pt")
        D = G = None
        if os.path.exists(D_path) and os.path.exists(G_path):
            D = torch.load(D_path)
            G = torch.load(G_path)
            return D, G
    return None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prefix_chars='-_')
    parser.add_argument("--use_cuda", action='store_true')
    parser.add_argument("--use_model", action='store_true')
    parser.add_argument("--h_dim", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--main_epoch", type=int, default=50000)
    parser.add_argument("--D_epoch", type=int, default=20)
    parser.add_argument("--D_lr", type=float, default=5e-6)
    parser.add_argument("--G_lr", type=float, default=5e-6)
    parser.add_argument("--const_lambda", type=float, default=0.2)
    parser.add_argument("--D_betas", type=tuple, default=(0.5, 0.9))
    parser.add_argument("--G_betas", type=tuple, default=(0.5, 0.9))
    opt = parser.parse_args()

    viz = visdom.Visdom()
    torch.manual_seed(23)
    np.random.seed(23)

    data_iter = data_generator()
    x = next(data_iter)
    D = G = None
    if opt.use_model:
        D, G = load_module()
    if D is None or G is None:
        if opt.use_cuda:
            G = Generator().cuda()
            D = Discriminator().cuda()
        else:
            G = Generator()
            D = Discriminator()
        G.apply(weights_init)
        D.apply(weights_init)
    optim_G = optim.Adam(G.parameters(), lr=opt.G_lr, betas=opt.G_betas)
    optim_D = optim.Adam(D.parameters(), lr=opt.D_lr, betas=opt.D_betas)

    print('batch:', next(data_iter).shape)
    viz.line([[0, 0]], [0], win='loss', opts=dict(title='loss', legend=['D', 'G']))
    loss_D = x_r = None

    for epoch in range(opt.main_epoch):
        for _ in range(opt.D_epoch):
            x_r = next(data_iter)
            if opt.use_cuda:
                x_r = torch.from_numpy(x_r).cuda()
            else:
                x_r = torch.from_numpy(x_r)
            pred_r = D(x_r)  # to maximize
            loss_r = -pred_r.mean()
            if opt.use_cuda:
                z = torch.randn(opt.batchsz, 2).cuda()
            else:
                z = torch.randn(opt.batchsz, 2)
            x_f = G(z).detach()
            pred_f = D(x_f)
            loss_f = pred_f.mean()

            gp = gradient_penalty(D, x_r, x_f.detach())

            loss_D = loss_r + loss_f + opt.const_lambda * gp

            optim_D.zero_grad()  # clear to zero
            loss_D.backward()
            optim_D.step()

        if opt.use_cuda:
            z = torch.randn(opt.batchsz, 2).cuda()
        else:
            z = torch.randn(opt.batchsz, 2)
        x_fake = G(z)
        pred_fake = D(x_fake)
        loss_G = -pred_fake.mean()

        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if epoch % 10 == 0:
            viz.line([[loss_D.item(), loss_G.item()]], [epoch], win='loss', update='append')
            generate_image(D, G, x_r, epoch)
            print(loss_D.item(), loss_G.item())

        if epoch % 100 == 0:
            save_module(D, G)
