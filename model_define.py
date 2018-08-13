import torch
import torch.nn as nn
from linearcoding import LinearCoding
from torch.autograd import Variable


class linear_coding(nn.Module):
    def __init__(self, basis_num, embedding_dim):
        super(linear_coding, self).__init__()
        self.basis_num = basis_num
        self.embedding_dim = embedding_dim
        self.register_buffer('basis', torch.zeros(self.basis_num, self.embedding_dim))

    def reset_basis(self, basis):
        if torch.is_tensor(basis):
            self.basis.copy_(basis)
        else:
            self.basis.copy_(basis.data)

    def forward(self, x):
        batch_size = x.size(0)
        sparsity = x.size(1)
        assert sparsity <= self.basis_num
        out = Variable(torch.zeros(batch_size, self.basis_num))
        if self.training:
            index = torch.LongTensor(batch_size).random_(self.basis_num)
        else:
            index = torch.LongTensor(batch_size).zero_()
        if x.is_cuda:
            index = index.cuda()
        basis_select = self.basis[index]
        basis_expand = self.basis.view(1, self.basis_num, self.embedding_dim).expand(batch_size, self.basis_num, self.embedding_dim)
        select_expand = basis_select.view(batch_size, 1, self.embedding_dim).expand(batch_size, self.basis_num, self.embedding_dim)
        distance = torch.norm(basis_expand-select_expand, 2, 2) # batch_size x basis_num
        _, indices = torch.sort(distance)
        indices = Variable(indices[:, 0:sparsity]) # batch_size x sparsity
        if x.is_cuda:
            out = out.cuda()
            indices = indices.cuda()
        out = out.scatter_(1, indices, x)
        out = torch.mm(out, Variable(self.basis))
        return out.view(out.size(0), out.size(1), 1, 1)


class _netG(nn.Module):
    def __init__(self, basis_num, embedding_dim, nz, ngf, nc):
        super(_netG, self).__init__()
        self.basis_num = basis_num
        self.embedding_dim = embedding_dim
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.lcc = linear_coding(self.basis_num, self.embedding_dim)
        # DCGAN
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.embedding_dim, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 1),
            nn.ReLU(True),
            # state size. (ngf*1) x 32 x 32
            nn.ConvTranspose2d(self.ngf * 1, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (ngf) x 64 x 64
        )

    def reset_basis(self, basis):
        self.lcc.reset_basis(basis)

    def forward(self, input):
        output = self.lcc(input)
        output = self.main(output)
        return output


class _netD(nn.Module):
    def __init__(self, nc, ndf):
        super(_netD, self).__init__()
        self.nc = nc
        self.ndf = ndf
        # DCGAN
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(self.ndf * 8, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1)


class _decoder(nn.Module):
    def __init__(self, nc, ngf, embedding_dim):
        super(_decoder, self).__init__()
        self.nc = nc
        self.ngf = ngf
        self.embedding_dim = embedding_dim
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.embedding_dim, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf * 4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf * 2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 1),
            nn.ReLU(True),
            # state size. (ngf * 1) x 32 x 32
            nn.ConvTranspose2d(self.ngf * 1, self.nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output


class _encoder(nn.Module):
    def __init__(self, nc, ndf, embedding_dim):
        super(_encoder, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.embedding_dim = embedding_dim
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf * 2) x 32 x 32
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf * 4) x 16 x 16
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf * 4) x 8 x 8
            nn.Conv2d(self.ndf * 8, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf * 8) x 4 x 4
            nn.Conv2d(self.ndf * 8, self.embedding_dim, 4, 1, 0, bias=False),
            # state size. (embedding_dim) x 1 x 1
        )

    def forward(self, input):
        output = self.main(input)
        # output = output.view(-1, self.embedding_dim)
        return output