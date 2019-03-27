from opt import opt
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
from model_define import _netG, _netD, _encoder, _decoder
import numpy as np


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.set_device(opt.gpu)
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


def createDataSet(opt, imageSize):
    if opt.dataset in ['oxford-192', 'celebA', 'lsun']:
        # folder dataset
        dataset = dset.ImageFolder(root=opt.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Scale(imageSize),
                                       transforms.CenterCrop(imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Scale(imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ])
        )
    return dataset


def dataparallel(model, ngpus, gpu0=0):
    if ngpus == 0:
        assert False, "only support gpu mode"
    gpu_list = list(range(gpu0, gpu0+ngpus))
    assert torch.cuda.device_count() >= gpu0+ngpus, "Invalid Number of GPUs"
    if isinstance(model, list):
        for i in range(len(model)):
            if ngpus >= 2:
                if not isinstance(model[i], nn.DataParallel):
                    model[i] = torch.nn.DataParallel(model[i], gpu_list).cuda()
            else:
                model[i] = model[i].cuda()
    else:
        if ngpus >= 2:
            if not isinstance(model, nn.DataParallel):
                model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:
            model = model.cuda()
    return model


class Trainer(object):
    def __init__(self, opt):
        self.opt = opt
        self.netG = _netG(opt.basis_num, opt.embedding_dim, opt.nz, opt.ngf, opt.nc)
        self.netD = _netD(opt.nc, opt.ndf)
        self.encoder = _encoder(opt.nc, opt.ndf, opt.embedding_dim)
        self.decoder = _decoder(opt.nc, opt.ngf, opt.embedding_dim)
        self.learnBasis = nn.Linear(self.opt.basis_num, self.opt.embedding_dim, bias=False)
        self.learnCoeff = nn.Linear(self.opt.basis_num, self.opt.batchSize_s2, bias=False)
        self.dataloader = torch.utils.data.DataLoader(createDataSet(self.opt, self.opt.imageSize), 
            batch_size=self.opt.batchSize_s1,
            shuffle=True, num_workers=int(self.opt.workers))

        self.criterion_bce = nn.BCELoss()
        self.criterion_l1 = nn.L1Loss(size_average=True)
        self.criterion_l2 = nn.MSELoss(size_average=True)

        # initialize the optimizers
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=opt.s3_lr, betas=(opt.beta1, opt.beta2))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt.s3_lr, betas=(opt.beta1, opt.beta2))
        self.optimizerEncoder = optim.Adam(self.encoder.parameters(), lr=opt.s1_lr, betas=(opt.beta1, opt.beta2))
        self.optimizerDecoder = optim.Adam(self.decoder.parameters(), lr=opt.s1_lr, betas=(opt.beta1, opt.beta2))
        self.optimizerBasis = optim.Adam(self.learnBasis.parameters(), lr=opt.s2_lr, betas=(opt.beta1, opt.beta2))
        self.optimizerCoeff = optim.Adam(self.learnCoeff.parameters(), lr=opt.s2_lr, betas=(opt.beta1, opt.beta2))
        
        # some variables
        real_img = torch.FloatTensor(opt.batchSize_s1, opt.nc, opt.imageSize, opt.imageSize)
        label = torch.FloatTensor(opt.batchSize_s3)
        self.one = torch.FloatTensor([1])
        self.mone = self.one * -1
        self.one = self.one.cuda()
        self.mone = self.mone.cuda()
        if opt.cuda:
            real_img, label = real_img.cuda(), label.cuda()
            self.netD = dataparallel(self.netD, opt.ngpu, opt.gpu)
            self.netG = dataparallel(self.netG, opt.ngpu, opt.gpu)
            self.encoder = dataparallel(self.encoder, opt.ngpu, opt.gpu)
            self.decoder = dataparallel(self.decoder, opt.ngpu, opt.gpu)
            self.learnBasis = dataparallel(self.learnBasis, opt.ngpu, opt.gpu)
            self.learnCoeff = dataparallel(self.learnCoeff, opt.ngpu, opt.gpu)
            self.criterion_bce.cuda()
            self.criterion_l1.cuda()
            self.criterion_l2.cuda()
        self.real_img = Variable(real_img)
        self.label = Variable(label)
        self.batchSize = self.opt.batchSize_s1


    def cal_local_loss(self, recoverd, latent, basis, lcc_coding):
        batch_size = latent.size(0)
        embedding_dim = latent.size(1)
        basis_num = basis.size(0)
        assert(batch_size == lcc_coding.size(0))
        assert(embedding_dim == basis.size(1))
        assert(basis_num == lcc_coding.size(1))
        # compute loss-1 and loss-3 
        l1 = self.criterion_l2(recoverd, latent)
        l3 = self.criterion_l2(basis, torch.zeros_like(basis).cuda())
        # compute loss-2: local loss
        latent_expand = latent.view(batch_size, 1, embedding_dim).expand(batch_size, basis_num, embedding_dim)
        basis_expand = basis.view(1, basis_num, embedding_dim).expand(batch_size, basis_num, embedding_dim)
        lcc_coding_expand = lcc_coding.abs().sqrt()
        lcc_coding_expand = lcc_coding_expand.view(batch_size, basis_num, 1).expand(batch_size, basis_num, embedding_dim)
        l2 = self.criterion_l2(lcc_coding_expand*(latent_expand-basis_expand), torch.zeros_like(basis_expand).cuda())
        loss = 0.5 * l1 + self.opt.LG * l2 + self.opt.LCCLAMBDA * l3
        return loss


    def train(self):
        self.real_label = 1
        self.fake_label = 0
        ############################
        # Stage1: Autoencoder
        ############################
        for epoch in range(self.opt.niter1):
            for i, data in enumerate(self.dataloader, 0):
                real_cpu, _ = data
                batch_size = real_cpu.size(0)
                if batch_size < opt.batchSize_s1:
                    break
                self.real_img.data.resize_(real_cpu.size()).copy_(real_cpu)
                self.encoder.zero_grad()
                self.decoder.zero_grad()

                real_img_hat = self.decoder(self.encoder(self.real_img))
                errRS = self.criterion_l1(real_img_hat, self.real_img)
                errRS.backward()
                self.optimizerEncoder.step()
                self.optimizerDecoder.step()

        ############################
        # Stage2: Train LCC
        ############################
        s2_iters = 10
        s2_batchSize = self.opt.batchSize_s2
        s2_basis_iters = 10
        s2_coeff_iters = 10
        self.dataloader = torch.utils.data.DataLoader(createDataSet(self.opt, self.opt.imageSize), 
            batch_size=s2_batchSize,
            shuffle=True, num_workers=int(self.opt.workers))
        for epoch in range(self.opt.niter2):
            for i, data in enumerate(self.dataloader, 0):
                real_cpu, _ = data
                batch_size = real_cpu.size(0)
                if batch_size < s2_batchSize:
                    break
                self.real_img.data.resize_(real_cpu.size()).copy_(real_cpu)
                latent = self.encoder(self.real_img).detach()
                prerec = self.decoder(latent)
                latent = latent.squeeze()
                # resnet coeffients for new data
                self.learnCoeff.reset_parameters()
                for t in range(s2_iters):
                    ############################
                    # Stage2 (a) Train Coefficients
                    ############################
                    self.learnBasis.eval()
                    self.learnCoeff.train()
                    # basis_T: embedding_dim x basis_num 
                    basis_T = self.learnBasis.weight.detach() 
                    for j in range(s2_coeff_iters):
                        self.learnCoeff.zero_grad()
                        output = self.learnCoeff(basis_T).transpose(0,1).contiguous()
                        # LCC Coding: batch_size x basis_num
                        lcc_coding = self.learnCoeff.weight
                        loss_coeff = self.cal_local_loss(output, latent, basis_T.transpose(0,1).contiguous(), lcc_coding)
                        loss_coeff.backward()
                        self.optimizerCoeff.step()
                    ############################
                    # Stage2 (b) Learn Basis
                    ############################
                    self.learnBasis.train()
                    self.learnCoeff.eval()
                    # LCC Coding: batch_size x basis_num
                    lcc_coding = self.learnCoeff.weight.detach()
                    for j in range(s2_basis_iters):
                        self.learnBasis.zero_grad()
                        output = self.learnBasis(lcc_coding)
                        # basis: basis_num x embedding_dim
                        basis = self.learnBasis.weight.transpose(0,1).contiguous()
                        loss_basis = self.cal_local_loss(output, latent, basis, lcc_coding)
                        loss_basis.backward()
                        self.optimizerBasis.step()

        ############################
        # Stage3: Training GAN
        ############################
        s3_batchSize = self.opt.batchSize_s3
        self.netG.reset_basis(self.learnBasis.weight.transpose(0,1).contiguous())
        self.dataloader = torch.utils.data.DataLoader(createDataSet(self.opt, self.opt.imageSize), 
            batch_size=s3_batchSize*self.opt.criticIters,
            shuffle=True, num_workers=int(self.opt.workers))
        counter_s3 = 0
        for epoch in range(self.opt.niter3):
            for i, data in enumerate(self.dataloader, 0):
                counter_s3 = counter_s3 + 1
                self.netG.train()
                real_cpu, _ = data
                batch_size = real_cpu.size(0)
                if batch_size < s3_batchSize:
                    break
                self.real_img.data.resize_(real_cpu.size()).copy_(real_cpu)
                self.label.data.resize_(batch_size).fill_(1)
                ############################
                # (1) Update D network
                ###########################
                self.netD.zero_grad()
                # train with real
                output = self.netD(self.real_img)
                errD_real = self.criterion_bce(output, self.label)
                errD_real.backward()
                D_x = output.data.mean()
                # train with fake
                noise = torch.randn(batch_size, self.opt.nz)
                noise = noise.cuda()
                noisev = autograd.Variable(noise)
                fake = self.netG(noisev)
                self.label.data.resize_(batch_size).fill_(0)
                output = self.netD(fake.detach())
                errD_fake = self.criterion_bce(output, self.label)
                errD_fake.backward()
                D_G_z1 = output.data.mean()
                errD = errD_real + errD_fake
                self.optimizerD.step()
                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                self.label.data.fill_(1)  # fake labels are real for generator cost
                output = self.netD(fake)
                errG = self.criterion_bce(output, self.label)
                errG.backward()
                D_G_z2 = output.data.mean()
                self.optimizerG.step()


if __name__ == '__main__':
    trainer = Trainer(opt)
    trainer.train()
