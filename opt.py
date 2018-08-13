import argparse

parser = argparse.ArgumentParser()
# training parameters
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | flowers')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize_s1', type=int, default=64, help='input batch size')
parser.add_argument('--batchSize_s2', type=int, default=1000, help='input batch size')
parser.add_argument('--batchSize_s3', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=3, help='size of the latent z vector')
parser.add_argument('--nc', type=int, default=3, help='the number of image channel')
parser.add_argument('--embedding_dim', type=int, default=100, help='the dimension of the extracted embedding from AutoEncoder')
parser.add_argument('--basis_num', type=int, default=128, help='the number of bases in lcc')

# optimizer configuration
parser.add_argument('--s1_lr', type=float, default=0.0001, help='0.01 for lcc | 0.00001 for AE | 0.0002 for GAN')
parser.add_argument('--s2_lr', type=float, default=0.01, help='0.01 for lcc | 0.00001 for AE | 0.0002 for GAN')
parser.add_argument('--s3_lr', type=float, default=0.0002, help='0.01 for lcc | 0.00001 for AE | 0.0002 for GAN')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam')
parser.add_argument('--criticIters', type=int, default=1, help='for WGAN and WGAN-GP, number of critic iters per gen iter')
parser.add_argument('--LCCLAMBDA', type=float, default=0.2, help='LCC hyperparameter')
parser.add_argument('--niter1', type=int, default=100, help='number of iterations in stage 1')
parser.add_argument('--niter2', type=int, default=3, help='number of iterations in stage 2')
parser.add_argument('--niter3', type=int, default=100, help='number of iterations in stage 3')

# model configuration
parser.add_argument('--ngf', type=int, default=64, help='feature number of generator')
parser.add_argument('--ndf', type=int, default=64, help='feature number of discriminator')

# general settings
parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', default=2, type=int, help='manual seed')

opt = parser.parse_args()
print(opt)