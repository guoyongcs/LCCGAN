## LCCGAN

Pytorch implementation for “Adversarial Learning with Local Coordinate Coding”.

## Demonstration of Local Coordinate Coding (LCC)
<div align=center>
<img src="./images/local_g.png" width="350px" />
</div>


## Architecture of LCCGAN
<div align=center>
<img src="./images/architecture.png" width="450px" />
</div>

- AutoEncoder (AE) learns embeddings on the latent manifold.
- Local Coordinate Coding (LCC) learns local coordinate systems.
- The LCC sampling method is conducted on the latent manifold.

## Gometric Views of LCC Sampling
<div align=center>
<img src="./images/lcc_sampling.jpg" width="450px" />
</div>

- With the help of LCC, we obtain local coordinate systems for sampling on the latent manifold.
- Using the local coordinate systems, LCC-GANs always sample some meaningful points to generate new images with different attributes.

## Objective Function
<div align=center>
<img src="./images/objective.png" width="300px" />
</div>

## Training Algorithm
<div align=center>
<img src="./images/algorithm.png" width="300px" />
</div>

## Dependencies
python 2.7

Pytorch

## Dataset
In our paper, to sample different images, we train our model on four datasets, respectively.

- Download [Oxford-102 Flowers](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)  dataset.

- Download [Large-scale CelebFaces Attributes (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  dataset.

- Download [ Large-scale Scene Understanding (LSUN) ](http://lsun.cs.princeton.edu/2016/)  dataset.

## Training
- Train AEGAN on Oxford-102 Flowers dataset.
```
python train.py --dataset flowers --dataroot your_images_folder --batchSize 16 --imageSize 512 --niter_stage1 100 --niter_stage2 1000 --cuda --outf your_images_output_folder --gpu 3
```
- If you want to train the model on Large-scale CelebFaces Attributes (CelebA), Large-scale Scene Understanding (LSUN) or your own dataset. Just replace the hyperparameter like these:
```
python train.py --dataset name_o_dataset --dataroot path_of_dataset
```

