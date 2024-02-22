# Better Reconstruction Loss for VQ-VAE

1. The paper I have chosen to work on is the [https://arxiv.org/abs/1711.00937](VQ-VAE: Neural Discrete Representation Learning)
2. The Extension -

A problem that occurs with VQ-VAE image outputs is image blurriness. The goal of my extension is to tackle this blurriness problem. This occurs due to the reconstruction loss just being the MSE loss.

There are a few different ways I can think to do that.

Firstly, the reconstruction loss term can be replaced with a GAN-like discriminator. This has been done in this paper quite successfully (https://arxiv.org/abs/2012.09841). This adds a fair bit of complexity in terms of training an extra model. 

My proposed extension would be to see if we could achieve an improvement in the blurriness and general image reconstruction domain using specialised loss functions without the need for extra training. This would mean replacing the image reconstruction loss with a greater array of loss terms targeting image reconstruction metrics. A few examples of these terms could be - Structured Similarity Index Metric, Peak Signal-to-Noise Ratio etc. We will experiment with multiple metrics/multiple combinations, to come up with reasonably computationally cheap ways to improve image reconstruction. 

I would build this on top of the VQ-VAE model (the forked project), and compare to that. If time permits, I would also compare it to a comparable implemention of VQGAN.


[Better Reconstruction Loss for VQ-VAE - Results](Final-Project.pdf)


# OLD README
## Reproducing Neural Discrete Representation Learning
### Course Project for [IFT 6135 - Representation Learning](https://ift6135h18.wordpress.com/)

Project Report link: [final_project.pdf](final_project.pdf)

### Instructions
1. To train the VQVAE with default arguments as discussed in the report, execute:
```
python vqvae.py --data-folder /tmp/miniimagenet --output-folder models/vqvae
```
2. To train the PixelCNN prior on the latents, execute:
```
python pixelcnn_prior.py --data-folder /tmp/miniimagenet --model models/vqvae --output-folder models/pixelcnn_prior
```
### Datasets Tested
#### Image
1. MNIST
2. FashionMNIST
3. CIFAR10
4. Mini-ImageNet

#### Video
1. Atari 2600 - Boxing (OpenAI Gym) [code](https://github.com/ritheshkumar95/pytorch-vqvae/tree/evan/video)

### Reconstructions from VQ-VAE
Top 4 rows are Original Images. Bottom 4 rows are Reconstructions.
#### MNIST
![png](samples/vqvae_reconstructions_MNIST.png)
#### Fashion MNIST
![png](samples/vqvae_reconstructions_FashionMNIST.png)

### Class-conditional samples from VQVAE with PixelCNN prior on the latents
#### MNIST
![png](samples/samples_MNIST.png)
#### Fashion MNIST
![png](samples/samples_FashionMNIST.png)

### Comments
1. We noticed that implementing our own VectorQuantization PyTorch function speeded-up training of VQ-VAE by nearly 3x. The slower, but simpler code is in this [commit](https://github.com/ritheshkumar95/pytorch-vqvae/tree/cde142670f701e783f29e9c815f390fc502532e8).
2. We added some basic tests for the vector quantization functions (based on `pytest`). To run these tests
```
py.test . -vv
```

### Authors
1. Rithesh Kumar
2. Tristan Deleu
3. Evan Racah
