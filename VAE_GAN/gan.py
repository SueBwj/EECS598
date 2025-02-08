from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
NOISE_DIM = 96


def hello_gan():
    print("Hello from gan.py!")


def sample_noise(batch_size, noise_dim, dtype=torch.float, device='cpu'):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - noise_dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, noise_dim) containing uniform
      random noise in the range (-1, 1).
    """
    noise = None
    ##############################################################################
    # TODO: Implement sample_noise.                                              #
    ##############################################################################
    # Replace "pass" statement with your code
    noise = torch.rand((batch_size, noise_dim),
                       device=device, dtype=dtype) * 2 - 1

    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################

    return noise


def discriminator():
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement discriminator.                                           #
    ############################################################################
    # Replace "pass" statement with your code
    input_size = 784
    hidden_size = 256
    output_size = 1
    model = nn.Sequential(
        nn.Linear(in_features=input_size, out_features=hidden_size),
        nn.LeakyReLU(0.01),
        nn.Linear(in_features=hidden_size, out_features=hidden_size),
        nn.LeakyReLU(0.01),
        nn.Linear(in_features=hidden_size, out_features=output_size)
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement generator.                                               #
    ############################################################################
    # Replace "pass" statement with your code
    hidden_size = 1024
    output_size = 784
    model = nn.Sequential(
        nn.Linear(in_features=noise_dim, out_features=hidden_size),
        nn.ReLU(),
        nn.Linear(in_features=hidden_size, out_features=hidden_size),
        nn.ReLU(),
        nn.Linear(in_features=hidden_size, out_features=output_size),
        nn.Tanh()
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement discriminator_loss.                                        #
    ##############################################################################
    # Replace "pass" statement with your code
    true_labels = torch.ones_like(
        logits_real, device=logits_real.device, dtype=logits_real.dtype)
    loss_real = F.binary_cross_entropy_with_logits(logits_real, true_labels)

    fake_labels = torch.zeros_like(
        logits_fake, device=logits_fake.device, dtype=logits_fake.dtype)
    loss_fake = F.binary_cross_entropy_with_logits(logits_fake, fake_labels)

    # 总损失是两者之和
    loss = (loss_real + loss_fake)
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement generator_loss.                                            #
    ##############################################################################
    # Replace "pass" statement with your code
    true_labels = torch.ones_like(
        logits_fake, device=logits_fake.device, dtype=logits_fake.dtype)
    # 计算二元交叉熵损失
    loss = F.binary_cross_entropy_with_logits(
        logits_fake, true_labels)
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = None
    ##############################################################################
    # TODO: Implement optimizer.                                                 #
    ##############################################################################
    # Replace "pass" statement with your code
    optimizer = optim.Adam(params=model.parameters(),
                           lr=1e-3, betas=(0.5, 0.999))
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return optimizer


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    ##############################################################################
    # TODO: Implement ls_discriminator_loss.                                     #
    ##############################################################################
    # Replace "pass" statement with your code
    real_labels = torch.ones_like(
        scores_real, dtype=scores_real.dtype, device=scores_real.device)
    # 生成数据标签 a = 0
    fake_labels = torch.zeros_like(
        scores_fake, dtype=scores_fake.dtype, device=scores_fake.device)

    # 计算真实数据的损失
    loss_real = F.mse_loss(scores_real, real_labels)
    # 计算生成数据的损失
    loss_fake = F.mse_loss(scores_fake, fake_labels)

    # 总损失为两者之和
    loss = 0.5 * (loss_real + loss_fake)
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.

    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    ##############################################################################
    # TODO: Implement ls_generator_loss.                                         #
    ##############################################################################
    # Replace "pass" statement with your code
    target_labels = torch.ones_like(
        scores_fake, dtype=scores_fake.dtype, device=scores_fake.device)

    # 计算生成器的损失
    loss = 0.5 * F.mse_loss(scores_fake, target_labels)
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def build_dc_classifier():
    """
    Build and return a PyTorch nn.Sequential model for the DCGAN discriminator implementing
    the architecture in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_classifier.                                     #
    ############################################################################
    # Replace "pass" statement with your code
    model = nn.Sequential(
        nn.Unflatten(dim=-1, unflattened_size=(1, 28, 28)),
        nn.Conv2d(in_channels=1, out_channels=32, stride=1, kernel_size=5),
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(in_features=4 * 4 * 64, out_features=4*4*64),
        nn.LeakyReLU(0.01),
        nn.Linear(in_features=4*4*64, out_features=1)
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def build_dc_generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the DCGAN generator using
    the architecture described in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_generator.                                      #
    ############################################################################
    # Replace "pass" statement with your code
    model = nn.Sequential(
        nn.Linear(in_features=noise_dim, out_features=1024),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=1024),
        nn.Linear(in_features=1024, out_features=7*7*128),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=7*7*128),
        nn.Unflatten(dim=1, unflattened_size=(128, 7, 7)),
        nn.ConvTranspose2d(in_channels=128, out_channels=64,
                           stride=2, padding=1, kernel_size=4),
        nn.ReLU(),
        nn.BatchNorm2d(num_features=64),
        nn.ConvTranspose2d(in_channels=64, out_channels=1,
                           kernel_size=4, stride=2, padding=1),
        nn.Tanh(),
        nn.Flatten()
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model
