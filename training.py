import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Critic, Generator, initialize_weights
from utils
import random
import numpy as np
import matplotlib.pyplot as plt

# hyper parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 1.0E-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
Z_DIM = 100
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10


# transform = transforms.Compose(
#     [transforms.Resize(IMAGE_SIZE),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])]
#     )

#CelebA
# dataset = datasets.MNIST(root='./dataset/', train=True, transform=transform, download=True)
# dataset = datasets.CelebA(root='./dataset/', split='train', transform=transforms, download=True)
# loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

class Loader():

critic = Critic(CHANNELS_IMG, FEATURES_DISC).to(device)
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
initialize_weights(critic)
initialize_weights(gen)

opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
# criterion = nn.BCELoss()

fixed_noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
writer_fake = SummaryWriter(f'./logs/fake')
writer_real = SummaryWriter(f'./logs/real')
step = 0

gen.train()
critic.train()

n_tsteps = len(loader)
#training loop
for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        # Train critic : max log(D(x)) + log(1 - D(G(z)))
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()
        # Train generator : min -E[ critic(gen_fake) ]
        output = critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1} / {NUM_EPOCHS}, Step {batch_idx+1} / {n_tsteps}, Loss D : {loss_critic.item():4f}'
                  f', Loss G : {loss_gen.item():4f}')
        with torch.no_grad():
            fake = gen(fixed_noise)
            image_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
            image_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
            writer_fake.add_image('Fake', image_grid_fake, global_step=step)
            writer_real.add_image('Real', image_grid_real, global_step=step)
            
        step += 1

def sample_dataset(data_loader):
    examples = iter(data_loader)
    samples, labels = examples.next()
    print(f'Samples shape : {samples.shape}, Labels shape : {labels.shape}')
    for i in range(max(9, BATCH_SIZE)):
        plt.subplot(3, 3, i+1)
        plt.imshow(samples[i][0], cmap='gray')
    plt.show()