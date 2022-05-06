import os, cv2
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision.io import read_image

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T
from torchvision.models import vgg19
from collections import OrderedDict

from torch.utils.data import Dataset, DataLoader, random_split

from degradation import image_degradation
from generator import Generator
from discriminator import UNetDiscriminator
from loss import generator_loss, discrminator_loss

project_dir = './'


def get_optimizers(G, D, g_learning_rate, d_learning_rate):
    g_optimizer = optim.Adam(G.parameters(), g_learning_rate)
    d_optimizer = optim.Adam(D.parameters(), d_learning_rate)
    return g_optimizer, d_optimizer


def reset_grad(g_optimizer, d_optimizer):
    g_optimizer.zero_grad()
    d_optimizer.zero_grad()


class VGG_feature(nn.Module):
    def __init__(self, outlayer_indices):
        super().__init__()
        # the deepest feature map output before activation(only use layers up to 35th)
        self.vgg = vgg19(pretrained=True).features[:35].eval().to('cuda')
        self.fhooks = []
        self.indices = outlayer_indices
        # dictionary to store feature map according to layer indices
        self.selected_out = OrderedDict()
        self.loss = nn.L1Loss()
        for param in self.vgg.parameters():
            param.requires_grad = False
        for index in outlayer_indices:
            self.fhooks.append(self.vgg[index].register_forward_hook(self.forward_hook(index)))

    def forward_hook(self, layer_index):
        def hook(module, input, output):
            self.selected_out[layer_index] = output

        return hook

    def forward(self, input):
        vgg_input_features = self.vgg(input)
        return self.selected_out


class OST300Dataset(Dataset):
    def __init__(self, project_directory='', img_shape=(256, 256), img_names=[], rand=True):
        self.img_dir = os.path.join(project_directory, 'data')
        if len(img_names) == 0:
            self.img_names = [f for f in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, f))]
        else:
            self.img_names = img_names
        if rand:
            self.resizer = T.RandomCrop(img_shape, padding=0, pad_if_needed=True)
        else:
            self.resizer = T.CenterCrop(img_shape)
        self.tensor = T.ToTensor()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = self.resizer(read_image(img_path)).permute(1, 2, 0)
        degraded_image = image_degradation(image.numpy())
        degraded_image = self.tensor(degraded_image.astype(float)).float()
        image = image.permute(2, 0, 1).float()
        return image, degraded_image


def train(G, D, device, g_weights='models/gen_final.pth', d_weights='models/dis_final.pth',
          project_directory='', num_epochs=1, batch_size=8, img_size=256, save_step=100, silent=False):
    g_weights = torch.load(os.path.join(project_directory, g_weights))['params_ema']
    G.load_state_dict(g_weights)

    d_weights = torch.load(os.path.join(project_directory, d_weights))['params']
    D.load_state_dict(d_weights)

    g_optimizer, d_optimizer = get_optimizers(G, D, 1e-4, 1e-4)
    vgg_r = VGG_feature([2, 7, 16, 25, 34])
    vgg_f = VGG_feature([2, 7, 16, 25, 34])

    data = OST300Dataset(project_directory, img_shape=(img_size, img_size))
    traindata, valdata = random_split(data, [int(len(data) * 0.9), len(data) - int(len(data) * 0.9)],
                                      generator=torch.Generator().manual_seed(42))
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
    adversarial_loss = torch.nn.BCEWithLogitsLoss().to(device)

    losses = []

    # Start training.
    print('Start training...')
    for epoch in range(num_epochs):
        print('Epoch:', epoch + 1, '/', num_epochs)
        for i, (imgs, deg_imgs) in enumerate(trainloader):
            # Fetch real images and degraded images
            imgs, deg_imgs = imgs.to(device), deg_imgs.to(device)

            d_loss, d_loss_real, d_loss_fake = discrminator_loss(G, D, imgs, deg_imgs, adversarial_loss)

            reset_grad(g_optimizer, d_optimizer)
            d_loss.backward()
            d_optimizer.step()

            g_loss, vgg_loss, generator_loss_, reconstruction_loss = generator_loss(G, D, imgs, deg_imgs, vgg_r, vgg_f,
                                                                                    adversarial_loss)
            reset_grad(g_optimizer, d_optimizer)

            g_loss.backward()
            g_optimizer.step()

            # Logging.
            loss = [d_loss_real.item(), d_loss_fake.item(), g_loss.item()]
            losses.append(loss)

            if not silent:
                print('Iteration:', str(i + 1) + '/' + str(len(trainloader.dataset) // batch_size),
                      'Loss Real:', loss[0],
                      'Loss SR:', loss[1],
                      'Generator Loss:', loss[2],
                      end='\r')

            if (i + 1) % save_step == 0:
                time = str(datetime.now())
                gf = os.path.join(project_directory, 'models/gen_' + str(epoch) + '_' + str(i + 1) + '_' + time)
                df = os.path.join(project_directory, 'models/dis_' + str(epoch) + '_' + str(i + 1) + '_' + time)
                print('Generator Model saved at:', gf)
                print('Discriminator Model saved at:', df)
                print('Models saved at', time)
                torch.save(G.state_dict(), gf)
                torch.save(D.state_dict(), df)

    time = str(datetime.now())
    torch.save(G.state_dict(), os.path.join(project_directory, 'models/gen_' + str(num_epochs) + '_final_' + time))
    torch.save(D.state_dict(), os.path.join(project_directory, 'models/dis_' + str(num_epochs) + '_final_' + time))
    print('Training Finished')
    return np.array(losses)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(upsample=4, beta=0.2).to(device)
    D = UNetDiscriminator().to(device)

    losses = train(G, D, device, project_directory=project_dir, num_epochs=10, batch_size=8, img_size=256,
                   save_step=150, silent=False)
    d_loss_real = losses[:, 0]
    d_loss_fake = losses[:, 1]
    g_loss = losses[:, 2]

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 10 * 3))
    ax1.plot(d_loss_real)
    ax1.set_title('Discriminator Real Loss')
    ax2.plot(d_loss_fake)
    ax2.set_title('Discriminator Fake Loss')
    ax3.plot(g_loss)
    ax3.set_title('Generator Loss')
    plt.savefig('test.png')
