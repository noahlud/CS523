from collections import OrderedDict

import torch
import torch.nn as nn


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


def generator_loss(G, D, x_real, x_degraded, vgg_feature_r, vgg_feature_f, criterion,
                   perceptual_weights=[0.1, 0.1, 1, 1, 1], lambda_l1=1, lambda_percep=1, lambda_gen=0.1):
    batch_size = x_real.shape[0]
    x_SR = G(x_degraded)
    # l1 loss
    loss = nn.L1Loss()
    l1_loss = loss(x_SR, x_real)

    # GAN generator loss
    real = torch.ones((batch_size, 1), requires_grad=False).to('cuda')  # real_label
    # given a pixel wise realness D(x_SR), reshape and take mean of each D-image
    out_src = torch.mean(D(x_SR).view(batch_size, -1), 1, True)
    g_loss_fake = criterion(out_src, real)

    # create two VGG_feature extractor for real input and fake input
    # list of feature maps output
    selected_out_r = vgg_feature_r(x_real.detach())
    selected_out_f = vgg_feature_f(x_SR)
    selected_r = vgg_feature_r.selected_out
    selected_f = vgg_feature_f.selected_out
    vgg_loss = 0
    # with weights, summing l1 losses comparing different real/fake feature maps
    for fea_r, fea_f, w in zip(selected_r, selected_f, perceptual_weights):
        vgg_loss += w * loss(selected_f[fea_f], selected_r[fea_r])

    g_loss = vgg_loss * lambda_percep + lambda_gen * g_loss_fake + lambda_l1 * l1_loss

    return g_loss, vgg_loss, g_loss_fake, l1_loss


def discrminator_loss(G, D, x_real, x_degraded, criterion):
    batch_size = x_real.shape[0]
    # loss for real image
    realness = D(x_real).view(batch_size, -1)
    averaged_realness = torch.mean(realness, 1, True)
    real = torch.ones((batch_size, 1), requires_grad=False).to('cuda')
    d_loss_real = criterion(averaged_realness, real)

    # loss for fake
    x_fake = G(x_degraded)
    fake = torch.zeros((batch_size, 1), requires_grad=False).to('cuda')
    d_loss_fake = criterion(torch.mean(D(x_fake).view(batch_size, -1), 1, True).detach(), fake)

    d_loss = d_loss_real + d_loss_fake
    return d_loss, d_loss_real, d_loss_fake
