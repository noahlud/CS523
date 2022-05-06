import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn

class InceptionV3(nn.Module):
    #Pretrained InceptionV3 network returning feature maps

    def __init__(self,
                 resize_input=True,
                 requires_grad=False):
        
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input

        self.blocks = nn.ModuleList()

        
        inception = models.inception_v3(pretrained=True)

        # Inception model except for last linear layer
        block = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
        self.block=nn.Sequential(*block)


        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        # get latent space
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)


        out = self.block(x)

        return out