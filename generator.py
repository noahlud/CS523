import torch
import torch.nn as nn
import torch.nn.functional as F


class RDenseBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, beta, channels: int = 64, growths: int = 8, n: int = 3) -> None:
        super(RDenseBlock, self).__init__()
        self.beta = beta
        self.rdb1 = DenseBlock()
        self.rdb2 = DenseBlock()
        self.rdb3 = DenseBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Execute n chained dense blocks appending each output as the next input
        out = x + self.rdb1(x) * self.beta
        out = out + self.rdb2(out) * self.beta
        out = out + self.rdb3(out) * self.beta
        return out * self.beta + x


class DenseBlock(nn.Module):
    def __init__(self, nf=64, growths=32, bias=True):
        super(DenseBlock, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, growths, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + growths, growths, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * growths, growths, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * growths, growths, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * growths, nf, kernel_size=3, stride=1, padding=1, bias=bias)
        self.LRelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.LRelu(self.conv1(x))
        x2 = self.LRelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.LRelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.LRelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5


class Generator(nn.Module):
    """Generator network."""

    # n = number of repeated block in each RRDB module
    # B = number of RRDB block
    # beta = weight of the output from RRDBlock when adding to input pre-RRDBlock
    def __init__(self, upsample: int = 4, conv_dim: int = 64, beta: float = 1.2, n: int = 5, B: int = 23):
        super(Generator, self).__init__()
        self.upsample = upsample
        self.conv_dim = conv_dim
        self.beta = beta
        self.n = n
        self.B = B

        self.unshuffle = nn.PixelUnshuffle(4 // upsample)
        self.shuffle = nn.PixelShuffle(self.upsample)

        # Convert to low-resolution, high-dimension space
        self.conv_first = nn.Conv2d(3 * ((4 // upsample) ** 2), conv_dim, kernel_size=3, stride=1, padding=1)
        self.LRelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Residual in residual dense block built using RDenseBlock. ESRGAN paper uses 23 of these
        self.body = nn.ModuleList()
        for i in range(B):
            self.body.append(
                RDenseBlock(self.beta, channels=conv_dim, growths=32, n=3, )
            )
        self.conv_body = nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1)

        # Upsampling layer which doubles the resolution of the image each time
        self.conv_up1 = nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1)
        self.conv_up2 = nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        # Connecting convolution layer after upsampling
        self.conv_hr = nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1)

        # Convert back to 3-channel image
        self.conv_last = nn.Conv2d(conv_dim, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample < 4:
            out = self.unshuffle(x)
        else:
            out = x
        out = self.conv_first(out)
        out = self.LRelu(out)

        # Iterate over each dense block
        x = out
        for i in range(self.B):
            out = self.body[i](out)

        # Post RRDB
        out = self.conv_body(out)

        # Upscale
        out = self.conv_up1(self.upsample1(out))
        out = self.LRelu(out)

        out = self.conv_up2(self.upsample2(out))
        out = self.LRelu(out)
        # Post upsample
        out = self.conv_hr(out)

        # Convert to 3-channel output
        out = self.conv_last(out)
        return out
