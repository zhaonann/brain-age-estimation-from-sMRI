# --coding:utf-8--
import torch
import torch.nn as nn

class InputTransition2d(nn.Module):
    # TODO: input transition convert image to feature space
    def __init__(self, in_channels, out_channels):
        """
        initial parameter for input transition <input size equals to output feature size>
        :param in_channels: input image channels
        :param out_channels: output feature channles

        in_channels: 1
        out_channels: 16
        """
        super(InputTransition2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.activate1 = nn.ReLU()

    def forward(self, x):
        x = self.activate1(self.bn1(self.conv1(x)))
        return x

class Conv2D_Block(nn.Module):

    def __init__(self, in_feat, out_feat=None, kernel=3, stride=1, padding=1, innnermost=False, residual=None):

        super(Conv2D_Block, self).__init__()

        self.conv1_blk = nn.Sequential(
                        nn.Conv2d(in_feat, out_feat, kernel_size=kernel,
                                    stride=stride, padding=padding, bias=True),
                        nn.InstanceNorm2d(out_feat, affine=True),
                        nn.ReLU())

        self.residual = residual

        if self.residual is not None:
            self.downsample = nn.Conv2d(in_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):

        identity = x
        out = self.conv1_blk(x)
        if not self.residual:
            return out
        else:
            out = out + self.downsample(identity)
            return out

def residual_block2d(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=1, padding=1),
        nn.InstanceNorm2d(out_channel, affine=True),
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=1, padding=1),
        nn.InstanceNorm2d(out_channel, affine=True),
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=1, padding=1),
        nn.InstanceNorm2d(out_channel, affine=True),
        nn.ReLU()
    )

class Encoder2d(nn.Module):
    def __init__(self, in_channels=1, residual='conv'):

        super(Encoder2d, self).__init__()

        feats = [16, 32, 64, 128]
        # Encoder downsamplers
        self.pool1 = nn.MaxPool2d((2,2), stride=2)
        self.pool2 = nn.MaxPool2d((2,2), stride=2)
        self.pool3 = nn.MaxPool2d((2,2), stride=2)

        # Encoder conv
        self.in_tr = InputTransition2d(in_channels, feats[0])
        self.down_conv_bk2 = Conv2D_Block(feats[0], feats[1], residual=residual)
        self.down_conv_bk3 = Conv2D_Block(feats[1], feats[2], residual=residual)
        self.down_conv_bk4 = Conv2D_Block(feats[2], feats[3], residual=residual)

        self.block_1_pool = nn.MaxPool2d(2, stride=2)
        self.block_2_1 = residual_block2d(feats[3], feats[3])
        self.block_2_pool = nn.MaxPool2d(2, stride=2)
        self.block_3_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feats[3], feats[3])
        )

    def forward(self, x):
        out1 = self.in_tr(x)
        down1 = self.pool1(out1)
        out2 = self.down_conv_bk2(down1)
        down2 = self.pool2(out2)
        out3 = self.down_conv_bk3(down2)
        down3 = self.pool3(out3)
        base = self.down_conv_bk4(down3)

        x = base
        x = self.block_1_pool(x)
        x = self.block_2_1(x) + x
        x = self.block_2_pool(x)
        x = self.block_3_1(x)

        return x

class Threeview_CNN(nn.Module):
    def __init__(self):

        super(Threeview_CNN, self).__init__()

        feats = [16, 32, 64, 128]
        self.drop_rate = 0.5

        self.encoder_B = Encoder2d()

        self.fusion_blk = nn.Sequential(
            nn.Linear(3*feats[3], 2*feats[3]),
            nn.ReLU(),
            nn.Linear(2*feats[3], feats[3]),
            nn.ReLU(),
            nn.Linear(feats[3], 1)
        )

    def forward(self, x1, x2, x3):        
        
        out1 = self.encoder_B(x1)
        out2 = self.encoder_B(x2)
        out3 = self.encoder_B(x3)
 
        out = torch.cat((out1, out2, out3), dim=1)
        out = self.fusion_blk(out)

        return out