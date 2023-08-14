# --coding:utf-8--
import torch
import torch.nn as nn

class InputTransition3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        initial parameter for input transition <input size equals to output feature size>
        :param in_channels: input image channels
        :param out_channels: output feature channles

        in_channels: 1
        """
        super(InputTransition3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.activate1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.activate1(self.bn1(self.conv1(x)))
        return x


class Conv3D_Block(nn.Module):

    def __init__(self, in_feat, out_feat=None, kernel=3, stride=1, padding=1, innnermost=False, residual=None):

        super(Conv3D_Block, self).__init__()

        self.conv1 = nn.Sequential(
                        nn.Conv3d(in_feat, out_feat, kernel_size=kernel,
                                    stride=stride, padding=padding, bias=True),
                        nn.InstanceNorm3d(out_feat, affine=True),
                        nn.ReLU())

        self.residual = residual

        if self.residual is not None:
            self.residual_upsampler = nn.Conv3d(in_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):

        res = x

        if not self.residual:
            return self.conv1(x)
        else:
            return self.conv1(x) + self.residual_upsampler(res)

def residual_block3d(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv3d(in_channel, out_channel, kernel_size=(
            3, 3, 3), stride=1, padding=1),
        nn.InstanceNorm3d(out_channel, affine=True),
        nn.ReLU(),
        nn.Conv3d(out_channel, out_channel, kernel_size=(
            3, 3, 3), stride=1, padding=1),
        nn.InstanceNorm3d(out_channel, affine=True),
        nn.ReLU(),
        nn.Conv3d(out_channel, out_channel, kernel_size=(
            3, 3, 3), stride=1, padding=1),
        nn.InstanceNorm3d(out_channel, affine=True),
        nn.ReLU()
    )


class Encoder3d(nn.Module):
    def __init__(self, in_channels=1, residual='conv'):
        '''
        in_channels: 1        
        '''
        super(Encoder3d, self).__init__()

        # Encoder downsamplers
        self.pool1 = nn.MaxPool3d((2,2,2))
        self.pool2 = nn.MaxPool3d((2,2,2))
        self.pool3 = nn.MaxPool3d((2,2,2))

        feats = [16, 32, 64, 128]
        # Encoder conv
        self.in_tr = InputTransition3d(in_channels, feats[0]) # can be regarded as down_conv_bk1

        self.down_conv_bk2 = Conv3D_Block(feats[0], feats[1], residual=residual)
        self.down_conv_bk3 = Conv3D_Block(feats[1], feats[2], residual=residual)
        self.down_conv_bk4 = Conv3D_Block(feats[2], feats[3], residual=residual)

        self.block_1_pool = nn.MaxPool3d(2, stride=2)
        self.block_2_1 = residual_block3d(feats[3], feats[3])
        self.block_2_pool = nn.MaxPool3d(2, stride=2)
        self.block_3_1 = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(feats[3], feats[3])
        )

    def forward(self, x):
        # with torch.no_grad():
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
    
class Threedim_GAF_CNN(nn.Module):

    def __init__(self, in_channels=1, residual='conv'):
        '''
        in_channels: 1
        '''
        super(Threedim_GAF_CNN, self).__init__()

        feats = [16, 32, 64, 128, 256]

        self.encoder3d = Encoder3d(in_channels=in_channels)

        self.GAF_block = nn.Sequential(
            nn.Linear(219, feats[4]),
            nn.ReLU(),
            nn.Linear(feats[4], feats[3]),
        )

        self.kv_block = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feats[3], feats[3])
        )
        self.q_block = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feats[3], feats[3])
        )
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feats[3], num_heads=8, batch_first=True)

        self.block_3_2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feats[3], 1)
        )

    def forward(self, x, GAF):
        
        x = self.encoder3d(x)
        
        x_cli = self.GAF_block(GAF)

        kv = self.kv_block(x).unsqueeze(1)
        q = self.q_block(x_cli).unsqueeze(1)

        out1 = self.multihead_attn(query=q, key=kv, value=kv, need_weights=False)[0].squeeze(1)

        x = self.block_3_2(out1)

        return x
    