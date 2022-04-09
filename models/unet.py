import torch # 1.9
import torch.nn as nn
from torch.nn import functional as F
import torchvision


class convBlock(nn.Module):
    def __init__(self, in_ch, out_ch, padding = 'same', kernel_size=3):
        super().__init__()
        kernel_size = kernel_size
        pad_size = lambda kernel_size:(kernel_size-1)//2
        if padding=='same':
            self.padding = pad_size(kernel_size)
        else:
            self.padding = padding
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=self.padding, bias=False)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=self.padding, bias=False)
        self.Norm = nn.InstanceNorm2d(out_ch, affine=True)
        
    def forward(self, x):
        x = self.Norm(self.conv1(x))
        x = self.relu(x)
        x = self.Norm(self.conv2(x))
        x = self.relu(x)
        return x

class UpSampleConvs(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.relu  = nn.ReLU()
        self.upSample = nn.Upsample(scale_factor=2)
        self.Norm = torch.nn.InstanceNorm2d(out_ch)
        
    def forward(self, x):
        x = self.upSample(x)
#         x = self.relu(x)
        x = self.conv(x)
        x = self.Norm(x)
        x = self.relu(x)
#         return self.relu(self.conv2(self.relu(self.upSample(x))))
        return x

class Encoder(nn.Module):
    def __init__(self, chs=(3,32,64,128,256,512), padding='same'):
        super().__init__()
        self.FPN_enc_ftrs = nn.ModuleList([convBlock(chs[i], chs[i+1], padding) for i in range(len(chs)-1)])
#         self.pool = nn.MaxPool2d(2)
        self.pool = torch.max_pool2d
        
    def forward(self, x):
        features = []
        
        for block in self.FPN_enc_ftrs:
            x = block(x)
            features.append(x)
#             print(x.shape)
            x = self.pool(x, kernel_size=2)
        return features

class Decoder(nn.Module):
    def __init__(self, chs=(512, 256, 128, 64, 32), padding='same'):
        super().__init__()

        self.chs = chs
        self.padding = padding
#         self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])  # 轉置卷積
        self.upconvs = nn.ModuleList([UpSampleConvs(chs[i], chs[i+1]) for i in range(len(chs)-1)]) # 上採樣後卷積
        self.FPN_dec_ftrs = nn.ModuleList([convBlock(chs[i], chs[i+1], padding=padding) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            enc_ftrs = encoder_features[i]
            x = self.upconvs[i](x)
#             if self.padding == 0:
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.FPN_dec_ftrs[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

class UNet(nn.Module):
    def __init__(self, 
                 enc_chs=(3,64,128,256,512),
                 dec_chs=(512, 256, 128),
                 padding='same'):
        super().__init__()
        self.encoder     = Encoder(enc_chs, padding=padding)
        self.decoder     = Decoder(dec_chs, padding=padding)

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:]) # 把不同尺度的所有featuremap都輸入decoder，我們在decoder需要做featuremap的拼接
        return out