import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def trans_fourier_mask(img,target,mask):
    """
    batch process
    """
    target = torch.fft.fft2(target)
    target = torch.fft.fftshift(target)
    img = torch.fft.fft2(img)
    img = torch.fft.fftshift(img)
    mask = torch.from_numpy(mask)
    src_amp,src_pha=torch.abs(img),torch.angle(img)
    tar_amp,_=torch.abs(target),torch.angle(target)
    new_img_amp=torch.mul(tar_amp, mask)+torch.mul(src_amp, 1-mask)
    new_img=torch.fft.ifftshift(new_img_amp*torch.exp(1j*src_pha))
    new_img=torch.real(torch.fft.ifft2(new_img))
    imin = new_img.min()
    imax = new_img.max()
    image = (255 * (new_img - imin) / (imax - imin)).detach().numpy().astype("uint8")
    image=torch.from_numpy(image).float().requires_grad_()
    return image


class Fourier_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, mask,ori, target):
        ori=ori.squeeze(1)
        img=trans_fourier_mask(ori,target,mask)
        # loss=F.mse_loss(img.type(torch.float16),target.type(torch.float16))+F.mse_loss(img.type(torch.float16),ori.type(torch.float16))
        loss=F.mse_loss(img.float(),ori.float())+F.mse_loss(img.float(),target.float())
        return loss.mean()
    
    
class TFUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TFUNet, self).__init__()

        self.feature_extractor = timm.create_model('visformer_small', pretrained=True)

        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        #112x112
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        #56x56
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        #28x28
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        #14x14
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        #7x7
        self.center = nn.Sequential(
            nn.Conv2d(1024+768, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
        )

        self.up5 = nn.Sequential(
            nn.Conv2d(2048+1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.up4 = nn.Sequential(
            nn.Conv2d(1024 + 512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.out = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        feature=self.feature_extractor.forward_features(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.center(torch.cat([x5, feature], dim=1))
        x = self.up5(torch.cat([x, x5], dim=1))
        x = self.up4(torch.cat([x, x4], dim=1))
        x = self.up3(torch.cat([x, x3], dim=1))
        x = self.up2(torch.cat([x, x2], dim=1))
        x = self.up1(torch.cat([x, x1], dim=1))
        x = self.out(x)
        return x

