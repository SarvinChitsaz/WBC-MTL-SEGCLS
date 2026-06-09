import torch
import torch.nn as nn
import torchvision.models as models


class UNetResNet34MultiTask(nn.Module):
    def __init__(self, n_classes_seg=3, n_classes_cls=4):
        super().__init__()

        backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool0 = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4

        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = nn.Sequential(nn.Conv2d(768, 512, 3, padding=1), nn.ReLU(inplace=True))

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(inplace=True))

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(192, 128, 3, padding=1), nn.ReLU(inplace=True))

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True))

        self.out_seg = nn.Conv2d(64, n_classes_seg, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_cls = nn.Linear(1024, n_classes_cls)

    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(self.pool0(enc0))
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        center = self.center(enc4)

        up4 = self.up4(center)
        dec4 = self.dec4(torch.cat([up4, enc3], dim=1))

        up3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([up3, enc2], dim=1))

        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc1], dim=1))

        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc0], dim=1))

        seg_out = self.out_seg(dec1)
        cls_out = self.fc_cls(self.avgpool(center).view(center.size(0), -1))

        return seg_out, cls_out        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True))

        self.out_seg = nn.Conv2d(64, n_classes_seg, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_cls = nn.Linear(1024, n_classes_cls)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool0(e0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        c = self.center(e4)

        d4 = self.dec4(torch.cat([self.up4(c), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e0], dim=1))

        seg_out = self.out_seg(d1)
        cls_out = self.fc_cls(self.avgpool(c).view(c.size(0), -1))

        return seg_out, cls_out
