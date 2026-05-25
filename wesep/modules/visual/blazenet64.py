# Modified from https://github.com/modelscope/ClearerVoice-Studio/blob/6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61/train/target_speaker_extraction_online/models/visual_frontend/blazenet64.py
# Default parameters are from https://github.com/modelscope/ClearerVoice-Studio/blob/6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61/train/target_speaker_extraction_online/config/config_LRS3_lip_SkiM-ar_2spk.yaml
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8


def _rgb_to_grayscale_bt_hw(video_btchw: torch.Tensor) -> torch.Tensor:
    """(B, T, 3, H, W) float → (B, T, H, W), ITU-R BT.601 luma."""
    r = video_btchw[:, :, 0]
    g = video_btchw[:, :, 1]
    b = video_btchw[:, :, 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


class visualNet(nn.Module):
    def __init__(self, causal=False, image_size=128):
        super(visualNet, self).__init__()
        self.causal = causal
        self.image_size = image_size
        if self.causal:
            padding = (4,3,3)
        else:
            padding = (2,3,3)

        
        self.conv = nn.Conv3d(1, 8, kernel_size=(5,7,7), stride=(1,1,1), padding=padding)
        self.norm = nn.BatchNorm3d(8, momentum=0.01, eps=0.001)
        self.act = nn.ReLU()
                            
        self.v_net = BlazeNet(back_model=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def prepared_input(self, visual):
        """
        To adapt to the current data video format, 
        we need to resize the video to the input size of the visual network
        
        input: (B, H, W, 3, T_image)
        output: (B, T_image,76, 76)
        """
        assert (
            visual.shape[1] == self.image_size
            and visual.shape[2] == self.image_size
            and visual.shape[3] == 3
        ), "The input video size is not correct (expect B,H,W,3,T with H=W=image_size)"
        # (B, H, W, 3, T_image) -> (B, T_image, 3, H, W)
        visual = visual.permute(0, 4, 3, 1, 2).contiguous()
        assert visual.shape[2] == 3 and visual.shape[3] == self.image_size, (
            "Expected (B, T, 3, H, W) with H=W=image_size after permute"
        )
        visual = _rgb_to_grayscale_bt_hw(visual)
        
        # Resize to 76x76
        ymin, ymax, xmin, xmax = 15, 91, 27, 103
        visual = visual[:, :, ymin:ymax, xmin:xmax] #(B,T_image,76,76) 
        assert visual.shape[2] == 76 and visual.shape[3] == 76, (
            "The blazenet64 visual network expected crop size 76×76"
        )
        return visual

    def forward(self, batch):
        """
        its preprocessing for the visual network
        input: (B, H, W, 3, T_image)
        output: (B, 224, T_image)
        """
        # Preprocess the input video
        # (B,H,W,3,T_image) -> (B,T_image,76,76)
        batch = self.prepared_input(batch)
        # Resize to 128x128
        batch = F.interpolate(batch, size=(self.image_size,self.image_size), mode='bilinear', align_corners=False)

        batchsize = batch.shape[0]
        batch = self.conv(batch.unsqueeze(1))
        if self.causal:
            batch = batch[:,:,:-4,:,:]
        batch = self.act(self.norm(batch))

        batch = batch.transpose(1, 2)
        batch = batch.reshape(batch.shape[0]*batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4])
        """
        input of self.v_net: (B*T, 8, 128, 128)
        output of self.v_net: (T, 224, B)
        """
        batch = self.v_net(batch)
        batch = batch.reshape(batchsize, -1, batch.shape[1]).transpose(1,2)

        return batch


class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BlazeBlock, self).__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                      kernel_size=kernel_size, stride=stride, padding=padding, 
                      groups=in_channels, bias=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.norm = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        out = self.convs(h) + x
        out = self.norm(out)
        return self.act(out)

class FinalBlazeBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(FinalBlazeBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=kernel_size, stride=2, padding=0,
                      groups=channels, bias=True),
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = F.pad(x, (0, 2, 0, 2), "constant", 0)

        return self.act(self.convs(h))


class BlazeNet(nn.Module):
    def __init__(self, back_model=False):
        super(BlazeNet, self).__init__()
        self.back_model = back_model
        self._define_layers()

    def _define_layers(self):
        if self.back_model:
            self.backbone = nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=24, kernel_size=5, stride=2, padding=0, bias=True),
                nn.ReLU(inplace=True),

                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24, stride=2),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 48, stride=2),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 96, stride=2),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
            )
            self.final = FinalBlazeBlock(96)
            self.classifier_8 = nn.Conv2d(96, 2, 1, bias=True)
            self.classifier_16 = nn.Conv2d(96, 6, 1, bias=True)

        else:
            self.backbone1 = nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=24, kernel_size=5, stride=2, padding=0, bias=True),
                nn.ReLU(inplace=True),

                BlazeBlock(24, 24),
                BlazeBlock(24, 28),
                BlazeBlock(28, 32, stride=2),
                BlazeBlock(32, 36),
                BlazeBlock(36, 42),
                BlazeBlock(42, 48, stride=2),
                BlazeBlock(48, 56),
                BlazeBlock(56, 64),
                BlazeBlock(64, 72),
                BlazeBlock(72, 80),
                BlazeBlock(80, 88),
            )

            self.backbone2 = nn.Sequential(
                BlazeBlock(88, 96, stride=2),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
            )
            self.classifier_8 = nn.Conv2d(88, 1, 1, bias=True)
            self.classifier_16 = nn.Conv2d(96, 1, 1, bias=True)


    def forward(self, x):
        
        x = F.pad(x, (1, 2, 1, 2), "constant", 0)
        
        b = x.shape[0]     

        if self.back_model:
            x = self.backbone(x)          
            h = self.final(x)        
        else:
            x = self.backbone1(x)           
            h = self.backbone2(x)          
        
    
        
        c1 = self.classifier_8(x)      
        c1 = c1.permute(0, 2, 3, 1)     
        c1 = c1.reshape(b, -1, 1)       

        c2 = self.classifier_16(h)     
        c2 = c2.permute(0, 2, 3, 1)     
        c2 = c2.reshape(b, -1, 1)      

        c = torch.cat((c1, c2), dim=1)  

        return c