import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides_1=1, strides_2=1, strides_3=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides_1, bias=False)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, stride=strides_2, bias=False)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides_3, bias=False)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = (F.gelu(self.bn1(self.conv1(X))))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.bn3(self.conv3(X))
        Y += X
        return (Y)

class ResNetwork18(nn.Module):
    """
    The Very Low early deadline architecture is a 4-layer CNN.
    The first Conv layer has 64 channels, kernel size 7, and stride 4.
    The next three have 128, 256, and 512 channels. Each have kernel size 3 and stride 2.
    Think about what the padding should be for each layer to not change spatial resolution.
    Each Conv layer is accompanied by a Batchnorm and ReLU layer.
    Finally, you want to average pool over the spatial dimensions to reduce them to 1 x 1.
    Then, remove (Flatten?) these trivial 1x1 dimensions away.
    Look through https://pytorch.org/docs/stable/nn.html 
    TODO: Fill out the model definition below! 

    Why does a very simple network have 4 convolutions?
    Input images are 224x224. Note that each of these convolutions downsample.
    Downsampling 2x effectively doubles the receptive field, increasing the spatial
    region each pixel extracts features from. Downsampling 32x is standard
    for most image models.

    Why does a very simple network have high channel sizes?
    Every time you downsample 2x, you do 4x less computation (at same channel size).
    To maintain the same level of computation, you 2x increase # of channels, which 
    increases computation by 4x. So, balances out to same computation.
    Another intuition is - as you downsample, you lose spatial information. Want
    to preserve some of it in the channel dimension.
    """
    def __init__(self, num_classes=7000):
        super().__init__()

        self.backbone = nn.Sequential(
            # Note that first conv is stride 4. It is (was?) standard to downsample.
            # 4x early on, as with 224x224 images, 4x4 patches are just low-level details.
            # Food for thought: Why is the first conv kernel size 7, not kernel size 3?

            # TODO: Conv group 1
            nn.Conv2d(3, 64, 7, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            # TODO: Conv group 2
            Residual(64, 64),
            Residual(64, 64),
            Residual(64, 128, use_1x1conv=True,strides_1=2, strides_2=1, strides_3=2),
            #dow
            Residual(128, 128),
            Residual(128, 256, use_1x1conv=True,strides_1=2, strides_2=1, strides_3=2),
            #dow
            Residual(256, 256),
            Residual(256, 512, use_1x1conv=True,strides_1=2, strides_2=1, strides_3=2),
            #dow
            Residual(512, 512),
            # TODO: Average pool over & reduce the spatial dimensions to (1, 1)
            nn.AdaptiveAvgPool2d((1,1)),
            # TODO: Collapse (Flatten) the trivial (1, 1) dimensions
            nn.Flatten()
            ) 
        
        self.cls_layer = nn.Linear(512, num_classes)
    
    def forward(self, x, return_feats=False):
        """
        What is return_feats? It essentially returns the second-to-last-layer
        features of a given image. It's a "feature encoding" of the input image,
        and you can use it for the verification task. You would use the outputs
        of the final classification layer for the classification task.

        You might also find that the classification outputs are sometimes better
        for verification too - try both.
        """
        feats = self.backbone(x)
        #print(feats.shape)
        out = self.cls_layer(feats)

        if return_feats:
            return feats
        else:
            return out

class Verification():
    def __init__(self) -> None:
        self.model = ResNetwork18()
        self.model.load_state_dict(torch.load(f"./model_res_34_3.pt",map_location="cpu"))
        pass
    
    def verify(self, array):
        img = torch.from_numpy(array)
        print(img.shape)
        outputs_1 = self.model(img.permute(0, 3, 1, 2).float(), return_feats=True)
        return outputs_1.detach().numpy()
        pass