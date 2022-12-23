from __future__ import absolute_import, division, print_function
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_of_bands):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        num_classes=1000
        num_input_images=1
        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = 32, 32, 32
        self.conv1 = nn.Conv2d(
            num_input_images * num_of_bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)#10*512*4*4
        self.fc0 = nn.Linear (8192, 1024)#8192-50,2048-18
        self.fc1 = nn.Linear(1024,self.fc_hidden1)
        self.fc2 = nn.Linear(1024, self.fc_hidden2)

def initialize_weights(m):
    if isinstance(m,nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data,0)
    elif isinstance(m,nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data,0)

def resnet_multiimage_input(num_layers, pretrained,num_of_bands):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    num_input_images=1
    model = ResNetMultiImageInput(block_type, blocks, num_of_bands=num_of_bands)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class Encoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_of_bands):
        super(Encoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        #if num_input_images > 1:
        if num_of_bands== 3:
            self.encoder = resnets[num_layers](pretrained)		
        else:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_of_bands)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        x = input_image
        #x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.batch1(x)
        x_l1=self.encoder.relu(x)
        x_l2=self.encoder.layer1(self.encoder.maxpool(x_l1))
        x=self.encoder.layer2(x_l2)
        x_l3=self.encoder.layer3(x)
        x=self.encoder.layer4(x_l3)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x= self.encoder.fc0(x)
        mu = self.encoder.fc1(x)
        logvar = self.encoder.fc2(x)
        return mu, logvar,x_l1,x_l2,x_l3


class Decoder(nn.Module):
    def __init__(self,bands,patch_size):
        
        super(Decoder, self).__init__()
        self.patch_size=patch_size
        # Sampling vector
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (6, 6), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding
        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = 32, 32, 32
        self.fc3 = nn.Linear(self.fc_hidden2, 1024)
        self.fc4 = nn.Linear(1024, 8192)
        self.relu = nn.ReLU(inplace=True)
        
        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8192, out_channels=128, kernel_size=self.k1, stride=self.s4,
                               padding=self.pd4),
            #nn.BatchNorm2d(128, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=self.k1, stride=self.s3,
                               padding=self.pd3),
            #nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            #nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True) #Sigmoid   # y = (y1, y2, y3) \in [0 ,1]^3
        )
        self.convTrans9 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=bands, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            #nn.BatchNorm2d(bands, momentum=0.01),
            nn.Sigmoid() #Sigmoid   # y = (y1, y2, y3) \in [0 ,1]^3
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return (eps.mul(std).add_(mu))
        else:
            return mu 
    def forward(self, mu,logvar):
        # Stage 1d
        z=self.reparameterize(mu,logvar)
        z = self.fc3(z)
        z = self.fc4(z)
        x=z.view(z.size(0), 8192, 1, 1)
        x = self.convTrans6(x)
        d_l3 = self.convTrans7(x)
        d_l4 = self.convTrans8(d_l3)
        x = self.convTrans9(d_l4)
        return x,z,d_l3,d_l4


class AutoEncoder(nn.Module):
    def __init__(self,num_layers,pretrained,num_of_bands,patch_size):
        super(AutoEncoder, self).__init__()
        self.num_layers=num_layers
        self.num_of_bands=num_of_bands
        self.patch_size=patch_size
        self.pretrained=pretrained
        self.encoder = Encoder(self.num_layers,self.pretrained,self.num_of_bands)
        self.decoder = Decoder(self.num_of_bands,self.patch_size)
  
    def forward(self, x):
        mu, logvar,x_l1,x_l2,x_l3 = self.encoder(x)	
        d,z,d_l3,d_l4 = self.decoder(mu,logvar)
        return d,z,mu,logvar,x_l1,x_l2,x_l3,d_l3,d_l4
