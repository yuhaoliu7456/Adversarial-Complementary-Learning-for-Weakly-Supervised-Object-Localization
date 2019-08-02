import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo


class Model(nn.Module):
    def __init__(self, base, args):
        super(Model, self).__init__()
        self.base = base 
        self.args = args
        self.threshold = self.args.threshold

        self.cls = self.calssifier(512, self.args.num_classes)
        self.cls_erase = self.calssifier(512, self.args.num_classes)

    def calssifier(self, in_channels, out_channels):                     # Convolution operation after backbone
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=out_channels, kernel_size=1),
        )

    def forward(self, x, label=None):
        x = self.base(x)
        feats = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        
        out_A = self.cls(feats)
        self.map_A = out_A
        logits_A = torch.mean(torch.mean(out_A, dim=2), dim=2)            # GAP

        attention_map =  self.get_attention_map(out_A, label, True)
        self.attention = attention_map
        feats_erased = self.get_erased_feature(attention_map, feats, self.threshold)
        
        out_B = self.cls_erase(feats_erased)
        self.map_B = out_B
        logits_B = torch.mean(torch.mean(out_B, dim=2), dim=2)
        return logits_A, logits_B


    def get_attention_map(self, feature_maps, label, normalize=True):
        """
        Get the specified channel corresponding to the feature map according to the index of the non-zero value in the label.
        atten_map is used to store attention map , which is initialized to an all-zero feature map,
        When the img's label is non-zero more than 1, you need to use another array of all zeros as a 
        transition to perform cat operations on multiple channels.
        Because each time after the cat operation is two channels, the zero is initialized to one more than the attention.
        After performing the cat operation, in order to reduce the dimension, perform a maximum pooling between channels.
        """
        label = label.long()
        feature_map_size = feature_maps.size() 
        batch_size = feature_map_size[0]        

        for batch_idx in range(batch_size):
            atten_map = torch.zeros([feature_map_size[0], 1 , feature_map_size[2], feature_map_size[3]])  # [batchsize,64,64] fully-zero map
            atten_map = Variable(atten_map.cuda())
            len_class_maps = torch.nonzero(label.data[batch_idx])  # 34, 56, 90
            for i in range(len(len_class_maps)):
                zero_ = torch.zeros([feature_map_size[0], atten_map.size()[1] + 1,feature_map_size[2], feature_map_size[3]]).cuda()
                zero_[batch_idx,:,:,:] = torch.cat((atten_map[batch_idx,:,:,:], torch.unsqueeze(feature_maps[batch_idx, len_class_maps[i,:].item() ,:,:], dim=0)))
                atten_map = torch.cat((atten_map, torch.zeros([feature_map_size[0],1 ,feature_map_size[2], feature_map_size[3]]).cuda()), dim=1)
                atten_map = zero_
        atten_map, index = torch.max(atten_map, dim=1)
        
        if normalize:
            atten_map = self.normalize_atten_maps(atten_map)
        return atten_map


    def get_erased_feature(self, attention_map, feature_maps, threshold):
        if len(attention_map.size())>3:
            attention_map = torch.squeeze(attention_map)
        atten_shape = attention_map.size()

        pos = torch.ge(attention_map, threshold)
        mask = torch.ones(atten_shape).cuda()
        mask[pos.data] = 0.0                        
        mask = torch.unsqueeze(mask, dim=1)
        erased_feature_maps = feature_maps * Variable(mask)
        return erased_feature_maps

 
    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()
        # 1, 5, 10, 10 ---->  1, 5, 100           
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,))-batch_mins, batch_maxs - batch_mins)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed


    def get_localization_maps(self):
        return torch.max(self.map_A, self.map_B)


def _init_weight(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


def make_layers(cfg, dilation=None, batch_norm=False):
    layers = []
    in_channels = 3
    for v, d in zip(cfg, dilation):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d, dilation=d)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def model(args, pretrained=False ):
    # base and dilation can be modified according to your needs
    base = {'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512, 'N']}  
    dilation = {'D1': [1, 1, 'M', 1, 1, 'M', 1, 1, 1, 'M', 1, 1, 1, 'N', 1, 1, 1, 'N']}
    model_url = 'https://download.pytorch.org/models/vgg16-397923af.pth'

    model = Model(make_layers(base['D1'], dilation=dilation['D1']), args)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_url))
    return model


if __name__ == "__main__":
    model = model(num_classes=200, args=None, threshold=0.5) 
    x = torch.randn(1,3,100,100)
    for name,value in model.named_parameters():
        print(name, '---', value)

