import os
import torch
import cv2
import random
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np


class DETdataset(data.Dataset):
    def __init__(self, data_path, label_path, transform, split):
        self.data_path = data_path
        self.label_path = label_path
        self.split = split

        self.List = np.load(label_path)
        
        '''pre-processing setting'''
        self.transform = transform

    def __len__(self):
        return len(self.List)

    def __getitem__(self, idx):

        imgName = self.List[idx]['imgName']
        imgLabel = self.List[idx]['label']

        img_path = os.path.join(self.data_path, 'Data', 'DET', self.split, imgName + '.JPEG')
        img = Image.open(img_path).convert('RGB')
        """
        The label in npy can be written directly as 01, or it can be written as -1 +1 and then gt.
        attention: the method to read image_label from npy is different between train and val in my task!!!
        """
        
        imgLabel = torch.from_numpy(imgLabel)   
        if self.split == 'train':
            imgLabel = torch.gt(imgLabel,0)
            img = self.transform(img)
 
        return img, imgLabel, img_path



def dataLoader_DET(data_path, label_path, split, args):
    transform = transforms.Compose([transforms.Resize(args.img_size), 
                                        transforms.RandomCrop(args.crop_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = DETdataset(data_path=data_path, label_path=label_path, transform=transform, split=split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    return dataloader

