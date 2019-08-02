
import os
import cv2
import torch
import numpy as np


class SAVE_ATTEN(object):
    def __init__(self, save_dir='../save_bins'):
        """
        save_dir: the path for saving target
        """
        self.save_dir = save_dir             

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)


    def save_masked_img_batch(self, path_batch, atten_batch, label_batch):
        """
        process batch_size images; the path_batch and label_batch obtained from dataloader,and the atten_batch obtained directly from the modle(without any changes)
        atten_batch：get the attention_map after a layer in the forwad function of the model.py
        for example:
            def forward(self,x):
                x = self.layer1(x)
                x = self.layer2(x)
                return x
            def get_attention_map(self):
                return self.layer1(x)
        the procedure of obtaining attention_map
        :   external input ----> batch selection ----> channel selection ----> normalize ----> resize ----> denormalize ----> save
        """
        img_num = atten_batch.size()[0]
        for idx in range(img_num):
            atten = atten_batch[idx]
            atten = atten.cpu().data.numpy()
            label = label_batch[idx]
            label_list = self.get_label(label)                         # Label_list may be multiple labels, so it may correspond to multiple maps
            self._save_masked_img(path_batch[idx], atten, label_list)


    def _save_masked_img(self, img_path, atten, label_list):
        """
        Process each image in turn
        label: the target class which will be used as the index to get correspoing channel 
        """
        if not os.path.isfile(img_path):
            raise 'Image not exist:%s'%(img_path)

        for each_label in label_list:
            label = each_label[0]
            attention_map = atten[label,:,:]          # now is [width, height]
            atten_norm = attention_map
            
            img = cv2.imread(img_path)
            org_size = np.shape(img)
            w, h = org_size[0], org_size[1]

            # regularize each attention map
            atten_norm = self.normalize_map(atten_norm)
            atten_norm = cv2.resize(atten_norm, dsize=(h,w))

            atten_norm = atten_norm* 255
       
            heat_map = cv2.applyColorMap(atten_norm.astype(np.uint8), cv2.COLORMAP_JET)
            img = cv2.addWeighted(img.astype(np.uint8), 0.5, heat_map.astype(np.uint8), 0.5, 0)

            img_id = img_path.strip().split('/')[-1]
            img_id = img_id.strip().split('.')[0]
            
            save_dir = os.path.join(self.save_dir, img_id + '_' + str(label) +'.png')
            cv2.imwrite(save_dir, img)


    def normalize_map(self, atten_map):
        min_val = np.min(atten_map)
        max_val = np.max(atten_map)
        atten_norm = (atten_map - min_val)/(max_val - min_val)
        return atten_norm


    def get_label(self, gt_label):
        labels_idx = []
        labels_idx =  torch.nonzero(gt_label.squeeze()).cpu().numpy()
        # labels_idx is a list and it'type like this [[12], [22]], so you must use labels_idx[i][0] to get the correspoding label class
        return labels_idx
    

    def _merge_multi_class(self, atten, label_list):
        atten_norm = torch.zeros_like(atten)
        for each_label in label_list:
            label = each_label[0]        
            atten_norm += atten[label,:,:]
        # atten_norm can be processed outside of the for loop
        return atten_norm
           
            
    
