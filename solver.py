import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import model, _init_weight   
from loss import Loss
import tqdm as tqdm
from save_atten import SAVE_ATTEN
from PIL import Image
import numpy as np
 

class Solver(object):
    def __init__(self, train_loader, val_loader, test_loader, args):
        self.current_epoch = 0
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.build_model()

    def build_model(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = model(self.args).to(self.device)
        self.loss = Loss(self.args).to(self.device)
        self.net.train()
        self.net.apply(_init_weight)
        self.optimizer = self.get_optimizer(self.net)
        self.restore()
        self.print_param()
    
    def get_optimizer(self, model):
        lr = self.args.lr          # 0.001
        weight_list = [] 
        bias_list = []
        last_weight_list = []
        last_bias_list =[]
        for name, value in model.named_parameters():
            if 'cls' in name:    
                if 'weight' in name:
                    last_weight_list.append(value)
                elif 'bias' in name:
                    last_bias_list.append(value)
            else:
                if 'weight' in name:
                    weight_list.append(value)
                elif 'bias' in name:
                    bias_list.append(value)
        optmizer = torch.optim.SGD([{'params': weight_list,
                                     'lr': lr},
                                    {'params': bias_list,
                                     'lr': lr*2},
                                    {'params': last_weight_list,
                                     'lr': lr*10},
                                    {'params': last_bias_list,
                                     'lr': lr*20}], momentum=self.args.momentum, weight_decay=self.args.weight_decay, nesterov=True)
        return optmizer
                

    def restore(self):
        if self.args.pretrain and os.path.isfile(self.args.pretrain):
            ckpt = torch.load(self.args.pretrain)
            self.net.base.load_state_dict(ckpt, strict=False)
        else:
            restore_dir = self.args.snapshot
            model_list = os.listdir(restore_dir)
            model_list = [x for x in model_list if os.path.isfile(os.path.join(restore_dir, x)) and x.endswith('.pth.tar')]
            if len(model_list) > 0:
                model_list.sort(key=lambda fn: os.path.getmtime(os.path.join(restore_dir, fn)), reverse=True)
                snapshot = os.path.join(restore_dir, model_list[0])
            ckpt = torch.load(snapshot)  # if there is no .pth file in snapshot, it will worng!!
            self.net.load_state_dict(ckpt['state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.current_epoch = ckpt['epoch'] + 1

    def print_param(self):
        num_params = 0
        for p in self.net.parameters():
            if p.requires_grad:
                num_params += p.numel()
        print('model:', self.args.model_name)
        print('The number of parameters: ', num_params)
    
    def adjust_lr(self, current_iter, max_iter):
        lr = self.args.lr * (1 - current_iter/max_iter ) **self.args.power
        for param in self.optimizer.param_groups:
            param['lr'] = lr
        return lr
 


    def save_checkpoint(self, args, state, filename='checkpoint.pth.tar'):
        model_save_path = os.path.join(self.args.snapshot, filename)
        torch.save(state, model_save_path)
   

    def train(self):
        if self.args.val:
            best_val = 1.0
        else:
            best_val = None
        global_counter = self.args.global_counter
        max_iter = int(self.args.epochs * len(self.train_loader) )

        for epoch in range(self.current_epoch, self.args.epochs):
            if self.args.val:
                epoch = self.current_epoch
            epoch_loss = 0.0
            current_iteration = 0
            for i, (images, labels, _imgpath) in enumerate(self.train_loader):
                if (i + 1) > max_iter: break
                current_iteration += 1   #     
                global_counter += 1               # for calculate loss
                lr = self.adjust_lr(global_counter, max_iter)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                logits = self.net(images, labels)
                loss = self.loss(logits, labels)
                loss.backward() 
                self.optimizer.step()
                epoch_loss += loss.item()
                print("epoch: [%d/%d], iter: [%d/%d], loss: [%.4f], lr:[%.8f]" %(epoch, self.args.epochs, current_iteration, len(self.train_loader), loss.item(), lr))
                
            if (epoch + 1) % self.args.epoch_val == 0 and self.args.val:
                val_loss = self.val()
                print('---Best MAE: %.2f, Curr MAE: %.2f ---' %(best_val, val_loss))
                if val_loss < best_val:
                    best_val = val_loss
                    self.save_checkpoint(args=self.args, state={'epoch':epoch,
                                            'epoch_loss':val_loss,
                                            'state_dict':self.net.state_dict(),
                                            'optimizer':self.optimizer.state_dict()}, filename="best.pth.tar")
            if (epoch + 1) % self.args.epoch_save == 0:
                self.save_checkpoint(args=self.args, state={'epoch':epoch,
                                            'epoch_loss':epoch_loss,
                                            'state_dict':self.net.state_dict(),
                                            'optimizer':self.optimizer.state_dict()}, filename="epoch_%d.pth.tar" %(epoch+1))
        torch.save(self.net.state_dict(), "%s/final.pth" %(self.args.snapshot))


            
    def val(self):
        total_loss = 0.0
        self.net.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.val_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.net(images, labels)
                loss = self.val_metric(logits, labels)
                total_loss += loss.item()
        self.net.train()
        return total_loss/len(self.val_loader)
    
    def val_metric(self, logits, labels):
        loss1 = F.multilabel_soft_margin_loss(logits[0], labels.float())
        loss2 = F.multilabel_soft_margin_loss(logits[1], labels.float())
        return (loss1 + loss2)

    def test(self):
        self.net.eval()
        global_counter = self.args.global_counter
        save_atten = SAVE_ATTEN(save_dir='./save_bins/')
        for i, (images, labels,img_path) in enumerate(self.test_loader):
            global_counter += 1
            print('Has finished %d images' %global_counter) 
            images, labels = images.to(self.device), labels.to(self.device)
            logits = self.net(images, labels)
            last_featmaps = self.net.get_localization_maps()
            save_atten.save_masked_img_batch(img_path, last_featmaps, labels)


            
            
