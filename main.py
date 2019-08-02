import os
import argparse
from DET import dataLoader_DET
from solver import Solver

def main(args):
    if args.mode == 'train':
        train_loader = dataLoader_DET(args.root, args.label_path, args.split, args)
        if args.val:
            val_loader = dataLoader_DET(args.val_root, args.val_label, 'val', args)
            train = Solver(train_loader, val_loader, None, args)
        else:
            train = Solver(train_loader, None, None, args)
        train.train()   
    elif args.model == 'test':
        test_loader = dataLoader_DET(args.test_root, args.test_label, 'test', args)
        test = Solver(None, None, test_loader, args)
        test.test()
    else:
        raise ValueError('mode is not available!!!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset
    """
    Attention please: Training, testing, and verification correspond to different datasets and npy files
    """
    parser.add_argument('--name', type=str, default='voc')
    parser.add_argument('--root', type=str, default='./DET/ILSVRC/')
    parser.add_argument('--label_path', type=str, default='./detInfo.npy') # Make it according to your own needs
    parser.add_argument('--num_classes', type=int, default=200)
    parser.add_argument('--val_root', type=str, default='...')
    parser.add_argument('--val_label', type=str, default='...')
    parser.add_argument('--test_root', type=str, default='./test_dataset')
    parser.add_argument('--test_label', type=str, default='...')
    parser.add_argument('--split', type=str, default='train')
    
    # image
    parser.add_argument('--crop_size', type=int, default=221)
    parser.add_argument('--img_size', type=tuple, default=(320, 640))
    
    # dataloader
    parser.add_argument('--num_workers', type=int, default=0)

    # solver
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--power', type=float, default=0.9)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--resume', type=str, help="checkpoint that model resume from")       
    parser.add_argument('--pretrain', type=str , default='../vgg16-397923af.pth') # 
    parser.add_argument('--epoch_save', type=int, default=1)
    parser.add_argument('--epoch_val', type=int, default=2)
    parser.add_argument('--val', type=bool, default=False)
    parser.add_argument('--snapshot', type=str, default='./snapshots/')
    parser.add_argument('--global_counter', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train')
    

    # model
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--model_name', type=str, default='ACoL')



    args = parser.parse_args()
    main(args)