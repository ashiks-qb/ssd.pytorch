from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.dataset == 'COCO':
    if args.dataset_root == VOC_ROOT:
        if not os.path.exists(COCO_ROOT):
            parser.error('Must specify dataset_root if specifying dataset')
        print("WARNING: Using default COCO dataset_root because " +
                "--dataset_root was not specified.")
        args.dataset_root = COCO_ROOT
    cfg = coco
    dataset = COCODetection(root=args.dataset_root,
                            transform=SSDAugmentation(cfg['min_dim'],
                                                        MEANS))
elif args.dataset == 'VOC':
    if args.dataset_root == COCO_ROOT:
        parser.error('Must specify dataset if specifying dataset_root')
    cfg = voc
    dataset = VOCDetection(root=args.dataset_root,
                            transform=SSDAugmentation(cfg['min_dim'],
                                                        MEANS))


class SSDModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
        self.net = self.ssd_net     
        self.criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False)  


        if args.resume:
            print('Resuming training, loading {}...'.format(args.resume))
            self.ssd_net.load_weights(args.resume)
        else:
            vgg_weights = torch.load(args.save_folder + args.basenet)
            print('Loading base network...')
            self.ssd_net.vgg.load_state_dict(vgg_weights)

            print('Initializing weights...')
            # initialize newly added layers' weights with xavier method
            self.ssd_net.extras.apply(weights_init)
            self.ssd_net.loc.apply(weights_init)
            self.ssd_net.conf.apply(weights_init)

        self.net.train() 
        self.step_index = 0

    def forward(self, images):
        return self.net(images)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.net.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        def adjust_learning_rate(step):
            """Sets the learning rate to the initial LR decayed by 10 at every
                specified step
            # Adapted from PyTorch Imagenet example:
            # https://github.com/pytorch/examples/blob/master/imagenet/main.py
            """
            if step in cfg["lr_steps"]:
                self.step_index += 1
                lr_scale = args.gamma ** (self.step_index)
            else:
                lr_scale = 1

            return lr_scale

        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=adjust_learning_rate
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [lr_scheduler_config]
    
    def on_train_start(self):
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        print('\n\nUsing the specified args:\n')
        pp.pprint(vars(args))
        print('\n')

    def training_step(self, train_batch, batch_idx):
        images, targets = train_batch
        out = self(images)
        loss_l, loss_c = self.criterion(out, targets)
        loss = loss_l + loss_c
        self.log("train_loss", loss)
        return loss

    def train_dataloader(self):
        data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
        return data_loader

    def on_epoch_end(self):
        if self.current_epoch != 0 and self.current_epoch % 5000 == 0:
            torch.save(self.ssd_net.state_dict(), 'weights/ssd300_COCO_' +
                       repr(self.current_epoch) + '.pth')
    #TODO: Dataparallel


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()



if __name__ == '__main__':
    trainer = pl.Trainer(gpus=1)
    model = SSDModel()

    trainer.fit(model)