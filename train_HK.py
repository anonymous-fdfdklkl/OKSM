
import torch


import json

from utils import Losses
import argparse

from utils.utils import progress_bar,get_optim,get_dataloader,get_network,get_anchors

import os


class get_work():
    def __init__(self,args,cfg):
        print(vars(args))
        torch.manual_seed(0)

        if args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter('runs/{}_{}_{}CACUR'.format(args.dataset, args.trial, args.name))
        self.args, self.cfg = args, cfg
        self.trainloader, self.valloader= get_dataloader(args,cfg)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print('==> Building network..')

        self.net = get_network(args,cfg)

        # anchors = get_anchors(args,cfg)
        # self.net.set_anchors(anchors)

        self.net = self.net.to(self.device)

        # self.net.train()
        self.optimizer = get_optim(args,cfg,self.net)

        ## Get loss

        self.Loss = getattr(Losses, args.loss)(args,cfg)

    # Training
    def train(self,epoch):
        args = self.args
        cfg = self.cfg

        print('\nEpoch: %d' % epoch)
        self.net.train()
        ############### freeze encoder
        # for param in self.net.encoder.parameters():
        #     param.requires_grad = False
        # self.net.encoder.eval()
        train_loss = 0
        correctDist = 0
        total = 0
        torch.cuda.empty_cache()
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            an, se, pos, neg, targets = inputs[0].to(self.device),\
                                          inputs[1].to(self.device),\
                                          inputs[2].to(self.device), \
                                          inputs[3].to(self.device), \
                                          targets.to(self.device)
            # convert from original dataset label to known class label

            # targets = torch.Tensor([[1, 2, 3, 4, 0][x] for x in targets]).long().to(self.device)

            self.optimizer.zero_grad()

            outputs = self.net.train_forward(an, se, pos, neg)

            mainLoss,subLosses  = self.Loss(outputs, targets)

            if args.tensorboard and batch_idx % 3 == 0:
                self.writer.add_scalar('train/main_Loss', mainLoss.item(), batch_idx + epoch * len(self.trainloader))


            mainLoss.backward()

            self.optimizer.step()

            train_loss += mainLoss.item()

            # _, predicted = outputs[1].min(1)
            # predicted = predicted[targets!=-1]
            # total += targets[targets!=-1].size(0)
            # targets = targets[targets!=-1]
            # correctDist += predicted.eq(targets).sum().item()

            # _, predicted = outputs[0].max(1)
            #
            # total += targets.size(0)
            # correctDist += predicted.eq(targets).sum().item()

            total =1
            correctDist= 1
            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f center_loss: %.3f trip_loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), subLosses[0],subLosses[1], 100. * correctDist / total, correctDist, total))

            # print(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (train_loss / (batch_idx + 1), 100. * correctDist / total, correctDist, total))

        if args.tensorboard:
            acc = 100. * correctDist / total
            self.writer.add_scalar('train/accuracy', acc, epoch)

        if (epoch+1)%5==0 or (epoch+1)==cfg['training']['max_epoch']:

            save_name = ''
            state = {
                'net': self.net.state_dict(),
                'epoch': epoch,
            }
            print('Saving..')
            if not os.path.isdir('networks/weights/{}-{}'.format(args.dataset,args.name)):
                os.makedirs('networks/weights/{}-{}'.format(args.dataset,args.name))
            torch.save(state, 'networks/weights/{}-{}/'.format(args.dataset,args.name) + save_name + '{}_{}.pth'.format(epoch+1,args.trial))



if __name__=="__main__":

    ###################--------------------------- parameters

    parser = argparse.ArgumentParser(description='Open Set Classifier Training')
    parser.add_argument('--dataset', default='fiw', type=str, help='Dataset for training',
                        choices=['kfw1','kfw2','fiw'])
    parser.add_argument('--trial', default=0, type=int, help='Trial number, 0-4 provided')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from the checkpoint')
    parser.add_argument('--tensorboard', '-t', default=False, action='store_true', help='Plot on tensorboardX')
    # parser.add_argument('--name', default="HK_trip_tanhNet_7_type_baseline", type=str, help='Optional name for saving and tensorboard')
    # parser.add_argument('--loss', default="final_loss2", type=str, help='name of the loss used')
    parser.add_argument('--name', default="HK_trip_tanhNet_7type", type=str, help='Optional name for saving and tensorboard')
    parser.add_argument('--loss', default="final_loss3", type=str, help='name of the loss used')
    parser.add_argument('--real_sn', default=False, type=bool, help='whether test real scenario')
    parser.add_argument('--gpu', default=0, type=int, help='gpu number')
    parser.add_argument('--cs_num', default=0, type=int, help='gpu number')
    args = parser.parse_args()

    print('==> Preparing data..')
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)
    with open('datasets/config.json') as config_file:
        cfg = json.load(config_file)[args.dataset]

    if args.cs_num>0:
        i = args.cs_num - 1
        args.trial = i
        print('=====> cross validation {}'.format(i+1))

        start_epoch = 0

        # max_epoch = cfg['training']['max_epoch'] + start_epoch
        max_epoch = 60
        worker = get_work(args,cfg)
        for epoch in range(start_epoch, max_epoch):
            worker.train(epoch)
            # val(epoch)

        print('==> Finish traning..')
        print(vars(args))

    else:

        print('='*10,'run 5-cross-validation','='*10)
        for i in range(5):
            # i = args.cs_num - 1
            args.trial = i
            print('=====> cross validation {}'.format(i + 1))

            start_epoch = 0

            # max_epoch = cfg['training']['max_epoch'] + start_epoch
            max_epoch = 60
            worker = get_work(args, cfg)
            for epoch in range(start_epoch, max_epoch):
                worker.train(epoch)
                # val(epoch)

            print('==> Finish traning..')
            print(vars(args))

