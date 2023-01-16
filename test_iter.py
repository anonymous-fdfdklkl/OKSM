

import argparse
import json

import torchvision
import torchvision.transforms as tf
import torch
import torch.nn as nn
import logging


from utils.metrics import save_confusion_metric
from utils.utils import find_anchor_means, gather_outputs,get_network,get_test_dataloader
from sklearn import metrics as skmetrics
import utils.metrics as metrics
import scipy.stats as st
import numpy as np
from utils.loader3 import *
from sklearn.metrics import f1_score, fbeta_score
parser = argparse.ArgumentParser(description='Closed Set Classifier Training')
parser.add_argument('--dataset', default="fiw", type=str, help='Dataset for evaluation',
                    choices=['fiw'])
parser.add_argument('--num_trials', default=5, type=int, help='Number of trials to average results over')
parser.add_argument('--start_trial', default=0, type=int, help='Trial number to start eavaluation for?')


# N='fiw-HK_trip_CNN_point_Net_7type' ###  cnn-point250
# N='fiw-HK_trip_CNN_Net_7type'# attentionNet 200
# N='fiw-JLNet_7type'# JLNet 100
# N='fiw-close_7type'# softmax 100
# N='fiw-class_7type'# softmax*
# N='fiw-cac_7type'# CAC
# N='fiw-HK_trip_tanhNet_7type_hard'# HKMNet
# N='fiw-FG_adam'
# N='fiw-HK_trip_FG_adam'
N = 'fiw-HK_trip_CNN_Net_7type'
epoch = {'fiw-HK_trip_CNN_point_Net_7type':250,'fiw-HK_trip_CNN_Net_7type':60,'fiw-JLNet_7type':100,
         'fiw-close_7type':100,'fiw-class_7type':200,'fiw-cac_7type':150, 'fiw-HK_trip_tanhNet_7type':20,
         'fiw-HK_trip_tanhNet_7type_hard':40,'fiw-HK_trip_tanhNet_notanh':40,'fiw-HK_trip_tanhNet_different_7type':40,
         'fiw-FG_adam':10,'fiw-HK_trip_FG_adam':30,'fiw-HK_trip_tanhNet_7type_loss3':30, 'fiw-HK_trip_tanhNet_7type_loss4':30,
         'fiw-HK_trip_tanhNet_7type_loss5':40,'fiw-HK_trip_tanhNet_c_loss5':30,'fiw-HK_trip_tanhNet_7type_loss5_hard':30,
         'fiw-HK_trip_tanhNet_7type_loss3_hard':30,'fiw-HK_trip_tanhNet_7type_loss6':30,'fiw-HK_trip_tanhNet_7type_loss5_hard_old':30,
         'fiw-HK_trip_tanhNet_different':40

         }[N]


# N = 'class_4_type'# softmax* 200
# N='close_4_type'# softmax 100
# N='HK_trip_CNN_Net_4_type'# attentionNet 200
# N = 'cac_4type'# CACere
# N='HK_trip_CNN_point_Net_4type' ###  250
# N='JLNet_4_type'# JLNet 100
# N = 'fiw-JLNet_4type'# HKMNet
#
# epoch = {'fiw-class_4type':200,'fiw-close_4type':100,'fiw-HK_trip_CNN_Net_4type':250,
#          'fiw-cac_4type':100,'fiw-HK_trip_CNN_point_Net_4type':250,'fiw-JLNet_4type':100,
#          'fiw-HK_trip_tanhNet_4type':50,'fiw-HK_trip_tanhNet_4type_hard':50}[N]



parser.add_argument('--name', default=N, type=str, help='What iteration of gaussian fitting in open set training?')
parser.add_argument('--model', default=N, type=str, help='test model')
parser.add_argument('--real_sn', default=False, type=bool, help='whether test real scenario')
parser.add_argument('--trial', default=0, type=int, help='Trial number, 0-4 provided')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'



logger = logging.getLogger("test")
file_handler = logging.FileHandler('{}.txt'.format(N), "w")
stdout_handler = logging.StreamHandler()
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)
# stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
# file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger.setLevel(logging.INFO)

class print_logger(object):
    """docstring for ClassName"""
    def __init__(self,N):
        self.f = open('{}.txt'.format(N),"w")
        
    def __call__(self,str_out):
        print(str_out)
        self.f.write(str(str_out))
        self.f.write('\n')

    def close(self):
        self.f.close()


print_logger = print_logger(N)


# for num_iter in [10,15,20,25,30,35,40,45,50,55,60]:
for num_iter in [40,50,60]:
    epoch = num_iter
    logger.info("=="*20+"This is trained on epoch {}".format(num_iter)+"=="*20)
    

    all_accuracy = []
    all_auroc = []
    all_accuracy_with_neg = []
    all_f1 = []
    all_f1_1 = []
    all_f1_2 = []
    logger.info(vars(args))


    all_GT = []
    all_pred=[]
    name_dict={}

    all_k_precision=[]
    all_ukkin_precision=[]
    all_unk_precision=[]
    all_k_recall=[]
    all_ukkin_recall=[]
    all_unk_recall=[]
    all_k_f1=[]
    all_ukkin_f1=[]
    all_unk_f1=[]


    for trial_num in range(args.start_trial, args.start_trial + args.num_trials):
        logger.info('==> Preparing data for trial {}..'.format(trial_num))

        with open('datasets/config.json') as config_file:
            cfg = json.load(config_file)[args.dataset]

        cfg['batch_size'] = 6
        if '4_type' in args.name or '4type' in args.name:
            cfg['num_known_classes'] = 4
            cfg['num_classes'] = 4
        # Create dataloaders for training
        args.trial = trial_num
        # knownloader, unknownloader, testloader = get_test_dataloader(args, cfg)

        _, _, testloader_test  = fiw_loader(args, cfg).get_fiw_eval_loaders_img_new()

        _, _, testloader_eval  = fiw_loader(args, cfg).get_fiw_eval_loaders_img_new_get_thre()
        name = args.model
        if args.real_sn:
            name += 'real'

        ###################Closed Set Network Evaluation##################################################################
        logger.info('==> Building open set network for trial {}..'.format(trial_num))
        if "JLNet" in args.name:
            from networks.JLNet import JLNet_basic, JLNet_basic_7type

            if "4_type" in args.name or '4type' in args.name:

                net = JLNet_basic()
            else:
                net = JLNet_basic_7type()
        else:

            net = get_network(args, cfg)
        checkpoint = torch.load(
            'networks/weights/{}/{}_{}.pth'.format(args.model, epoch, args.trial))

        net = net.to(device)
        net_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in net_dict}
        # if 'close' in args.name or 'class' in args.name or 'JLNet' in args.name:
        net.load_state_dict(pretrained_dict)
        net.eval()
        # else:
        #     # if 'anchors' not in pretrained_dict.keys():
        #     #     pretrained_dict['anchors'] = checkpoint['net']['means']
        #     net.load_state_dict(pretrained_dict)
        #     net.eval()
        ###################################################### evaluate
        X = []
        y = []
        # neg_val_pos = []
        softmax = torch.nn.Softmax(dim=1)

        mapping = [0,1,2,3,4,5,6,7,8,9,10,-1]

        for i, data in enumerate(testloader_eval):
            images, labels,img1_pth, img2_pth = data
            # targets = labels.cuda()
            targets = torch.Tensor([mapping[x] for x in labels]).long().cuda()

            images = images.cuda()

            if 'class' in args.name:
                dis = net.similarity(images, False)
            else:
                dis = net.similarity(images)
            # scores = softmax(logits)

            X += dis.cpu().detach().tolist()
            y += targets.cpu().tolist()

        X = np.asarray(X)
        y = np.asarray(y)

        if '4_type' in args.name or '4type' in args.name:
            known_label = [True if i in [0, 1, 2, 3] else False for i in y]
            unknown_label = [True if i in [7, 8, 9, 10, -1] else False for i in y]
            kin_related_label = [True if i in [7, 8, 9, 10] else False for i in y]
            unknon_nonkin_label = [True if i in [-1] else False for i in y]
            name_dict = {'fiw-HK_trip_CNN_point_Net_4type': 'cnn_point', 'fiw-HK_trip_CNN_Net_4type': 'attention',
                                'fiw-JLNet_4type': 'jlnet',
                                'fiw-close_4type': 'atten+softmax', 'fiw-class_4type': 'atten_softmax*', 'fiw-cac_4type': 'cac',
                                'fiw-HK_trip_tanhNet_4type': 'hkmnet', 'ARPL_4_type': 'arpl','fiw-HK_trip_tanhNet_4type_hard':'4type_hkmnet_hard',
                                 'fiw-HK_trip_tanhNet_7type_loss3':'loss3', 'fiw-HK_trip_tanhNet_7type_loss4': 'loss4','fiw-HK_trip_tanhNet_7type_loss5': 'loss5',
                                'fiw-HK_trip_tanhNet_c_loss5':'c_loss5','fiw-HK_trip_tanhNet_7type_loss5_hard':'loss5_hard',
                                'fiw-HK_trip_tanhNet_7type_loss3_hard':'loss3_hard','fiw-HK_trip_tanhNet_7type_loss6':'loss6',
                         'fiw-HK_trip_tanhNet_7type_loss5_hard_old':'loss5_hard_old',
                         'fiw-HK_trip_tanhNet_different':'diff'
                         }
        else:
            known_label = [True if i in [0, 1, 2, 3, 4, 5, 6] else False for i in y]
            unknown_label = [True if i in [7, 8, 9, 10,-1] else False for i in y]
            kin_related_label = [True if i in [7, 8, 9, 10] else False for i in y]
            unknon_nonkin_label = [True if i in [-1] else False for i in y]
            name_dict = {'fiw-HK_trip_CNN_point_Net_7type': 'cnn_point', 'fiw-HK_trip_CNN_Net_7type': 'attention',
                                'fiw-JLNet_7type': 'jlnet',
                                'fiw-close_7type': 'atten+softmax', 'fiw-class_7type': 'atten_softmax*', 'fiw-cac_7type': 'cac',
                                'fiw-HK_trip_tanhNet_7type': 'hkmnet', 'ARPL': 'arpl',
                         'fiw-HK_trip_tanhNet_7type_hard':'7type_hkmnet_hard', 'fiw-HK_trip_tanhNet_notanh':'without_tanh',
                         'fiw-HK_trip_tanhNet_different_7type':'diff_fc','fiw-FG_adam':'fiw-FG_adam','fiw-HK_trip_FG_adam':'fiw-HK_trip_FG_adam',
                         'fiw-HK_trip_tanhNet_7type_loss3':'loss3', 'fiw-HK_trip_tanhNet_7type_loss4': 'loss4','fiw-HK_trip_tanhNet_7type_loss5': 'loss5',
                         'fiw-HK_trip_tanhNet_c_loss5':'c_loss5','fiw-HK_trip_tanhNet_7type_loss5_hard':'loss5_hard',
                         'fiw-HK_trip_tanhNet_7type_loss3_hard':'loss3_hard','fiw-HK_trip_tanhNet_7type_loss6':'loss6',
                         'fiw-HK_trip_tanhNet_7type_loss5_hard_old':'loss5_hard_old',
                         'fiw-HK_trip_tanhNet_different':'diff'}

        known_set = X[known_label]
        unk_set = X[unknown_label]
        ###
        unk_nonkin_set = X[unknon_nonkin_label]
        unk_kin_set = X[kin_related_label]


        if 'Sim' in args.name:
            inlow = False
        else:
            inlow = True

        #### 1) known vs unknown
        _, op_shr_knun = metrics.auroc(known_set, unk_set, in_low=inlow, show=False, sv_roc = False,name=[name,'known','unknown'],trial_num=trial_num)
        # print_logger('op_shr_knun is {}'.format(op_shr_knun))

        #### kin vs nonkin (within unknon sets)

        _, op_shr_kinnon = metrics.auroc(unk_kin_set, unk_nonkin_set, in_low=inlow, show=False, sv_roc = False,name=[name,'kin','non-kin'],trial_num=trial_num)
        # print_logger('op_shr_kinnon is {}'.format(op_shr_kinnon))



        ####################################### test
        X = []
        y = []
        # neg_val_pos = []
        softmax = torch.nn.Softmax(dim=1)

        mapping = [0,1,2,3,4,5,6,7,8,9,10,-1]

        for i, data in enumerate(testloader_test):
            images, labels,img1_pth, img2_pth = data
            # targets = labels.cuda()
            targets = torch.Tensor([mapping[x] for x in labels]).long().cuda()

            images = images.cuda()

            if 'class' in args.name:
                dis = net.similarity(images, False)
            else:
                dis = net.similarity(images)
            # scores = softmax(logits)

            X += dis.cpu().detach().tolist()
            y += targets.cpu().tolist()

        X = np.asarray(X)
        y = np.asarray(y)

        if '4_type' in args.name or '4type' in args.name:
            known_label = [True if i in [0, 1, 2, 3] else False for i in y]
            unknown_label = [True if i in [7, 8, 9, 10, -1] else False for i in y]
            kin_related_label = [True if i in [7, 8, 9, 10] else False for i in y]
            unknon_nonkin_label = [True if i in [-1] else False for i in y]
            name_dict = {'fiw-HK_trip_CNN_point_Net_4type': 'cnn_point', 'fiw-HK_trip_CNN_Net_4type': 'attention',
                                'fiw-JLNet_4type': 'jlnet',
                                'fiw-close_4type': 'atten+softmax', 'fiw-class_4type': 'atten_softmax*', 'fiw-cac_4type': 'cac',
                                'fiw-HK_trip_tanhNet_4type': 'hkmnet', 'ARPL_4_type': 'arpl','fiw-HK_trip_tanhNet_4type_hard':'4type_hkmnet_hard',
                                 'fiw-HK_trip_tanhNet_7type_loss3':'loss3', 'fiw-HK_trip_tanhNet_7type_loss4': 'loss4','fiw-HK_trip_tanhNet_7type_loss5': 'loss5',
                                'fiw-HK_trip_tanhNet_c_loss5':'c_loss5','fiw-HK_trip_tanhNet_7type_loss5_hard':'loss5_hard',
                                'fiw-HK_trip_tanhNet_7type_loss3_hard':'loss3_hard','fiw-HK_trip_tanhNet_7type_loss6':'loss6',
                         'fiw-HK_trip_tanhNet_7type_loss5_hard_old': 'loss5_hard_old',
                         'fiw-HK_trip_tanhNet_different': 'diff'
                         }
        else:
            known_label = [True if i in [0, 1, 2, 3, 4, 5, 6] else False for i in y]
            unknown_label = [True if i in [7, 8, 9, 10,-1] else False for i in y]
            kin_related_label = [True if i in [7, 8, 9, 10] else False for i in y]
            unknon_nonkin_label = [True if i in [-1] else False for i in y]
            name_dict = {'fiw-HK_trip_CNN_point_Net_7type': 'cnn_point', 'fiw-HK_trip_CNN_Net_7type': 'attention',
                                'fiw-JLNet_7type': 'jlnet',
                                'fiw-close_7type': 'atten+softmax', 'fiw-class_7type': 'atten_softmax*', 'fiw-cac_7type': 'cac',
                                'fiw-HK_trip_tanhNet_7type': 'hkmnet', 'ARPL': 'arpl',
                         'fiw-HK_trip_tanhNet_7type_hard':'7type_hkmnet_hard', 'fiw-HK_trip_tanhNet_notanh':'without_tanh',
                         'fiw-HK_trip_tanhNet_different_7type':'diff_fc','fiw-FG_adam':'fiw-FG_adam','fiw-HK_trip_FG_adam':'fiw-HK_trip_FG_adam',
                         'fiw-HK_trip_tanhNet_7type_loss3':'loss3', 'fiw-HK_trip_tanhNet_7type_loss4': 'loss4','fiw-HK_trip_tanhNet_7type_loss5': 'loss5',
                         'fiw-HK_trip_tanhNet_c_loss5':'c_loss5','fiw-HK_trip_tanhNet_7type_loss5_hard':'loss5_hard',
                         'fiw-HK_trip_tanhNet_7type_loss3_hard':'loss3_hard','fiw-HK_trip_tanhNet_7type_loss6':'loss6',
                         'fiw-HK_trip_tanhNet_7type_loss5_hard_old':'loss5_hard_old',
                         'fiw-HK_trip_tanhNet_different':'diff'}

        known_set = X[known_label]
        unk_set = X[unknown_label]
        ###
        unk_nonkin_set = X[unknon_nonkin_label]
        unk_kin_set = X[kin_related_label]


        if 'Sim' in args.name:
            inlow = False
        else:
            inlow = True

        #### 1) known vs unknown
        auroc_knun, _ = metrics.auroc(known_set, unk_set, in_low=inlow, show=False, sv_roc = True,name=[name,'known','unknown'],trial_num=trial_num)
        logger.info('op_shr_knun is {}'.format(op_shr_knun))

        #### kin vs nonkin (within unknon sets)

        auroc_kinon, _ = metrics.auroc(unk_kin_set, unk_nonkin_set, in_low=inlow, show=False, sv_roc = True,name=[name,'kin','non-kin'],trial_num=trial_num)
        logger.info('op_shr_kinnon is {}'.format(op_shr_kinnon))




        GT = np.zeros(len(y))
        GT[kin_related_label]=1
        GT[unknon_nonkin_label]=2
        pred = np.ones(len(y))
        if 'Sim' in args.name:
            pred[X >= op_shr_knun] = 0
            pred[X <= op_shr_kinnon ] = 2
        else:
            pred[X<=op_shr_knun] =0
            pred[X>=op_shr_kinnon]=2





        acc = np.sum(GT==pred)/len(y)

        all_GT =np.concatenate((all_GT,GT))
        all_pred=np.concatenate((all_pred,pred))

        save_confusion_metric(GT,pred,name_dict[args.name]+'{}'.format(trial_num),label=[0,1,2],pth = args.model)


        if not os.path.exists('./acc/{}/'.format(trial_num)):
            os.makedirs('./acc/{}/'.format(trial_num))

        with open('./acc/{}/{}_acc_{}.pickle'.format(trial_num,args.name,trial_num), 'wb') as ff:
            pickle.dump(pred, ff)

        F1 = f1_score(GT, pred, average='macro')

        print('acc:{}'.format(acc))
        gt_lb = GT
        pred_lb = pred

        k_tp = np.sum((gt_lb == pred_lb)[gt_lb == 0])
        k_fp = len(pred_lb[pred_lb == 0]) - k_tp
        k_fn = len(gt_lb[gt_lb == 0]) - k_tp

        ukkin_tp = np.sum((gt_lb == pred_lb)[gt_lb == 1])
        ukkin_fp = len(pred_lb[pred_lb == 1]) - ukkin_tp
        ukkin_fn = len(gt_lb[gt_lb == 1]) - ukkin_tp

        unk_tp = np.sum((gt_lb == pred_lb)[gt_lb == 2])
        unk_fp = len(pred_lb[pred_lb == 2]) - unk_tp
        unk_fn = len(gt_lb[gt_lb == 2]) - unk_tp

        ### precision = tp/(tp+fp) : correct/predict
        k_precision = k_tp / (k_tp + k_fp+1e-16)
        ukkin_precision = ukkin_tp / (ukkin_tp + ukkin_fp+1e-16)
        unk_precision = unk_tp / (unk_tp + unk_fp+1e-16)

        ### recall = tp/(tp+fn): correct/total_target

        k_recall = k_tp / (k_tp + k_fn+1e-16)
        ukkin_recall = ukkin_tp / (ukkin_tp + ukkin_fn+1e-16)
        unk_recall = unk_tp / (unk_tp + unk_fn+1e-16)

        #####F1 : 2*precision*recall/(precision+recall)
        k_f1 = 2 * k_precision * k_recall / (k_precision + k_recall+1e-16)
        ukkin_f1 = 2 * ukkin_precision * ukkin_recall / (ukkin_precision + ukkin_recall+1e-16)
        unk_f1 = 2 * unk_precision * unk_recall / (unk_precision + unk_recall+1e-16)

        logger.info('Precision:=> known:{:.4f}, unknown_kin-related:{:.4f}, unknown-non-kin:{:.4f}'.
              format(k_precision, ukkin_precision, unk_precision))
        # print(str_out)
        # f.write(str_out)

        logger.info('Recall:=> known:{:.4f}, unknown_kin-related:{:.4f}, unknown-non-kin:{:.4f}'.
              format(k_recall, ukkin_recall, unk_recall))
        # print(str_out)
        # f.write(str_out)

        logger.info('F1:=> known:{:.4f}, unknown_kin-related:{:.4f}, unknown-non-kin:{:.4f}'.
              format(k_f1, ukkin_f1, unk_f1))
        # print(str_out)
        # f.write(str_out)

        logger.info('ACC:{:.4f},AUC-known-unknon:{:.4f},AUC-kin-nonkin:{:.4f}, F1:{:.4f}'.format(acc,auroc_knun,auroc_kinon,F1))
        # print(str_out)
        # f.write(str_out)

        all_k_precision.append(k_precision)
        all_ukkin_precision.append(ukkin_precision)
        all_unk_precision.append(unk_precision)
        all_k_recall.append(k_recall)
        all_ukkin_recall.append(ukkin_recall)
        all_unk_recall.append(unk_recall)
        all_k_f1.append(k_f1)
        all_ukkin_f1.append(ukkin_f1)
        all_unk_f1.append(unk_f1)


        all_accuracy.append(acc)
        all_auroc.append(auroc_knun)
        all_accuracy_with_neg.append(auroc_kinon)
        all_f1.append(F1)



    logger.info('='*10+' average '+'='*10)
    logger.info(args.model)
    logger.info(name_dict[args.model])
    logger.info('average ACC:{:.4f},AUC-known-unknon:{:.4f},AUC-kin-nonkin:{:.4f}, F1:{:.4f}'.format(np.mean(all_accuracy),np.mean(all_auroc),np.mean(all_accuracy_with_neg),np.mean(all_f1)))
    logger.info('Primary:Precision:{:.4f}, Recall:{:.4f}, F1:{:.4f}'.format(np.mean(all_k_precision),np.mean(all_k_recall),np.mean(all_k_f1)))
    logger.info('Secondary:Precision:{:.4f}, Recall:{:.4f}, F1:{:.4f}'.format(np.mean(all_ukkin_precision),np.mean(all_ukkin_recall),np.mean(all_ukkin_f1)))
    logger.info('Non-kin:Precision:{:.4f}, Recall:{:.4f}, F1:{:.4f}'.format(np.mean(all_unk_precision),np.mean(all_unk_recall),np.mean(all_unk_f1)))
    #### 1) known vs unknown
# print_logger.close()

