import argparse
import json

import torchvision
import torchvision.transforms as tf
import torch
import torch.nn as nn



from utils.utils import find_anchor_means, gather_outputs,get_network,get_test_dataloader
from sklearn import metrics as skmetrics
import utils.metrics as metrics
import scipy.stats as st
import numpy as np
from utils.loader3 import *
from sklearn.metrics import f1_score, fbeta_score
parser = argparse.ArgumentParser(description='Closed Set Classifier Training')
parser.add_argument('--dataset', default="fiw", type=str, help='Dataset for evaluation',
					choices=['kfw1','kfw2','fiw'])
parser.add_argument('--num_trials', default=1, type=int, help='Number of trials to average results over')
parser.add_argument('--start_trial', default=2, type=int, help='Trial number to start evaluation for?')

# N='HK_trip_KATfinal_hard'
# N='FG_adam'
N='HK_trip_tanhNet_7type_loss5_hard'
parser.add_argument('--epoch', default=[40], type=int, help="test model's epoch")
parser.add_argument('--name', default=N, type=str, help='What iteration of gaussian fitting in open set training?')
parser.add_argument('--model', default=N, type=str, help='test model')

parser.add_argument('--real_sn', default=False, type=bool, help='whether test real scenario')
parser.add_argument('--trial', default=0, type=int, help='Trial number, 0-4 provided')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

for epochs in args.epoch:
	print('model:{}'.format(args.name))
	print('epoch:{}'.format(epochs))
	all_accuracy = []
	all_auroc = []
	all_accuracy_with_neg = []
	all_f1 = []
	all_f1_1 = []
	all_f1_2 = []
	print(vars(args))
	for trial_num in range(args.start_trial, args.start_trial + args.num_trials):
		print('==> Preparing data for trial {}..'.format(trial_num))
		with open('datasets/config.json') as config_file:
			cfg = json.load(config_file)[args.dataset]

		# Create dataloaders for training
		args.trial = trial_num
		# knownloader, unknownloader, testloader = get_test_dataloader(args, cfg)

		knownloader, unknownloader, testloader  = fiw_loader(args, cfg).get_fiw_eval_loaders_new()
		name = args.model + '_' + args.dataset
		if args.real_sn:
			name += 'real'

		###################Closed Set Network Evaluation##################################################################
		print('==> Building open set network for trial {}..'.format(trial_num))
		net = get_network(args,cfg)
		checkpoint = torch.load(
			'networks/weights/{}-{}/{}_{}.pth'.format(args.dataset, args.model, epochs, args.trial))

		net = net.to(device)
		net.load_state_dict(checkpoint['net'])
		net.eval()

		X = []
		y = []
		# neg_val_pos = []
		softmax = torch.nn.Softmax(dim=1)

		mapping = [0,1,2,3,4,5,6,7,8,9,10,-1]

		for i, data in enumerate(testloader):
			images, labels = data
			# targets = labels.cuda()
			targets = torch.Tensor([mapping[x] for x in labels]).long().cuda()

			images = images.cuda()

			dis = net.similarity(images)
			# scores = softmax(logits)

			X += dis.cpu().detach().tolist()
			y += targets.cpu().tolist()

		X = np.asarray(X)
		y = np.asarray(y)

		if ('4_type' in args.name) or ('4type' in args.name):
			known_label = [True if i in [0, 1, 2, 3] else False for i in y]
			unknown_label = [True if i in [7, 8, 9, 10, -1] else False for i in y]
			kin_related_label = [True if i in [7, 8, 9, 10] else False for i in y]
			unknon_nonkin_label = [True if i in [-1] else False for i in y]
		else:
			known_label = [True if i in [0, 1, 2, 3, 4, 5, 6] else False for i in y]
			unknown_label = [True if i in [7, 8, 9, 10,-1] else False for i in y]
			kin_related_label = [True if i in [7, 8, 9, 10] else False for i in y]
			unknon_nonkin_label = [True if i in [-1] else False for i in y]

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
		auroc_knun, op_shr_knun = metrics.auroc(known_set, unk_set, in_low=inlow, show=False, name=[name,'known','unknown'])


		#### kin vs nonkin (within unknon sets)

		auroc_kinon, op_shr_kinnon = metrics.auroc(unk_kin_set, unk_nonkin_set, in_low=inlow, show=False, name=[name,'kin','non-kin'])


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
		k_precision = k_tp / (k_tp + k_fp)
		ukkin_precision = ukkin_tp / (ukkin_tp + ukkin_fp)
		unk_precision = unk_tp / (unk_tp + unk_fp)

		### recall = tp/(tp+fn): correct/total_target

		k_recall = k_tp / (k_tp + k_fn)
		ukkin_recall = ukkin_tp / (ukkin_tp + ukkin_fn)
		unk_recall = unk_tp / (unk_tp + unk_fn)

		#####F1 : 2*precision*recall/(precision+recall)
		k_f1 = 2 * k_precision * k_recall / (k_precision + k_recall)
		ukkin_f1 = 2 * ukkin_precision * ukkin_recall / (ukkin_precision + ukkin_recall)
		unk_f1 = 2 * unk_precision * unk_recall / (unk_precision + unk_recall)

		print('Precision:=> known:{:.4f}, unknown_kin-related:{:.4f}, unknown-non-kin:{:.4f}'.
			  format(k_precision, ukkin_precision, unk_precision))
		print('Recall:=> known:{:.4f}, unknown_kin-related:{:.4f}, unknown-non-kin:{:.4f}'.
			  format(k_recall, ukkin_recall, unk_recall))
		print('F1:=> known:{:.4f}, unknown_kin-related:{:.4f}, unknown-non-kin:{:.4f}'.
			  format(k_f1, ukkin_f1, unk_f1))
		print('ACC:{:.4f},AUC-known-unknon:{:.4f},AUC-kin-nonkin:{:.4f}, F1:{:.4f}'.format(acc,auroc_knun,auroc_kinon,F1))
		all_accuracy.append(acc)
		all_auroc.append(auroc_knun)
		all_accuracy_with_neg.append(auroc_kinon)
		all_f1.append(F1)

	print('='*10,'average','='*10)
	print('average ACC:{:.4f},AUC-known-unknon:{:.4f},AUC-kin-nonkin:{:.4f}, F1:{:.4f}'.format(np.mean(all_accuracy),np.mean(all_auroc),np.mean(all_accuracy_with_neg),np.mean(all_f1)))



	
