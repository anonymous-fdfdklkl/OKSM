"""
	Metrics used to evaluate performance.

	Dimity Miller, 2020
"""
import numpy as np
import sklearn.metrics
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, plot_confusion_matrix
import pandas as pd
import seaborn as sn

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, fbeta_score
import pickle

def accuracy(x, gt):
    predicted = np.argmax(x, axis=1)
    total = len(gt)
    acc = np.sum(predicted == gt) / total
    return acc



#
# def auroc(inData, outData, in_low=True, show=False, name=''):
#     inDataMin = np.max(inData, 1)
#     outDataMin = np.max(outData, 1)
#
#     allData = np.concatenate((inDataMin, outDataMin))
#     labels = np.concatenate((np.zeros(len(inDataMin)), np.ones(len(outDataMin))))
#     fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, allData, pos_label=in_low)
#
#     y = tpr - fpr
#     youden_index = np.argmax(y)
#     op_thr = thresholds[youden_index]
#     ############ add
#     auc = sklearn.metrics.roc_auc_score(labels, allData)
#     if show:
#         plt.figure()
#         plt.plot(fpr, tpr, lw=2, label=' (AUC: {:0.2f})'.format(auc))
#         plt.plot(fpr[youden_index], tpr[youden_index], 'bo')
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.0])
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title('ROC of known vs unknown')
#         plt.legend(loc="best")
#         # plt.show()
#         plt.savefig('roc-{}.png'.format(name))
#
#     ################
#     # op_thr = optimal_thresh(tpr,fpr,thresholds)
#
#     return sklearn.metrics.auc(fpr, tpr), op_thr
#
#
# def auroc(inData, outData, in_low=True, show=False, name=None):
#     """
#
#     :param inData:
#     :param outData:
#     :param in_low: True: the measurements results of indata is smaller than outdata
#     :param show:
#     :param name:
#     :return:
#     """
#     allData = np.concatenate((inData, outData))
#     if in_low:
#         labels = np.concatenate((np.zeros(len(inData)), np.ones(len(outData))))
#     else:
#         labels = np.concatenate((np.ones(len(inData)), np.zeros(len(outData))))
#     fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, allData)
#
#     y = tpr - fpr
#     youden_index = np.argmax(y)
#     op_thr = thresholds[youden_index]
#     ############ add
#     auc = sklearn.metrics.roc_auc_score(labels, allData)
#     if show:
#         plt.figure()
#         plt.plot(fpr, tpr, lw=2, label=' (AUC: {:0.2f})'.format(auc))
#         plt.plot(fpr[youden_index], tpr[youden_index], 'bo')
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.0])
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title('ROC of {} vs {}'.format(name[1],name[2]))
#         plt.legend(loc="best")
#         # plt.show()
#         plt.savefig('roc-{}.png'.format(name[0]+'_'+name[1]+'_'+name[2]))
#
#
#     ################
#     # op_thr = optimal_thresh(tpr,fpr,thresholds)
#
#     with open('{}.pickle'.format(name[0]+'_'+name[1]+'_'+name[2]),'wb') as ff:
#         pickle.dump([fpr,tpr,auc],ff)
#
#
#     return sklearn.metrics.auc(fpr, tpr), op_thr
def auroc(inData, outData, in_low=True, show=False, name=None,trial_num=''):
    """

    :param inData:
    :param outData:
    :param in_low: True: the measurements results of indata is smaller than outdata
    :param show:
    :param name:
    :return:
    """
    allData = np.concatenate((inData, outData))
    if in_low:
        labels = np.concatenate((np.zeros(len(inData)), np.ones(len(outData))))
    else:
        labels = np.concatenate((np.ones(len(inData)), np.zeros(len(outData))))
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, allData)

    y = tpr - fpr
    youden_index = np.argmax(y)
    op_thr = thresholds[youden_index]
    ############ add
    auc = sklearn.metrics.roc_auc_score(labels, allData)
    if show:
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=' (AUC: {:0.2f})'.format(auc))
        plt.plot(fpr[youden_index], tpr[youden_index], 'bo')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC of {} vs {}'.format(name[1],name[2]))
        plt.legend(loc="best")
        # plt.show()
        plt.savefig('roc-{}.png'.format(name[0]+'_'+name[1]+'_'+name[2]))


    ################
    # op_thr = optimal_thresh(tpr,fpr,thresholds)

    with open('{}.pickle'.format(name[0]+'_'+name[1]+'_'+name[2]+'_'+str(trial_num)),'wb') as ff:
        pickle.dump([fpr,tpr,auc],ff)


    return sklearn.metrics.auc(fpr, tpr), op_thr

def accuracy_with_neg(x, gt, threshold=None, sv_cm=False, name=''):
    if threshold is None:
        pass
    else:
        predicted = np.argmax(x, axis=1)
        max_value = np.max(x, 1)
        predicted[max_value < threshold] = -1
        total = len(gt)
        acc = np.sum(predicted == gt) / total

        f1_1 = f1_score(gt, predicted, average='macro')
        f1_2 = f1_score(gt, predicted, labels=[0, 1, 2, 3], average='macro')
        if sv_cm:
            save_confusion_metric(gt, predicted, [-1, 0, 1, 2, 3], name)
        return acc, f1_1, f1_2



def optimal_thresh(tpr, fpr, thresh):
    y = tpr - fpr
    youden_index = np.argmax(y)
    op_thr = thresh[youden_index]
    return op_thr


def save_confusion_metric(gt, pred, label, name):
    plt.figure()

    confu_m = confusion_matrix(gt, pred, labels=label, normalize='true')
    df_cm = pd.DataFrame(confu_m, ['Non', 'F-D', 'F-S', 'M-D', 'M-S'], ['Non', 'F-D', 'F-S', 'M-D', 'M-S'])

    # confu_m = confusion_matrix(gt, pred, labels=[0, 1, 2, 3], normalize='true')
    # df_cm = pd.DataFrame(confu_m, [ 'F-D', 'F-S', 'M-D', 'M-S'], [ 'F-D', 'F-S', 'M-D', 'M-S'])

    sn.set(font_scale=0.8)  # for label size
    sn.heatmap(df_cm, vmin=0, vmax=1, cmap='Blues', annot=True, annot_kws={"size": 16})  # font size
    # plt.show()
    # plt.savefig('stage3-{}_test1{}_hm{}.png'.format(number,stage3_joint_config.kin_config.model_name, '_avg'))
    plt.savefig('hm-{}.png'.format(name))