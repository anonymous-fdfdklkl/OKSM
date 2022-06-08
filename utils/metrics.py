import os.path

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



def auroc(inData, outData, in_low=True, show=False,sv_roc = True, name=None,trial_num=''):
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
    if sv_roc:
        with open('./roc/{}/{}.pickle'.format(trial_num,name[0]+'_'+name[1]+'_'+name[2]+'_'+str(trial_num)),'wb') as ff:
            pickle.dump([fpr,tpr,auc],ff)


    return sklearn.metrics.auc(fpr, tpr), op_thr



def save_confusion_metric(gt, pred, name, label=[0,1,2],pth = ''):
    plt.figure()

    confu_m = confusion_matrix(gt, pred, labels=label, normalize='true')
    # df_cm = pd.DataFrame(confu_m, ['Primary-Kin', 'Secondary-Kin', 'Non-Kin'], ['Primary-Kin', 'Secondary-Kin', 'Non-Kin'])
    df_cm = pd.DataFrame(confu_m, ['P', 'S', 'N'],
                         ['P', 'S', 'N'])

    sn.set(font_scale=2.5)  # for label size
    sn.heatmap(df_cm, vmin=0, vmax=1, cmap='Blues', annot=True, annot_kws={"size": 26})  # font size
    # plt.show()
    # plt.savefig('stage3-{}_test1{}_hm{}.png'.format(number,stage3_joint_config.kin_config.model_name, '_avg'))
    if not os.path.exists('./hm/{}/'.format(pth)):
        os.makedirs('./hm/{}/'.format(pth))
    plt.savefig('./hm/{}/hm-{}.png'.format(pth,name))