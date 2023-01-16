import pickle
import glob
import sklearn.metrics
from matplotlib import pyplot as plt

picks = glob.glob('*.pickle')

picks_kn_nokin = [item for item in picks if item.split('_')[-1]=='unknown.pickle']
picks_kn_unk =[item for item in picks if item.split('_')[-1]=='non-kin.pickle']



dict_kn_unk={}
for paths in picks_kn_unk:
    with open(paths, 'rb') as file:
        dict_kn_unk[paths.split('.')[0]] =pickle.load(file)



plt.figure()
for roc_name in dict_kn_unk:
    fpr = dict_kn_unk[roc_name][0]
    tpr = dict_kn_unk[roc_name][1]
    auc = dict_kn_unk[roc_name][2]
    plt.plot(fpr, tpr, lw=2, label=' (AUC: {:0.2f})'.format(auc))
# plt.plot(fpr[youden_index], tpr[youden_index], 'bo')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of {} vs {}'.format('known','unknown'))
plt.legend(loc="best")
# plt.show()
plt.savefig('roc-{}.png'.format('known_unknown'))