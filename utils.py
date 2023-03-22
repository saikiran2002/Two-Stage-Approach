import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import balanced_accuracy_score as BAS

class Utils:
    def __init__(self,history,pred_labels,test_labels,CLASSES):
        self.history = history
        self.pred_labels = pred_labels
        self.test_labels = test_labels
        self.CLASSES = CLASSES

    def plot_ROC_curves(self):
        fig, ax = plt.subplots(1, 3, figsize = (30, 5))
        ax = ax.ravel()

        for i, metric in enumerate(["acc", "auc", "loss"]):
            ax[i].plot(self.history.history[metric])
            ax[i].plot(self.history.history["val_" + metric])
            ax[i].set_title("Model {}".format(metric))
            ax[i].set_xlabel("Epochs")
            ax[i].set_ylabel(metric)
            ax[i].legend(["train", "val"])
        plt.show()

    def Classification_report(self):
        def roundoff(arr):
            arr[np.argwhere(arr != arr.max())] = 0
            arr[np.argwhere(arr == arr.max())] = 1
            return arr

        for labels in self.pred_labels:
            labels = roundoff(labels)

        print(classification_report(self.test_labels, self.pred_labels, target_names=self.CLASSES))

    def Confusion_matrix(self, title=""):
        pred_ls = np.argmax(self.pred_labels, axis=1)
        test_ls = np.argmax(self.test_labels, axis=1)

        conf_arr = confusion_matrix(test_ls, pred_ls)

        plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

        ax = sns.heatmap(conf_arr, cmap='Greens', annot=True, fmt='d', xticklabels=self.CLASSES, yticklabels=self.CLASSES)

        plt.title(title)
        plt.xlabel('Prediction')
        plt.ylabel('Truth')
        plt.show(ax)

    def balanced_accuracy_score(self):
        return "Balanced Accuracy Score: {} %".format(round(BAS(self.test_ls, self.pred_ls) * 100, 2))
    def Mathews_correlation_coefficient(self):
        return "Matthew's Correlation Coefficient: {} %".format(round(MCC(self.test_ls, self.pred_ls) * 100, 2))
