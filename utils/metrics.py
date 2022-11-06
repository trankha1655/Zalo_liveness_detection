
import numpy as np
from prettytable import PrettyTable
def eval_metric(Accuracy, Precision, Recall, F1):
    class_dict_df ={0: 'Fake', 1: 'Real'}
    
    #Accuracy, Precision, Recall, F1 = metrics.get_metrics()

    # Precision = metrics.p
    # Recall = metrics.r
    # F1 = 2 * ((Precision * Recall) / (Precision + Recall))
    # Accuracy = metrics.acc
    print(Accuracy, Precision, Recall, F1)
    Precision = np.around(Precision, decimals=4)
    Recall = np.around(Recall, decimals=4)
    Accuracy = np.around(Accuracy, decimals=4)
    F1 = np.around(F1, decimals=4)

    P = np.sum(Precision[:]) / len(Precision[:])
    R = np.sum(Recall[:]) / len(Recall[:])
    F = np.sum(F1[:]) / len(F1[:])
    A = np.sum(Accuracy[:]) / len(Accuracy[:])

    t = PrettyTable(['label_index', 'label_name', 'Accuracy', 'Precision', 'Recall', 'F1'])
    for key in class_dict_df:
        t.add_row([key, class_dict_df[key], Accuracy[key], Precision[key], Recall[key], F1[key]])
    print(t.get_string(title="Validation results"))
    print('\nAcc:{:.4f}        Precision:{:.4f}'
          '\nRecall:{:.4f}     F1:{:.4f}'
         .format(A, P, R, F))
          

    result = args.save_seg_dir + '/results.txt'
    with open(result, 'w') as f:
        f.write(str(t.get_string(title="Validation results")))
        f.write('\nAcc:{:.4f}        Precision:{:.4f}'
                '\nRecall:{:.4f}     F1:{:.4f}'
                .format(A, P, R, F))

    return A, P, R, F



class Classify_Metrics:
    def __init__(self, numClass):
        self.confusionMatrix = np.zeros((numClass, numClass))
        self.numClass = numClass
        self.p = 0
        self.acc = 0
        self.r = 0
        self.f1 = 0

    def accuracy(self, confusionMatrix= None):

        if confusionMatrix is None:
            confusionMatrix = self.confusionMatrix
        #acc = (TP + TN) / (TP + TN + FP + FN)
        acc = np.diag(confusionMatrix) /  confusionMatrix.sum()
        return acc

    def precision(self, confusionMatrix = None):
        # precision = TP / (TP+FP)
        if confusionMatrix is None:
            confusionMatrix = self.confusionMatrix

        precision = np.diag(confusionMatrix) / np.sum(confusionMatrix, axis=1)

        return precision

    def recall(self, confusionMatrix = None):
        # recall = TP / (TP+FN)
        if confusionMatrix is None:
            confusionMatrix = self.confusionMatrix

        recall = np.diag(confusionMatrix) / np.sum(confusionMatrix, axis=0)
        return recall

    def f1_score(self, precision, recall):
        f1 = precision * recall * 2
        f1/= (precision + recall)
        return f1

    def get_ConfusionMatrix(self, target, predict):

        temp = self.numClass * target + predict

        count = np.bincount(temp, minlength=self.numClass**2)
        confusionMatrix = count.reshape((self.numClass, self.numClass))
        return confusionMatrix

    def addBatch(self, target, predict):
        
        assert target.shape == predict.shape 
        temp = self.get_ConfusionMatrix(target, predict)
        self.confusionMatrix += temp

        return self.get_metrics(target, predict)

    def get_metrics(self, target = None, predict = None):

        if target is not None:
            temp = self.get_ConfusionMatrix(target, predict)
        else:
            temp = None
        p = self.precision(temp)
        r = self.recall(temp)
        acc = self.accuracy(temp)
        f1 = self.f1_score(p, r)
        if temp is None:
            return eval_metric(acc, p, r, f1)
        
        return np.sum(acc)

    




    

    


        