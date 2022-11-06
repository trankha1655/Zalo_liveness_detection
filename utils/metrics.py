
import numpy as np
class Classify_Metrics:
    def __init__(self, numClass):
        self.confusionMatrix = np.zeros((numClass, numClass))
        self.numClass = numClass

    def accuracy(self, confusionMatrix= None):

        if confusionMatrix is not None:
            confusionMatrix = self.confusionMatrix
        #acc = (TP + TN) / (TP + TN + FP + FN)
        acc = np.diag(confusionMatrix).sum() /  confusionMatrix.sum()
        return acc

    def precision(self, confusionMatrix = None):
        # precision = TP / (TP+FP)
        if confusionMatrix is not None:
            confusionMatrix = self.confusionMatrix

        precision = np.diag(confusionMatrix) / np.sum(confusionMatrix, axis=1)

        return precision

    def recall(self, confusionMatrix = None):
        # recall = TP / (TP+FN)
        if confusionMatrix is not None:
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
        temp = self.genConfusionMatrix(target, predict)
        self.confusionMatrix += temp

        return self.get_metrics(target, predict)

    def get_metrics(self, target = None, predict = None):

        if target is not None:
            temp = self.genConfusionMatrix(target, predict)
        else:
            temp = None
        p = self.precision(temp)
        r = self.recall(temp)
        acc = self.accuracy(temp)
        f1 = self.f1_score(p, r)
        return acc, p, r, f1

    




    

    


        