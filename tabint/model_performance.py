
import tabint
from tabint.utils import *
from tabint.dataset import *
from tabint.visual import *
from tabint.learner import *
from sklearn import metrics
from sklearn.metrics import *


class model_performace:
    def __init__(self, y_true, y_pred, **kargs):
        self.y_true, self.y_pred = y_true, y_pred
        self.data = data

    @classmethod
    def from_learner(cls, learner, ds, **kargs):
        y_pred = learner.predict(ds.x_val)
        y_true = ds.y_val
        return cls.from_series(y_true, y_pred, **kargs)

    @classmethod
    def from_series(cls, y_true, y_pred, **kargs):
        res = cls.calculate(y_true, y_pred, **kargs)
        return cls(y_true, y_pred, *res)
    
    @staticmethod
    def calculate(y_true, y_pred, **kargs):
        return y_true, y_pred

    def plot(self, **kargs): None


class actual_vs_predict(model_performace):
    def __init__(self, actual, predict, df, data):
        self.actual, self.predict = actual, predict
        self.df, self.data = df, data
        
    @classmethod
    def from_learner(cls, learner, ds):
        actual = ds.y_val
        predict = learner.predict(ds.x_val)
        return cls.from_series(actual, predict, ds.x_val)

    @classmethod
    def from_series(cls, actual, predict, df):
        data = cls.calculate(actual, predict)
        return cls(actual, predict, df, data)
    
    @staticmethod
    def calculate(actual, predict):
        data = pd.DataFrame({'actual':actual, 'predict':predict, 'mse': (actual-predict)**2})
        return ResultDF(data, 'mse')
    
    def plot(self, hue = None, num = 100, **kagrs):
        if hue is not None: hue = self.df[hue]
        concat = np.concatenate([self.actual, self.predict])
        plot_scatter(self.actual, self.predict, xlabel='actual', ylabel='predict', hue=hue)
        plot_bisectrix(np.min(concat), np.max(concat), num)
        if hue is not None: plot_legend()


class ReceiverOperatingCharacteristic(model_performace):
    def __init__(self, fpr, tpr, data, roc_auc):
        self.fpr, self.tpr, self.data, self.roc_auc = fpr, tpr, data, roc_auc

    @classmethod
    def from_series(cls, y_true, y_pred): 
        fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
        data = pd.DataFrame.from_dict({'threshold': threshold, 'tpr':tpr, 'fpr':fpr})
        roc_auc = metrics.auc(fpr, tpr)
        return cls(fpr, tpr, data, roc_auc)
    
    def plot(self): plot_roc_curve(self.fpr, self.tpr, self.roc_auc)


class PrecisionRecall(model_performace):
    def __init__(self, precision, recall, threshold):
        self.precision,self.recall,self.threshold = precision, recall, threshold

    @classmethod
    def from_series(cls, y_true, y_pred):
        precision, recall, threshold = cls.calculate(y_true, y_pred)
        return cls(precision, recall, threshold)

    @staticmethod
    def calculate(y_true, y_pred, **kargs): return precision_recall_curve(y_true, y_pred)

    def plot(self, **kwargs):
        plot_line([self.threshold]*2, 
                  [self.precision[:-1], self.recall[:-1]], 
                  ['precision', 'recall'], 
                  ["r--", "b-"], 
                  xlabel = "threshold", **kwargs)