from sklearn.metrics import confusion_matrix, accuracy_score
from metric.metric import Metric
import numpy as np

class FPRParity(Metric):

    def __init__(self, name='FPR_parity', weight_vector=None, \
                 threshold=0.5, attribute_index=1):

        super().__init__(name)
        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.idx = attribute_index

    def _FPR_np(self, y_true, y_pred):
        epsilon = 1e-7
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        #print("confusion", tn, fp, fn, tp)
        return(fp.astype(np.float32)/(fp.astype(np.float32) + tn.astype(np.float32) + epsilon))


    def evaluate(self, y_true, y_pred, x, model):
        a = y_true[:,[self.idx]]
        y_true = y_true[:,[0]]

        y_pred = np.rint(y_pred)

        y_pred_0 = y_pred[a==0]
        y_true_0 = y_true[a==0]

        y_pred_1 = y_pred[a==1]
        y_true_1 = y_true[a==1]
        
        FPR_0 = self._FPR_np(y_true_0, y_pred_0)
        FPR_1 = self._FPR_np(y_true_1, y_pred_1)
        
        FPR = min(FPR_0, FPR_1)/max(FPR_0, FPR_1)
        return(np.round(FPR, 20))


class TPRParity(Metric):

    def __init__(self, name='TPR_parity', weight_vector=None, \
                 threshold=0.5, attribute_index=1):

        super().__init__(name)
        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.idx = attribute_index

    def _TPR_np(self, y_true, y_pred):
        epsilon = 1e-7
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        #print("confusion", tn, fp, fn, tp)
        return(tp.astype(np.float32)/(tp.astype(np.float32) + fn.astype(np.float32) + epsilon))


    def evaluate(self, y_true, y_pred, x, model):
        a = y_true[:,[self.idx]]
        y_true = y_true[:,[0]]

        y_pred = np.rint(y_pred)

        y_pred_0 = y_pred[a==0]
        y_true_0 = y_true[a==0]

        y_pred_1 = y_pred[a==1]
        y_true_1 = y_true[a==1]
        
        TPR_0 = self._TPR_np(y_true_0, y_pred_0)
        TPR_1 = self._TPR_np(y_true_1, y_pred_1)
        
        TPR = min(TPR_0, TPR_1)/max(TPR_0, TPR_1)
        return(np.round(TPR, 20))

class TNRParity(Metric):

    def __init__(self, name='TNR_parity', weight_vector=None, \
                 threshold=0.5, attribute_index=1):

        super().__init__(name)
        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.idx = attribute_index

    def _TNR_np(self, y_true, y_pred):
        epsilon = 1e-7
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        #print("confusion", tn, fp, fn, tp)
        return(tn.astype(np.float32)/(tn.astype(np.float32) + fp.astype(np.float32) + epsilon))


    def evaluate(self, y_true, y_pred, x, model):
        a = y_true[:,[self.idx]]
        y_true = y_true[:,[0]]

        y_pred = np.rint(y_pred)

        y_pred_0 = y_pred[a==0]
        y_true_0 = y_true[a==0]

        y_pred_1 = y_pred[a==1]
        y_true_1 = y_true[a==1]
        
        TNR_0 = self._TNR_np(y_true_0, y_pred_0)
        TNR_1 = self._TNR_np(y_true_1, y_pred_1)
        
        TNR = min(TNR_0, TNR_1)/max(TNR_0, TNR_1)
        return(np.round(TNR, 20))

class FNRParity(Metric):

    def __init__(self, name='FNR_parity', weight_vector=None, \
                 threshold=0.5, attribute_index=1):

        super().__init__(name)
        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.idx = attribute_index

    def _FNR_np(self, y_true, y_pred):
        epsilon = 1e-7
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        #print("confusion", tn, fp, fn, tp)
        return(fn.astype(np.float32)/(fn.astype(np.float32) + tp.astype(np.float32) + epsilon))


    def evaluate(self, y_true, y_pred, x, model):
        a = y_true[:,[self.idx]]
        y_true = y_true[:,[0]]

        y_pred = np.rint(y_pred)

        y_pred_0 = y_pred[a==0]
        y_true_0 = y_true[a==0]

        y_pred_1 = y_pred[a==1]
        y_true_1 = y_true[a==1]
        
        FNR_0 = self._FNR_np(y_true_0, y_pred_0)
        FNR_1 = self._FNR_np(y_true_1, y_pred_1)
        
        FNR = min(FNR_0, FNR_1)/max(FNR_0, FNR_1)
        return(np.round(FNR, 20))

class DemParity(Metric):

    def __init__(self, name='DP_parity', weight_vector=None, \
                 threshold=0.5, attribute_index=1):

        super().__init__(name)
        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.idx = attribute_index

    def _DP_np(self, y_true, y_pred):
        epsilon = 1e-7
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        pr = (tp.astype(np.float32)+fp) / (tp+fp+tn+fn)
        return(pr + epsilon)


    def evaluate(self, y_true, y_pred, x, model):
        a = y_true[:,[self.idx]]
        y_true = y_true[:,[0]]

        y_pred = np.rint(y_pred)

        y_pred_0 = y_pred[a==0]
        y_true_0 = y_true[a==0]

        y_pred_1 = y_pred[a==1]
        y_true_1 = y_true[a==1]
        
        DP_0 = self._DP_np(y_true_0, y_pred_0)
        DP_1 = self._DP_np(y_true_1, y_pred_1)
        
        DP = min(DP_0, DP_1)/max(DP_0, DP_1)
        return(np.round(DP, 20))


class FPRDiff(Metric):

    def __init__(self, name='FPR_diff', weight_vector=None, \
                 threshold=0.5, attribute_index=1):

        super().__init__(name)
        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.idx = attribute_index

    def _FPR_np(self, y_true, y_pred):
        epsilon = 1e-7
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        #print("confusion", tn, fp, fn, tp)
        return(fp.astype(np.float32)/(fp.astype(np.float32) + tn.astype(np.float32) + epsilon))


    def evaluate(self, y_true, y_pred, x, model):
        a = y_true[:,[self.idx]]
        y_true = y_true[:,[0]]

        y_pred = np.rint(y_pred)

        y_pred_0 = y_pred[a==0]
        y_true_0 = y_true[a==0]

        y_pred_1 = y_pred[a==1]
        y_true_1 = y_true[a==1]
        
        FPR_0 = self._FPR_np(y_true_0, y_pred_0)
        FPR_1 = self._FPR_np(y_true_1, y_pred_1)
        
        diff = np.abs(FPR_0 - FPR_1)
        return(1 - diff)


class TPRDiff(Metric):

    def __init__(self, name='TPR_diff', weight_vector=None, \
                 threshold=0.5, attribute_index=1):

        super().__init__(name)
        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.idx = attribute_index

    def _TPR_np(self, y_true, y_pred):
        epsilon = 1e-7
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        #print("confusion", tn, fp, fn, tp)
        return(tp.astype(np.float32)/(tp.astype(np.float32) + fn.astype(np.float32) + epsilon))


    def evaluate(self, y_true, y_pred, x, model):
        a = y_true[:,[self.idx]]
        y_true = y_true[:,[0]]

        y_pred = np.rint(y_pred)

        y_pred_0 = y_pred[a==0]
        y_true_0 = y_true[a==0]

        y_pred_1 = y_pred[a==1]
        y_true_1 = y_true[a==1]
        
        TPR_0 = self._TPR_np(y_true_0, y_pred_0)
        TPR_1 = self._TPR_np(y_true_1, y_pred_1)
        
        diff = np.abs(TPR_0 - TPR_1)
        return(1 - diff)

class TNRDiff(Metric):

    def __init__(self, name='TNR_diff', weight_vector=None, \
                 threshold=0.5, attribute_index=1):

        super().__init__(name)
        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.idx = attribute_index

    def _TNR_np(self, y_true, y_pred):
        epsilon = 1e-7
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        #print("confusion", tn, fp, fn, tp)
        return(tn.astype(np.float32)/(tn.astype(np.float32) + fp.astype(np.float32) + epsilon))


    def evaluate(self, y_true, y_pred, x, model):
        a = y_true[:,[self.idx]]
        y_true = y_true[:,[0]]

        y_pred = np.rint(y_pred)

        y_pred_0 = y_pred[a==0]
        y_true_0 = y_true[a==0]

        y_pred_1 = y_pred[a==1]
        y_true_1 = y_true[a==1]
        
        TNR_0 = self._TNR_np(y_true_0, y_pred_0)
        TNR_1 = self._TNR_np(y_true_1, y_pred_1)
        
        diff = np.abs(TNR_0 - TNR_1)
        return(1 - diff)

class FNRDiff(Metric):

    def __init__(self, name='FNR_diff', weight_vector=None, \
                 threshold=0.5, attribute_index=1):

        super().__init__(name)
        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.idx = attribute_index

    def _FNR_np(self, y_true, y_pred):
        epsilon = 1e-7
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        #print("confusion", tn, fp, fn, tp)
        return(fn.astype(np.float32)/(fn.astype(np.float32) + tp.astype(np.float32) + epsilon))


    def evaluate(self, y_true, y_pred, x, model):
        a = y_true[:,[self.idx]]
        y_true = y_true[:,[0]]

        y_pred = np.rint(y_pred)

        y_pred_0 = y_pred[a==0]
        y_true_0 = y_true[a==0]

        y_pred_1 = y_pred[a==1]
        y_true_1 = y_true[a==1]
        
        FNR_0 = self._FNR_np(y_true_0, y_pred_0)
        FNR_1 = self._FNR_np(y_true_1, y_pred_1)
        
        diff = abs(FNR_0 - FNR_1)
        return(1 - diff)

class DPDiff(Metric):

    def __init__(self, name='DP_diff', weight_vector=None, \
                 threshold=0.5, attribute_index=1, good_value=1):
        super().__init__(name)
        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.idx = attribute_index
        self.good_value = good_value

    def _DP_np(self, y_true, y_pred):
        epsilon = 1e-7
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        if(self.good_value):
            pr = (tp.astype(np.float32)+fp) / (tp+fp+tn+fn)
        else:
        	pr = (tn.astype(np.float32)+fn) / (tp+fp+tn+fn)
        return(pr + epsilon)


    def evaluate(self, y_true, y_pred, x, model):
        a = y_true[:,[self.idx]]
        y_true = y_true[:,[0]]

        y_pred = np.rint(y_pred)

        y_pred_0 = y_pred[a==0]
        y_true_0 = y_true[a==0]

        y_pred_1 = y_pred[a==1]
        y_true_1 = y_true[a==1]
        
        DP_0 = self._DP_np(y_true_0, y_pred_0)
        DP_1 = self._DP_np(y_true_1, y_pred_1)
        
        diff = abs(DP_0 - DP_1)
        return(1 - diff)

class CFGap(Metric):

    def __init__(self, name='CF_gap', weight_vector=None, \
                 threshold=0.5, sen_attributes_idx=[1]):

        super().__init__(name)
        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.sen_attributes_idx = sen_attributes_idx

    def _get_counterfactuals(self, x):
        i = self.sen_attributes_idx[0]
        x1 = x.clone()
        x1[:,[-i]] = 1 - x1[:,[-i]]
        return(x1)


    def evaluate(self, y_true, y_pred, x, model):
        x_new = self._get_counterfactuals(x)
        y_pred_new = model(x_new)
        y_pred_new = y_pred_new.cpu().detach().numpy()
        pred_diff = np.abs(y_pred - y_pred_new)
        cf_gap = np.mean(pred_diff)

        return(1-cf_gap)

        


class Accuracy(Metric):
    def evaluate(self, y_true, y_pred, x, model):
        y_pred = np.rint(y_pred)
        return(accuracy_score(y_true[:,[0]], y_pred))