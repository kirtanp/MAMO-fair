from loss.loss_class import Loss

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BCELoss(Loss):
    def __init__(self, name='BCELoss', weight_vector=None):
        super().__init__(name)
        self.weight_vector = weight_vector

    def compute_loss(self, y_true, y_pred):
        y_true = y_true[:, [0]]
        if self.weight_vector is not None:
            batch_weights = torch.zeros_like(y_true, device=device)
            batch_weights[y_true==0] = self.weight_vector[0]
            batch_weights[y_true==1] = self.weight_vector[1]
            loss_fn = nn.BCELoss(weight=batch_weights)
        else:
            loss_fn = nn.BCELoss()
        return(loss_fn(y_pred, y_true))


class MSELoss(Loss):
    def __init__(self, name='MSELoss'):
        super().__init__(name)

    def compute_loss(self, y_true, y_pred):
        y_true = y_true[:, [0]]
        loss_fn = nn.MSELoss()
        return(loss_fn(y_pred, y_true))


class DPLoss(Loss):
    def __init__(self, name='DPLoss', weight_vector=None, \
                 threshold=0.5, attribute_index=1, reg_lambda=0.1,
                 reg_type='tanh', reg_beta=0.0, good_value=1):

        super().__init__(name)
        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.idx = attribute_index
        self.reg_lambda = reg_lambda
        self.reg_beta = reg_beta
        self.reg_type = reg_type
        self.good_value = good_value
        
    def _differentiable_round(self, x):
        x = x.float()
        return torch.tanh(3*(x - self.threshold))/2 + 0.5
    
    def _DP_torch(self, y_true, y_pred, reg):
        if(reg=='tanh'):
            y_pred = self._differentiable_round(y_pred)
            if(self.good_value):
                y_pred = y_pred[y_pred > self.threshold]
            else:
                y_pred = y_pred[y_pred < self.threshold]
        elif(reg=='ccr'):
            if(self.good_value):
                y_pred = y_pred[y_pred > self.threshold]
            else:
                y_pred = y_pred[y_pred < self.threshold]
        elif(y_pred=='linear'):
            y_pred = y_pred
        total = y_true.shape[0] + 1e-7
        return(torch.sum(y_pred)/total)
    
    def compute_loss(self, y_true, y_pred):
        a = y_true[:, self.idx]
        y_true = y_true[:, [0]]
        y_pred = torch.clamp(y_pred, 1e-7, 1-1e-7)
        
        y_pred_0 = y_pred[a==0]
        y_true_0 = y_true[a==0]

        y_pred_1 = y_pred[a==1]
        y_true_1 = y_true[a==1]
        
        DP_0 = self._DP_torch(y_true_0, y_pred_0, self.reg_type) 
        DP_1 = self._DP_torch(y_true_1, y_pred_1, self.reg_type) 

        if self.weight_vector is not None:
            batch_weights = torch.zeros_like(y_true, device=device)
            batch_weights[y_true==0] = self.weight_vector[0]
            batch_weights[y_true==1] = self.weight_vector[1]
            loss_fn = nn.BCELoss(weight=batch_weights)
        else:
            loss_fn = nn.MSELoss()
        
        return( torch.abs((DP_0 - DP_1)) + self.reg_lambda*loss_fn(y_pred, y_true) +\
         self.reg_beta*torch.mean(y_pred**2)) 

class FPRLoss(Loss):
    def __init__(self, name='FPRLoss', weight_vector=None, \
                 threshold=0.5, attribute_index=1, reg_lambda=0.1,
                 reg_type='tanh'):

        super().__init__(name)
        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.idx = attribute_index
        self.reg_lambda = reg_lambda
        self.reg_type = reg_type
        
    def _differentiable_round(self, x):
        x = x.float()
        return torch.tanh(3*(x - self.threshold))/2 + 0.5
    
    def _FPR_torch(self, y_true, y_pred, reg):
        if(reg=='tanh'):
            y_pred = self._differentiable_round(y_pred)
            y_pred = y_pred[(y_true==0) & (y_pred > self.threshold)]
        elif(reg=='ccr'):
            y_pred = y_pred[(y_true==0) & (y_pred > self.threshold)]
        elif(y_pred=='linear'):
            y_pred = y_pred
        total_negatives = torch.sum(y_true==0) + 1e-7
        return(torch.sum(y_pred)/total_negatives)
    
    def compute_loss(self, y_true, y_pred):
        a = y_true[:, self.idx]
        y_true = y_true[:, [0]]
        y_pred = torch.clamp(y_pred, 1e-7, 1-1e-7)
        
        y_pred_0 = y_pred[a==0]
        y_true_0 = y_true[a==0]

        y_pred_1 = y_pred[a==1]
        y_true_1 = y_true[a==1]
        
        FPR_0 = self._FPR_torch(y_true_0, y_pred_0, self.reg_type) 
        FPR_1 = self._FPR_torch(y_true_1, y_pred_1, self.reg_type) 

        if self.weight_vector is not None:
            batch_weights = torch.zeros_like(y_true, device=device)
            batch_weights[y_true==0] = self.weight_vector[0]
            batch_weights[y_true==1] = self.weight_vector[1]
            loss_fn = nn.BCELoss(weight=batch_weights)
        else:
            loss_fn = nn.MSELoss()
        
        return( torch.abs((FPR_0 - FPR_1)) + self.reg_lambda*loss_fn(y_pred, y_true)) 


class FNRLoss(Loss):
    def __init__(self, name='FNRLoss', weight_vector=None, \
                 threshold=0.5, attribute_index=1, reg_lambda=0.1,
                 reg_type='tanh'):

        super().__init__(name)
        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.idx = attribute_index
        self.reg_lambda = reg_lambda
        
    def _differentiable_round(self, x):
        x = x.float()
        return torch.tanh(3*(x - self.threshold))/2 + 0.5
    
    def _FNR_torch(self, y_true, y_pred, reg):
        if(reg=='tanh'):
            y_pred = self._differentiable_round(y_pred)
            y_pred = y_pred[(y_true==1) & (y_pred < self.threshold)]
        elif(reg=='ccr'):
            y_pred = y_pred[(y_true==1) & (y_pred < self.threshold)]
        elif(y_pred=='linear'):
            y_pred = y_pred
        total_positives = torch.sum(y_true==1) + 1e-7
        return(torch.sum(y_pred)/total_positives)
    
    def compute_loss(self, y_true, y_pred):
        a = y_true[:, self.idx]
        y_true = y_true[:, [0]]
        y_pred = torch.clamp(y_pred, 1e-7, 1-1e-7)
        
        y_pred_0 = y_pred[a==0]
        y_true_0 = y_true[a==0]

        y_pred_1 = y_pred[a==1]
        y_true_1 = y_true[a==1]
        
        FNR_0 = self._FNR_torch(y_true_0, y_pred_0, self.reg_type)
        FNR_1 = self._FNR_torch(y_true_1, y_pred_1, self.reg_type)

        if self.weight_vector is not None:
            batch_weights = torch.zeros_like(y_true, device=device)
            batch_weights[y_true==0] = self.weight_vector[0]
            batch_weights[y_true==1] = self.weight_vector[1]
            loss_fn = nn.BCELoss(weight=batch_weights)
        else:
            loss_fn = nn.MSELoss()
        
        return(torch.abs((FNR_0 - FNR_1)) + self.reg_lambda*loss_fn(y_pred, y_true) )


class TNRLoss(Loss):
    def __init__(self, name='TNRLoss', weight_vector=None, \
                 threshold=0.5, attribute_index=1, reg_lambda=0.1,
                 reg_type='tanh'):

        super().__init__(name)
        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.idx = attribute_index
        self.reg_lambda = reg_lambda
        self.reg_type = reg_type
        
    def _differentiable_round(self, x):
        x = x.float()
        return torch.tanh(5*(x - self.threshold))/2 + 0.5
    
    def _TNR_torch(self, y_true, y_pred, reg):
        if(reg=='tanh'):
            y_pred = self._differentiable_round(y_pred)
            y_pred = y_pred[(y_true==0) & (y_pred < self.threshold)]
        elif(reg=='ccr'):
            y_pred = y_pred[(y_true==0) & (y_pred < self.threshold)]
        elif(y_pred=='linear'):
            y_pred = y_pred
        total_negatives = torch.sum(y_true==0) + 1e-7
        return(torch.sum(y_pred)/total_negatives)
    
    def compute_loss(self, y_true, y_pred):
        a = y_true[:, self.idx]
        y_true = y_true[:, [0]]
        y_pred = torch.clamp(y_pred, 1e-7, 1-1e-7)
        
        y_pred_0 = y_pred[a==0]
        y_true_0 = y_true[a==0]

        y_pred_1 = y_pred[a==1]
        y_true_1 = y_true[a==1]
        
        TNR_0 = self._TNR_torch(y_true_0, y_pred_0, self.reg_type)
        TNR_1 = self._TNR_torch(y_true_1, y_pred_1, self.reg_type)

        if self.weight_vector is not None:
            batch_weights = torch.zeros_like(y_true, device=device)
            batch_weights[y_true==0] = self.weight_vector[0]
            batch_weights[y_true==1] = self.weight_vector[1]
            loss_fn = nn.BCELoss(weight=batch_weights)
        else:
            loss_fn = nn.MSELoss()
        
        return(torch.abs((TNR_0 - TNR_1)) + self.reg_lambda*loss_fn(y_pred, y_true))


class TPRLoss(Loss):
    def __init__(self, name='TPRLoss', weight_vector=None, \
                 threshold=0.5, attribute_index=1, reg_lambda=0.1,
                 reg_type='tanh'):

        super().__init__(name)
        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.idx = attribute_index
        self.reg_lambda = reg_lambda
        self.reg_type = reg_type
        
    def _differentiable_round(self, x):
        x = x.float()
        return torch.tanh(5*(x - self.threshold))/2 + 0.5
    
    def _TPR_torch(self, y_true, y_pred, reg):
        if(reg=='tanh'):
            y_pred = self._differentiable_round(y_pred)
            y_pred = y_pred[(y_true==1) & (y_pred > self.threshold)]
        elif(reg=='ccr'):
            y_pred = y_pred[(y_true==1) & (y_pred > self.threshold)]
        elif(y_pred=='linear'):
            y_pred = y_pred
        total_positives = torch.sum(y_true==1) + 1e-7
        return(torch.sum(y_pred)/total_positives)
    
    def compute_loss(self, y_true, y_pred):
        a = y_true[:, self.idx]
        y_true = y_true[:, [0]]
        y_pred = torch.clamp(y_pred, 1e-7, 1-1e-7)
        
        y_pred_0 = y_pred[a==0]
        y_true_0 = y_true[a==0]

        y_pred_1 = y_pred[a==1]
        y_true_1 = y_true[a==1]
        
        TPR_0 = self._TPR_torch(y_true_0, y_pred_0, self.reg_type)
        TPR_1 = self._TPR_torch(y_true_1, y_pred_1, self.reg_type)

        if self.weight_vector is not None:
            batch_weights = torch.zeros_like(y_true, device=device)
            batch_weights[y_true==0] = self.weight_vector[0]
            batch_weights[y_true==1] = self.weight_vector[1]
            loss_fn = nn.BCELoss(weight=batch_weights)
        else:
            loss_fn = nn.MSELoss()
        
        return(torch.abs((TPR_0 - TPR_1)) + self.reg_lambda*loss_fn(y_pred, y_true))



class CFLoss(Loss):
    def __init__(self, name='CounterfactualLoss', sen_attributes_idx=[1], \
                 reg_lambda=0.1, weight_vector=None, augmentation=False):

        super().__init__(name)
        self.reg_lambda = reg_lambda
        self.needs_model = True
        self.sen_attributes_idx = sen_attributes_idx
        self.weight_vector = weight_vector
        self.augmentation = augmentation

    def _logit(self, x):
        eps = torch.tensor(1e-7)
        x = x.float()
        x = torch.clamp(x, eps, 1-eps)
        return torch.log(x/(1-x)) 

    def _get_subsets(self, s):     
        x = len(s)
        subset_list = []
        for i in range(1, 1 << x):
            subset_list.append([s[j] for j in range(x) if (i & (1 << j))])
        return(subset_list)
        
    def _get_counterfactuals(self, x):
        i = self.sen_attributes_idx[0]
        x1 = x.clone()
        x1[:,[-i]] = 1 - x1[:,[-i]]
        return(x1)

    def compute_loss(self, y_true, y_pred, x, model):
            
        y_pred_x = self._logit(y_pred)
        x_new = self._get_counterfactuals(x)
        y_pred_new = model(x_new)
        y_pred_new_logit = self._logit(y_pred_new)
        pred_diff = torch.abs(y_pred_x - y_pred_new)

        if self.weight_vector is not None:
            batch_weights = torch.zeros_like(y_true, device=device)
            batch_weights[y_true==0] = self.weight_vector[0]
            batch_weights[y_true==1] = self.weight_vector[1]
            loss_fn = nn.BCELoss(weight=batch_weights)
        else:
            loss_fn = nn.MSELoss()

        if(self.augmentation):
        	loss = loss_fn(y_pred, y_true) + loss_fn(y_pred_new, y_true)
        else:
            loss = loss_fn(y_pred, y_true)
        
        return(self.reg_lambda*torch.mean(pred_diff) + loss)


class addLosses(Loss):
    def __init__(self, name='addLosses', loss_list=[], \
                 loss_weights=[]):

        super().__init__(name)
        self.loss_list = loss_list
        self.loss_weights = loss_weights
        self.needs_model = True
    
    def compute_loss(self, y_true, y_pred, x, model):
        final_loss = []
        for alpha, loss_fn in zip(self.loss_weights, self.loss_list):
            if(loss_fn.needs_model):
                loss = alpha * loss_fn.compute_loss(y_true, y_pred, x, model)
            else:
                loss = alpha * loss_fn.compute_loss(y_true, y_pred)
            final_loss.append(loss)
        
        final_loss = torch.stack(final_loss)
        total_loss = torch.sum(final_loss)
        return(total_loss)

