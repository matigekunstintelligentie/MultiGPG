import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from csv import writer
import argparse
import os
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sympy import *
import torch.nn as nn
import torch.optim as optim
import re
import copy


class FeedforwardNN(nn.Module):
    def __init__(self, input_n, n_neurons, n_layers):
        super(FeedforwardNN, self).__init__()
        self.begin = nn.Linear(input_n, n_neurons)
        l =[]
        for i in range(n_layers-2):
            l.append(nn.Linear(n_neurons, n_neurons))
        self.mid = nn.ModuleList(l)
        self.end = nn.Linear(n_neurons, 1)
    
    def forward(self, x):
        x = torch.nn.functional.elu(self.begin(x))
        for module in self.mid:
            x = torch.nn.functional.elu(module(x))
        x = self.end(x)
        return x

class NNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        # batch size, num_epochs, n_layers, n_inputs
        for k in kwargs:
            setattr(self, k, kwargs[k])




    def ls_terms(self, y, p):
        y_mean = y.mean();
        p_mean = p.mean();
        
        y_res = y - y_mean;
        p_res = p - p_mean;
        
        denominator = p_res.square().sum();
        
        if (denominator == 0 or torch.isnan(denominator) or torch.isinf(torch.abs(denominator))):
          slope = 0.;
          interc = y_mean;
        else:
          slope = (y_res*p_res).sum() / denominator;
          interc = y_mean - slope * p_mean;
        
        return interc, slope
        

    def fit(self, X, y):

        X_train_tensor = torch.tensor(X, dtype=torch.float32)
        y_train_tensor = torch.tensor(y, dtype=torch.float32)
        input_n = X_train_tensor.shape[1]
        model = FeedforwardNN(input_n, self.n, self.n_layers)

        criterion = nn.MSELoss()
        
        interc = torch.tensor(0., requires_grad=True)
        slope = torch.tensor(1., requires_grad=True)
        
        optimizer = optim.Adam(list(model.parameters()), lr=0.001)
        best_loss = np.finfo(np.float32).max
        best_model = copy.deepcopy(model)
        for epoch in range(self.num_epochs):
            with torch.no_grad():  
                y_pred = model(X_train_tensor)
                interc.data, slope.data = self.ls_terms(y_train_tensor, y_pred)
                
            permutation = torch.randperm(X_train_tensor.size()[0])
            for i in range(0,X_train_tensor.size()[0], self.batch_size):
                # Backward pass and optimization
                optimizer.zero_grad()
                 
                indices = permutation[i:i+self.batch_size]
                
                
                # Forward pass
                y_pred = model(X_train_tensor[indices])
                
                loss = criterion(y_pred.squeeze(), y_train_tensor[indices])
                
                loss.backward()
                optimizer.step()

            loss = criterion(interc + slope * model(X_train_tensor), y_train_tensor)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model = copy.deepcopy(model)
            if epoch%100==0:
                print("Epoch {} {}% done - loss {} R2 {}, mse {}, var {}".format(epoch,100*epoch/self.num_epochs,best_loss, 1. - (best_loss/np.var(y)), criterion(model(X_train_tensor), y_train_tensor), np.var(y)))
                    
                    
        
        model = best_model
        
        symbs = ["var_{:02d}".format(i) for i in range(input_n)]
        sympy_symbols = symbols(" ".join(symbs))      
        
        layers = ["x","model.begin"] + ["model.mid[{}]".format(i) for i in range(input_n-2)] + ["model.end"]
        
        def recursion(layer, idx, model):
            if layer==0:
                return symbs[idx] #"x[:,{}]".format(idx)
            stri = str(eval(layers[layer] + ".bias[{}].item()".format(idx)))
            for nxt_idx, weight in enumerate(eval(layers[layer] + ".weight[{}].detach().numpy()".format(idx))):
                stri += "+{}*max({},0.0)".format(weight,recursion(layer-1, nxt_idx, model))
            return stri
        
        
        stri = recursion(4,0,model)
        backup_stri = stri
        occurence_max = backup_stri.count("max")
        occurence_plus = backup_stri.count("+")
        occurence_times = backup_stri.count("*")
        for symb in symbs:
            backup_stri = backup_stri.replace(symb,"var")
        occurence_x = backup_stri.count("var")
        occurence_floats = len(re.findall("\d+\.\d+", backup_stri))
        

        print(occurence_max + occurence_plus + occurence_times + occurence_x + occurence_floats)
        
        stri = str(simplify(stri))
        
        backup_stri = stri
        occurence_max = backup_stri.count("Max")
        occurence_plus = backup_stri.count("+")
        occurence_times = backup_stri.count("*")
        for symb in symbs:
            backup_stri = backup_stri.replace(symb,"var")
        occurence_times = backup_stri.count("x")
        occurence_floats = len(re.findall("\d+\.\d+", backup_stri))
        
        print("fin", occurence_max + occurence_plus + occurence_times + occurence_x + occurence_floats)
        
        stri = stri.replace("max", "np.max")
        for idx, symb in enumerate(symbs):
            stri = stri.replace(symb, "x[:,{}]".format(idx))
        
        self.model = eval("lambda x:{}".format(stri))

    def predict(self, X, model=None):
        return self.model(X)
    
    def complexity(self):
        pass
        
    
est = NNRegressor(n=32,n_layers=4,batch_size=256, num_epochs=40000)
data = pd.read_csv('./dataset/{}.csv'.format("bike"))
X_train, X_test, y_train, y_test = train_test_split(data.drop(['target'], axis=1), data['target'], test_size=0.25, random_state=1)
est.fit(X_train.values, y_train.values)

print((np.mean(est.predict(X_train.values)-y_train.values)**2))
print(1.-(np.mean(est.predict(X_train.values)-y_train.values)**2)/np.var(y_test.values))
print(1.-(np.mean(est.predict(X_test.values)-y_test.values)**2)/np.var(y_test.values))
