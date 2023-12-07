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
from sklearn.preprocessing import StandardScaler as SS

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

parser = argparse.ArgumentParser()
parser.add_argument('--repetition', type=int)
args = parser.parse_args()
repetition = args.repetition

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

for dataset in ["tower", "air", "concrete", "tower","dowchemical"]:
    # Load data from a CSV file using pandas
    data = pd.read_csv('./dataset/{}.csv'.format(dataset))
    for n in [2,8,16]:       
        print(dataset, n)
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data.drop(['target'], axis=1), data['target'], test_size=0.25, random_state=repetition+1)
        
        s = SS()
        X_train = s.fit_transform(X_train)
        
        # Convert the data into PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        

        
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
        
        y_train_var = torch.var(y_train_tensor)
        y_test_var = torch.var(y_test_tensor)
        
        # Define the neural network architecture
        class FeedforwardNN(nn.Module):
            def __init__(self, n):
                super(FeedforwardNN, self).__init__()
                with torch.no_grad():
                    self.omega0 = 30
                    self.layer1 = nn.Linear(X_train_tensor.shape[1], n)
                    self.layer1.weight.uniform_(-self.omega0 / X_train_tensor.shape[1], 
                                                 self.omega0 / X_train_tensor.shape[1])   
                    self.layer2 = nn.Linear(n, n)
                    self.layer2.weight.uniform_(-np.sqrt(6 / n), np.sqrt(6 / n))  
                    
                    self.layer3 = nn.Linear(n, n)
                    self.layer3.weight.uniform_(-np.sqrt(6 / n), np.sqrt(6 / n)) 
                    
                    self.layer4 = nn.Linear(n, 1)
                    self.layer4.weight.uniform_(-np.sqrt(6 / n), np.sqrt(6 / n)) 
            
            def forward(self, x):
                x = torch.sin(self.layer1(x))
                x = torch.sin(self.layer2(x))
                x = torch.sin(self.layer3(x))
                x = self.layer4(x)
                return x



        # Initialize the model
        model = FeedforwardNN(n)
        print(count_parameters(model))
        
        slope = torch.tensor(1., requires_grad=True)
        interc = torch.tensor(0., requires_grad=True)

        
        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(list(model.parameters()) + [slope, interc], lr=0.01,  weight_decay=0.001)
        
        # Train the model
        best_test_loss = -999999999999999.
        best_train_loss = -999999999999999.
        num_epochs = 10000
        for epoch in range(num_epochs):

            
            permutation = torch.randperm(X_train_tensor.size()[0])
            

            for i in range(0,X_train_tensor.size()[0], 16):
                indices = permutation[i:i+16]
                # Forward pass
                #y_pred = interc + slope * model(X_train_tensor[indices])
                y_pred = model(X_train_tensor[indices] + torch.rand_like(X_train_tensor[indices])*0.05)
                loss = criterion(y_pred.squeeze(), y_train_tensor[indices])
                
# =============================================================================
#                 l1_lambda = 0.001
#                 l1_norm = sum(torch.linalg.norm(p, 1) for p in model.parameters())
# 
#                 loss = loss + l1_lambda * l1_norm
#                 
# =============================================================================
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            

        
            with torch.no_grad():
                y_pred_test = model(X_test_tensor)
                y_pred_train = model(X_train_tensor)
                test_loss = 1. - (criterion(interc + slope*y_pred_test.squeeze(), y_test_tensor)/y_test_var).item()
                train_loss = 1. - (criterion(interc + slope*y_pred_train.squeeze(), y_train_tensor)/y_train_var).item()

                y_mean = y_train_tensor.mean()
                p_mean = y_pred_train.mean()

                
                y_res = y_train_tensor.unsqueeze(1) - y_mean
                p_res = y_pred_train - p_mean
                
                
                denom = p_res.square().sum()
                
                if denom==0 or torch.isnan(denom) or torch.isinf(denom):
                    slope_tmp = torch.tensor(0.)
                    interc_tmp = y_mean
                else:
                    slope_tmp = (y_res*p_res).sum()/denom
                    interc_tmp = y_mean - slope * p_mean
                
                slope.data = slope_tmp
                interc.data = interc_tmp
                

                if train_loss > best_train_loss:
                    best_test_loss = test_loss
                    best_train_loss = train_loss

                    

            if epoch%10==0:
                print(epoch, best_train_loss, best_test_loss)

            
        with open('./results/neural_net/{}_{}_{}.csv'.format(repetition, n, dataset), 'w') as f_object:
            writer_object = writer(f_object, delimiter='\t')
            writer_object.writerow([repetition, n, dataset, best_train_loss, best_test_loss])


