import torch.nn as nn
import numpy as np
import torch.optim as optim
import pickle
import os
from DataLoader import *
      

# build the model
class NNPModel(nn.Module, AtomicDescriptors): # inherits two classes
    # later define a func to parse the network details
    def __init__(self, n_layers =1,  n_nodes=8):
        super(NNPModel, self).__init__()
        super(AtomicDescriptors, self).__init__() 
        self.descr_dim = self.descr_dim()
        all_layers = [nn.Linear(self.descr_dim, n_nodes), nn.SiLU()]
        for l in range(n_layers-1):
            all_layers.append(nn.Linear(self.descr_dim, n_nodes), nn.SiLU())
        all_layers.append(nn.Linear(n_nodes, 1)) # output is one value for energy
        self.model = nn.Sequential(all_layers)

    def train_model(self, error_criterion = nn.MSELoss(), epochs=50, lr = 0.001):
        model = NNPModel()
        optimizer = optim.Adam(model.parameters(), lr= lr) # add optimizer choice
        model.train() # Puts the model in training mode
        batches = self.bulid_batches(self.batch_size)
        for epoch in range(epochs):
            total_loss = 0
            batch = self.get_batch(batches, epoch)
            data, labels = batch[:,0], batch[:,1] 
            for d, label in zip(data, labels):
                optimizer.zero_grad()
                predictions = model(d)
                loss = error_criterion(predictions, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(batch):.6f}") 

# Model initialization
error_criterion = nn.MSELoss()
# Train the model
inst = NNPModel()
inst.train_model(error_criterion)
