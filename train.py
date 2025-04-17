import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import pickle
import os
from tools.DataLoader import *
from tools.parser import *
      

# build the model
class network(nn.Module): 
    # later define a func to parse the network details
    def __init__(self, descr_array, energy_arr, input_file):
        super(network, self).__init__()
        self.descr_array = torch.from_numpy(descr_array).float()
        self.energy_arr = torch.from_numpy(energy_arr).float()
        self.descr_dim = self.descr_array[0].shape[1]
        self.info = input_parser(input_file)

    def make_model(self):    
        all_layers = [nn.Linear(self.descr_dim, self.info['n_nodes'][0]), nn.SiLU()]
        for l in range(self.info['n_layers']-1):
            all_layers.append(nn.Linear(self.info['n_nodes'][l],self.info['n_nodes'][l+1]))
            all_layers.append(nn.SiLU())
        all_layers.append(nn.Linear(self.info['n_nodes'][-1], 1)) # output is one value for energy
        self.model = nn.Sequential(*all_layers)
        return self.model

    def train_model(self, error_criterion = nn.MSELoss()):
        model = self.make_model()
        optimizer = optim.Adam(model.parameters(), lr= self.info['learning_rate']) # add optimizer choice
        model.train() # Puts the model in training mode
        # batches = self.build_batches(self.batch_size) # use data loader
        ## if restart:
        for epoch in range(self.info['n_epochs']):
            total_loss = 0
            # batch = self.get_batch(batches, epoch)
            # data, labels = batch[:,0], batch[:,1] 
            data, labels = self.descr_array, self.energy_arr 
            for d, label in zip(data, labels):
                optimizer.zero_grad()
                predictions = model(d)
                loss = error_criterion(predictions, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # print(f"Epoch {epoch+1}/{self.info['n_epochs']}, Loss: {total_loss/len(batch):.6f}") 
        torch.save(model.state_dict(), 'model_weights.pth')


