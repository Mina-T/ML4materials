import pickle as pk
import numpy as np
from ase.io import read, write , extxyz, Trajectory
import json
import os

class ReadDataset:
    ''''
    Handles reading/ converting dataset files from a directory.

    Parameters:
    - data_path (str): Path to the dataset directory.
    - data_file (str, optional): Specific dataset file to use (if any).
    '''
    def __init__(self, data_path, data_file = None):
        self.data_path = data_path
        self.data_file = data_file
        self.files = []

    def convert_dataset(self):
        if self.data_file:
            if self.file.endswith('.xyz'):
                print('Dataset file is xyz.', flush = True)
            elif self.file.endswith('.extxyz'):
                print('Dataset file is extended xyz.', flush = True)
                self.extxyz_to_json(list(self.file))
            else:
                print('other formats')
                return None # for now

        else:
            self.files = os.listdir(self.data_path)
            print(f'There are {len(self.files)} dataset files.', flush=True)
            if self.files[0].endswith('.xyz'):
                print('Dataset files are xyz.', flush = True)
            elif self.files[0].endswith('.extxyz'):
                print('Dataset file is extended xyz.', flush = True)
                self.extxyz_to_json(list(self.files))

            else:
                print('other formats')
                return None
            
    def extxyz_to_json(self, _files):
        for _file in _files:
            read = read(_file, index = ':')
            
        




        




# for now it reads a pkl file
class AtomicDescriptors:
    def __init__(self, descriptor_file):
        '''
        Reads an array from descriptor_file
        '''
        self.dataset = pickle.load(open(descriptor_file, 'rb'))
        print('Dataset wad loaded.')

    def descr_dim(self):
        descr_dim = self.dataset[0].shape
        return descr_dim

    def bulid_batches(self, batch_size):
        '''
        returns a 2d array of data and labels.
        '''
        dataset_length = len(self.dataset)
        n_batches = dataset_length // batch_size 
        remaining_data = dataset_length % n_batches
        if remaining_data > 0 :
            n_batches += 1
        batches = [self.dataset[n * batch_size: (n + 1) * batch_size] for n in range(n_batches)]
        return batches
    
    def get_batch(self, batches, epoch):
        if epoch > len(batches):
            batch = batches[epoch%len(batches)]
        else:
            batch = batches[epoch]
        return batch

    