import pickle as pk
import numpy as np
from ase.io import read, write , extxyz, Trajectory
import json
import os
import ase

ptable = ['null', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
          'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
          'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 
          'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 
          'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba',
          'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 
          'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
          'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 
          'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 
          'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut', 'Uuq', 
          'Uup', 'Uuh', 'Uus', 'Uuo']

def get_energy(Atoms):
    try:
        energy = Atoms.info['REF_energy']
    except KeyError:
        try:
            energy = Atoms.info['Energy']
        except KeyError:
            try:
                energy = Atoms.info['energy[eV]']
            except KeyError:
                try:
                    energy = Atoms.get_potential_energy()
                except (AttributeError, RuntimeError):
                    print('No value for total energy was found')
                    energy = 10
    return energy

def get_forces(Atoms):
    try:
        forces = Atoms.arrays['forceseV/Ang']
    except KeyError:
        try:
            forces = np.array(Atoms.arrays['REF_forces'])
        except KeyError:
            try:
                forces = Atoms.arrays['force']
            except KeyError:
                try:
                    forces = np.array(Atoms.get_forces(apply_constraint=False))
                except (AttributeError, RuntimeError):
                    print('No value for forces was found')
                    n_atoms = len(Atoms.get_positions())
                    forces = np.zeros((n_atoms, 3))

    return forces

def get_positions(Atoms):
    pos = np.array(Atoms.get_positions())
    return pos

def get_cell(Atoms):
    cell = np.array(Atoms.get_cell())
    return cell

def get_pbc(Atoms):
    pbc = Atoms.get_pbc()
    return pbc

def get_atomic_numbers(Atoms):
    atomic_no = str(Atoms.numbers)
    return atomic_no

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
                inst = Extxyz_to_json(self.data_path, list(self.file))
                inst.work_flow()
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
                inst = Extxyz_to_json(self.data_path, list(self.files))
                inst.work_flow()

            else:
                print('other formats')
                return None

class Extxyz_to_json:
    def __init__(self, _path, _file):
        self.file = _file
        self.path = _path

    def load_xyz(self):
        File = open(self.path + self.file, 'r')
        file_name = self.file.remove('.extxyz')
        print(f'file {self.file}', flush = True)
        return File, file_name

        
    def read_system(self, idx):
        File, file_name = self.read_system()
        structure = list(extxyz.read_xyz(File, index = idx, properties_parser = extxyz.key_val_str_to_dict))
        atomic_numbers = get_atomic_numbers(structure)
        natom = len(atomic_numbers)
        cell = get_cell(structure)
        pos = get_positions(structure)
        energy = get_energy(structure)
        force = get_forces(structure)
        symbols = [ptable[i] for i in atomic_numbers]
        Id = str(idx)+'_'+ file_name
        lst = [natom, atomic_numbers, cell, symbols, pos, energy, force, Id]
        return lst

    def build_atoms(self, data, idx):
        pos = data[4].tolist()
        natoms = data[0]
        force = data[6].tolist()
        atoms = []
        symbols = data[3]
        labels = list(range(natoms))
        for idx, (atom, pos, force) in enumerate(zip(symbols, pos, force)):
                species = atom
                atoms.append([labels[idx],species, pos, force])

        return atoms

            
    def dump_json(self, idx, data):
        json_dict = {}
        json_dict["key"] = idx
        json_dict["atomic_position_unit"] = self.pos_unit
        json_dict["unit_of_length"] = self.length_unit
        json_dict["energy"] = [data[4], "eV"]
        json_dict["lattice_vectors"] = data[1].tolist()
        json_dict["atoms"] = self.build_atoms(data, idx)
        id = self.file.split('.')[0]
        with open(f'{data[-1]}.json', 'w') as json_file:
            json.dump(json_dict, json_file)
            if idx % 500 == 0 :
               print(f'{id}_{idx}.json  Done', flush = True)

    def count_atomic_structures_ase(self):
        structures = ase.io.read(self.path + self.file, index=':')
        return len(structures)
    

    def work_flow(self):
        all_idx = self.count_atomic_structures_ase()
        ref_energy = []
        if not os.path.exists(self.path + f'{self.file.split(".")[0]}_jsons'):
            os.makedirs(self.path + f'{self.file.split(".")[0]}_jsons')
        os.chdir(self.path + f'{self.file.split(".")[0]}_jsons')
        for idx in range(all_idx):
            data = self.read_system(idx)
            if data:
                self.dump_json(idx, data)
                ref_energy.append([data[2], data[4]])
            else:
                print('End of the file')

        if self.File:
            self.File.close()
            os.chdir(self.path)


        




        




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

    