import math
import numpy as np
import os
import copy
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from cec15_test_func import *

ini_flag = 0
problem_types = copy.deepcopy(functions)
problem_types['Composition'] = Composition
problem_types['Hybrid'] = Hybrid


class Tester(Dataset):
    def __init__(self, filename, problem_type, dim, num_samples, offset):
        super(Tester, self).__init__()
        self.dim = dim
        self.data = []
        if filename is not None:
            self.data = Tester.dataset_read(filename, problem_type, num_samples, offset)
        else:
            self.data = Tester.dataset_gen("", num_samples, problem_type, dim, store=False)
        self.N = len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.N

    @staticmethod
    def dataset_gen(path, size, problem_type, dim=0, cf_num=0, problem_list=None, store=True):  # Generate a set of problems
        if problem_type not in problem_types.keys():
            return
        if problem_type == 'Composition' or problem_type == 'Hybrid':
            file = problem_type + '_D' + str(dim) + '.txt'
            file = path + file
            data = problem_types[problem_type].generator(file, dim, cf_num, problem_list, size, store)
        else:
            file = problem_type + '_D' + str(dim) + '.txt'
            file = path + file
            data = Problem.generator(problem_type, dim, size)
            if store:
                Problem.store_instance(data, file)
        return data

    @staticmethod
    def dataset_read(path, problem_type, size, offset):  # Load a dataset from file
        if problem_type not in problem_types.keys():
            return
        if problem_type == 'Composition' or problem_type == 'Hybrid':
            data = problem_types[problem_type].read(path, offset + size)[offset:]
        else:
            data = problem_types[problem_type].read(path, problem_type, offset + size)[offset:]
        return data

    # train + valid
    # base + composition


dim = 10
bs = 2
Type = 'Composition'
T = Tester('Composition_D10.txt', Type, dim, 2, 0)
print(T.data)
x = np.zeros(dim)
training_dataloader = DataLoader(T, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
for batch_id, batch in enumerate(training_dataloader):
    instances = problem_types[Type].get_instance(batch)
    print(instances[0].func(x))
