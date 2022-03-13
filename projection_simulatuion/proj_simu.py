import os
import re
import glob
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data import Dataset
import ctlib
from torch.autograd import Function
systemMat = torch.sparse.FloatTensor()
cuda = True if torch.cuda.is_available() else False

class fan_ed(Function):
    def __init__(self, views, dets, width, height, dImg, dDet, dAng, s2r, d2r, binshift):
        self.options = torch.Tensor([views, dets, width, height, dImg, dDet, dAng, s2r, d2r, binshift])
        self.options = self.options.double().cuda()

    def forward(self, image):
        return ctlib.projection(image, self.options, 0)


class trainset_loader(Dataset):
    def __init__(self):
        self.files_A = sorted(glob.glob('train/label/data' + '*.mat'))
        
    def __getitem__(self, index):
        file_A = self.files_A[index]
        file_B = file_A.replace('label','projection')
        input_data = sio.loadmat(file_A)['data']
        input_data = torch.FloatTensor(input_data.reshape(1,256,256))
        return input_data, file_B

    def __len__(self):
        return len(self.files_A)

class testset_loader(Dataset):
    def __init__(self):
        self.files_A = sorted(glob.glob('test/label/data' + '*.mat'))
        
    def __getitem__(self, index):
        file_A = self.files_A[index]
        file_B = file_A.replace('label','projection')
        input_data = sio.loadmat(file_A)['data']
        input_data = torch.FloatTensor(input_data.reshape(1,256,256))
        return input_data, file_B

    def __len__(self):
        return len(self.files_A)

class valiset_loader(Dataset):
    def __init__(self):
        self.files_A = sorted(glob.glob('validate/label/data' + '*.mat'))
        
    def __getitem__(self, index):
        file_A = self.files_A[index]
        file_B = file_A.replace('label','projection')
        input_data = sio.loadmat(file_A)['data']
        input_data = torch.FloatTensor(input_data.reshape(1,256,256))
        return input_data, file_B

    def __len__(self):
        return len(self.files_A)

if __name__ == "__main__":
    train_data = DataLoader(trainset_loader(), batch_size=5, shuffle=False, num_workers=2)
    test_data = DataLoader(testset_loader(), batch_size=5, shuffle=False, num_workers=2)
    # vali_data = DataLoader(valiset_loader(), batch_size=5, shuffle=False, num_workers=2)
    projector = fan_ed(1024, 512, 256, 256, 0.006641, 0.0072, 0.006134, 2.5, 2.5, 0)
    for batch_index, data in enumerate(train_data):
        input_data, file_name = data
        input_data = input_data.double()
        if cuda:
            input_data = input_data.cuda()
        proj = projector.forward(input_data)
        for i in range(proj.size(0)):
            temp = proj[i].cpu().numpy().reshape(1024,512)
            sio.savemat(file_name[i], {'data':temp})
    for batch_index, data in enumerate(test_data):
        input_data, file_name = data
        input_data = input_data.double()
        if cuda:
            input_data = input_data.cuda()
        proj = projector.forward(input_data)
        for i in range(proj.size(0)):
            temp = proj[i].cpu().numpy().reshape(1024,512)
            sio.savemat(file_name[i], {'data':temp})

    # for batch_index, data in enumerate(vali_data):
    #     input_data, file_name = data
    #     input_data = input_data.double()
    #     if cuda:
    #         input_data = input_data.cuda()
    #     proj = projector.forward(input_data)
    #     for i in range(proj.size(0)):
    #         temp = proj[i].cpu().numpy().reshape(1024,512)
    #         sio.savemat(file_name[i], {'data':temp})