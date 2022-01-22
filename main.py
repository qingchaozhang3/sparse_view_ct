import argparse
import os
import re
import glob
import numpy as np
import scipy.io as sio
from vis_tools import Visualizer

import torch
import torch.nn as nn
import torch.optim as optim
import model

from datasets import trainset_loader
from datasets import testset_loader
from torch.utils.data import DataLoader
from torch.autograd import Variable
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
import time
os.environ["CUDA_VISIBLE_DEVICES"]="1"
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("--n_block", type=int, default=3)
parser.add_argument("--n_cpu", type=int, default=4)
parser.add_argument("--model_save_path", type=str, default="saved_models/1st")
parser.add_argument('--checkpoint_interval', type=int, default=1)

opt = parser.parse_args()
cuda = True if torch.cuda.is_available() else False
#train_vis = Visualizer(env='training_magic')

class net():
    def __init__(self):
        self.model = model.LDA(opt.n_block, views=1024, dets=512, width=256, height=256, 
            dImg=0.006641, dDet=0.0072, dAng=0.006134, s2r=2.5, d2r=2.5, binshift=0)
        self.loss = nn.MSELoss()
        self.path = opt.model_save_path
        self.train_data = DataLoader(trainset_loader("mayo_data_low_dose_256", '64views'),
            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
        self.test_data = DataLoader(testset_loader("mayo_data_low_dose_256",'64views'),
            batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
        self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr)
        self.start = 0
        self.epoch = opt.epochs
        self.check_saved_model()       
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size = 5, gamma=0.8)
        if cuda:
            self.model = self.model.cuda()

    def check_saved_model(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            self.initialize_weights()
        else:
            model_list = glob.glob(self.path + '/model_epoch_*.pth')
            if len(model_list) == 0:
                self.initialize_weights()
            else:
                last_epoch = 0
                for model in model_list:
                    epoch_num = int(re.findall(r'model_epoch_(-?[0-9]\d*).pth', model)[0])
                    if epoch_num > last_epoch:
                        last_epoch = epoch_num
                self.start = last_epoch
                print('%s/model_epoch_%04d.pth' % (self.path, last_epoch))
                self.model.load_state_dict(torch.load(
                    '%s/model_epoch_%04d.pth' % (self.path, last_epoch)), strict=False)

    def displaywin(self, img, low=0.42, high=0.62):
        img[img<low] = low
        img[img>high] = high
        img = (img - low)/(high - low) * 255
      #  img = img* 255
        return img

    def initialize_weights(self):
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            if isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def train(self):
        self.start = 0
        Training_loss = []
        Training_loss_var = []
        for epoch in range(self.start, self.epoch):
            training_loss_epoch = []
            print('Epoch')
            for batch_index, data in enumerate(self.train_data):
                input_data, label_data, prj_data = data                
                if cuda:
                    input_data = input_data.cuda()
                    label_data = label_data.cuda()
                    prj_data = prj_data.cuda()
                self.optimizer.zero_grad()
                output = self.model(input_data, prj_data, opt.n_block)
                loss = self.loss(output, label_data)
                loss.backward()
                self.optimizer.step()

                print('batch_index')
                training_loss = loss.item()
                print(
                    "[Epoch %d/%d] [Batch %d/%d]: [loss: %f]"
                    % (epoch+1, self.epoch, batch_index+1, len(self.train_data), training_loss)
                )
                training_loss_epoch.append(training_loss)
                # train_vis.plot('Loss', loss.item())
                # train_vis.img('Ground Truth', self.displaywin(label_data.detach()).cpu())
                # train_vis.img('Result', self.displaywin(output.detach()).cpu())
                # train_vis.img('Input', self.displaywin(input_data.detach()).cpu())
            self.scheduler.step()
            if opt.checkpoint_interval != -1 and (epoch+1) % opt.checkpoint_interval == 0:
                torch.save(self.model.state_dict(), '%s/model_epoch_%04d.pth' % (self.path, epoch+1))
            Training_loss.append(np.mean(training_loss_epoch))
            Training_loss_var.append(np.var(training_loss_epoch))
            sio.savemat('train_loss.mat', {'training_loss_tmi': Training_loss, 'training_loss_tmi_var': Training_loss_var})

    def test(self):
        Psnr = np.zeros(100)
        Ssim = np.zeros(100)
        Psnr_input = np.zeros(100)
        Ssim_input = np.zeros(100)
        j = 0
        for batch_index, data in enumerate(self.test_data):
            input_data, label_data, prj_data, res_name = data
            if cuda:
                input_data = input_data.cuda()
                label_data = label_data.cuda()
                prj_data = prj_data.cuda()
            with torch.no_grad():
                output = self.model(input_data, prj_data, opt.n_block)
            res = output.cpu().numpy()
            output = (self.displaywin(output, low=0.0, high=1.0) / 255).view(-1,256,256).cpu().numpy()
            label = (self.displaywin(label_data, low=0.0, high=1.0) / 255).view(-1,256,256).cpu().numpy()
            input_data = (self.displaywin(input_data, low=0.0, high=1.0) / 255).view(-1,256,256).cpu().numpy()
            psnr = np.zeros(output.shape[0])
            ssim = np.zeros(output.shape[0])
            psnr_input = np.zeros(output.shape[0])
            ssim_input = np.zeros(output.shape[0])
            for i in range(output.shape[0]):
                psnr[i] = compare_psnr(label[i], output[i])
                ssim[i] = compare_ssim(label[i], output[i])
                psnr_input[i] = compare_psnr(label[i], input_data[i])
                ssim_input[i] = compare_ssim(label[i], input_data[i])
                Psnr[j] = psnr[i]
                Ssim[j] = ssim[i]
                Psnr_input[j] = psnr_input[i]
                Ssim_input[j] = ssim_input[i]
                j = j+1
                
                # print("avepsnr: %f, ssim: %f" % (np.mean(psnr), np.mean(ssim)))
                # print("psnr: %f, ssim: %f" % (psnr[i], ssim[i]))
                # sio.savemat(res_name[i], {'data':res[i,0], 'psnr':psnr[i], 'ssim':ssim[i]})
        print("avepsnr_input: %f, avessim_input: %f" % (np.mean(Psnr_input), np.mean(Ssim_input)))
        print("avepsnr: %f, avessim: %f" % (np.mean(Psnr), np.mean(Ssim)))







if __name__ == "__main__":
    network = net()
    network.train()
    network.test()
