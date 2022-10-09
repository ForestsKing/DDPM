import os
from time import time

import torch
import numpy as np
from matplotlib import pyplot as plt, animation
from torch import optim, nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm

from model.diffusion import DiffusionModel
from model.unet import UNetModel
from utils.earlystop import EarlyStopping


class Exp:
    def __init__(self, config):
        self.__dict__.update(config)
        self.dm = DiffusionModel(timesteps=self.timesteps)
        self._get_data()
        self._get_model()

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

    def _get_data(self):
        traindataset = MNIST(root="./dataset", train=True, download=False,
                             transform=Compose([
                                 ToTensor(),
                                 Normalize(mean=[0.5], std=[0.5])
                             ]))
        testdataset = MNIST(root="./dataset", train=False, download=False,
                            transform=Compose([
                                ToTensor(),
                                Normalize(mean=[0.5], std=[0.5])
                            ]))

        self.trainloader = DataLoader(traindataset, batch_size=self.batch_size, shuffle=False)
        self.testloader = DataLoader(testdataset, batch_size=self.batch_size, shuffle=False)

    def _get_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNetModel(in_channels=1,
                               model_channels=96,
                               out_channels=1,
                               channel_mult=(1, 2, 2),
                               attention_resolutions=[]).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.75 ** ((epoch - 1) // 2))
        self.early_stopping = EarlyStopping(patience=self.patience, path=self.model_dir + 'model.pkl')
        self.criterion = nn.MSELoss()
        print(self.device)

    def _process_one_batch(self, batch_x):
        batch_size = batch_x.shape[0]  # 可能最后一个 batch 不满
        x_start = batch_x.to(self.device)

        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()

        noise = torch.randn_like(x_start)
        x_noisy = self.dm.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t)

        loss = self.criterion(noise, predicted_noise)

        return loss

    def train(self):
        for e in range(self.epochs):
            start = time()

            self.model.train()
            train_loss = []
            for (batch_x, _) in tqdm(self.trainloader):
                self.optimizer.zero_grad()
                loss = self._process_one_batch(batch_x)
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                test_loss = []
                for (batch_x, _) in tqdm(self.testloader):
                    loss = self._process_one_batch(batch_x)
                    test_loss.append(loss.item())

            train_loss, test_loss = np.average(train_loss), np.average(test_loss)
            end = time()
            print("Epoch: {0} || Train Loss: {1:.6f} | Test Loss: {2:.6f} || Cost: {3:.6f}".format(e + 1, train_loss,
                                                                                                   test_loss,
                                                                                                   end - start))

            self.early_stopping(test_loss, self.model)
            if self.early_stopping.early_stop:
                break
            self.scheduler.step()

        self.model.load_state_dict(torch.load(self.model_dir + 'model.pkl'))

    def test(self):
        self.model.load_state_dict(torch.load(self.model_dir + 'model.pkl'))
        samples = self.dm.sample(self.model, image_size=self.image_size, batch_size=8, channels=self.channels)

        for index in range(8):
            fig = plt.figure()
            ims = []
            for i in range(self.timesteps):
                im = plt.imshow(samples[i][index].reshape(self.image_size, self.image_size, self.channels),
                                cmap="gray", animated=True)
                ims.append([im])

            animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
            animate.save(self.result_dir + 'diffusion model ' + str(index) + '.gif')
            plt.show()
