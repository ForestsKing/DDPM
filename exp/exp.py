import os
from time import time

import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm

from model.diffusion import DiffusionModel
from model.unet import UNetModel


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
        dataset = MNIST(root="./dataset", train=True, download=False,
                        transform=Compose([
                            ToTensor(),
                            Normalize(mean=[0.5], std=[0.5])
                        ]))

        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _get_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNetModel(in_channels=1,
                               model_channels=96,
                               out_channels=1,
                               channel_mult=(1, 2, 2),
                               attention_resolutions=[]).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
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
            for (batch_x, _) in tqdm(self.dataloader):
                self.optimizer.zero_grad()
                loss = self._process_one_batch(batch_x)
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            train_loss = np.average(train_loss)
            end = time()
            print("Epoch: {0} || Train Loss: {1:.6f} || Cost: {2:.6f}".format(e + 1, train_loss, end - start))

        self.model.load_state_dict(torch.load(self.model_dir + 'model.pkl'))

    def test(self):
        self.model.load_state_dict(torch.load(self.model_dir + 'model.pkl'))
        generated_images = self.dm.sample(self.model, image_size=self.image_size, batch_size=64, channels=self.channels)

        # generate new images
        fig = plt.figure(figsize=(12, 12), constrained_layout=True)
        gs = fig.add_gridspec(8, 8)

        imgs = generated_images[-1].reshape(8, 8, 28, 28)
        for n_row in range(8):
            for n_col in range(8):
                f_ax = fig.add_subplot(gs[n_row, n_col])
                f_ax.imshow((imgs[n_row, n_col] + 1.0) * 255 / 2, cmap="gray")
                f_ax.axis("off")
        fig.savefig(self.result_dir + 'result.png')
        fig.show()

        # show the denoise steps
        fig = plt.figure(figsize=(12, 12), constrained_layout=True)
        gs = fig.add_gridspec(16, 16)

        for n_row in range(16):
            for n_col in range(16):
                f_ax = fig.add_subplot(gs[n_row, n_col])
                t_idx = (self.timesteps // 16) * n_col if n_col < 15 else -1
                img = generated_images[t_idx][n_row].reshape(28, 28)
                f_ax.imshow((img + 1.0) * 255 / 2, cmap="gray")
                f_ax.axis("off")

        fig.savefig(self.result_dir + 'result_step.png')
        fig.show()
