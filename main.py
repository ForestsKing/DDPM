import argparse

import torch

from exp.exp import Exp
from utils.set_seed import set_seed

if __name__ == '__main__':
    set_seed(42)
    torch.cuda.set_device(1)

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=5)

    parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--timesteps', type=int, default=500)

    parser.add_argument('--result_dir', type=str, default='./result/')
    parser.add_argument('--model_dir', type=str, default='./checkpoint/')

    config = vars(parser.parse_args())

    exp = Exp(config)
    exp.train()
    exp.test()
