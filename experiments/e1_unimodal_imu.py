import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from unimodals.common_models import LeNet, MLP
from datasets.samosa.get_data import get_dataloader
from training_structures.unimodal import train, test

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = 'datasets/samosa'   
MODAL_NUM = 0                  # 0 = IMU, 1 = Audio
CHANNELS = 3
EPOCHS = 100
LR = 0.01
WEIGHT_DECAY = 0.0001
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    traindata, validdata, testdata = get_dataloader(DATA_DIR, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # IMU input: (B, 1, 100, 9)
    # LeNet(1,3,2) outputs (B,12,12); Flatten gives (B,144) = (B, CHANNELS*48)
    encoder = torch.nn.Sequential(LeNet(1, CHANNELS, 2), torch.nn.Flatten()).to(device)
    head    = MLP(CHANNELS * 48, 40, 27).to(device)

    print("Training unimodal IMU model...")
    train(
        encoder, head,
        traindata, validdata,
        total_epochs=EPOCHS,
        optimtype=torch.optim.SGD,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        modalnum=MODAL_NUM,
        save_encoder='encoder_imu.pt',
        save_head='head_imu.pt',
        early_stop=True,
    )

    print("\nTesting best unimodal IMU model:")
    encoder = torch.load('encoder_imu.pt', map_location=device, weights_only=False).to(device)
    head    = torch.load('head_imu.pt',    map_location=device, weights_only=False).to(device)
    test(encoder, head, testdata, modalnum=MODAL_NUM, no_robust=True)
