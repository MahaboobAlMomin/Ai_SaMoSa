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
from fusions.common_fusions import Concat
from datasets.samosa.get_data import get_dataloader
from training_structures.Supervised_Learning import train, test

# ── Configuration ────────────────────────────────────────────────────────────
# LR and EPOCHS match E1/E2 exactly so the comparison is fair.
DATA_DIR     = 'datasets/samosa'
CHANNELS     = 3
EPOCHS       = 100
LR           = 0.01
WEIGHT_DECAY = 0.0001
# ─────────────────────────────────────────────────────────────────────────────

# Intermediate (feature-level) fusion:
#   Each modality has its own encoder → encoded representations are concatenated
#   → shared MLP head classifies the combined feature vector.
#   Fusion happens AFTER encoders but BEFORE the final classifier.
#   This is NOT late fusion (which combines per-modality predictions).

if __name__ == '__main__':
    traindata, validdata, testdata = get_dataloader(DATA_DIR, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # IMU:   LeNet(1,3,2) on (B,1,100,9)   → (B,12,12) → Concat flattens → 144
    # Audio: LeNet(1,3,5) on (B,1,100,480) → (B,96,7)  → Concat flattens → 672
    # Total after Concat: 816 = CHANNELS * 272
    encoders = [
        LeNet(1, CHANNELS, 2).to(device),   # IMU encoder
        LeNet(1, CHANNELS, 5).to(device),   # Audio encoder
    ]
    fusion = Concat().to(device)
    head   = MLP(CHANNELS * 272, 100, 27).to(device)

    print("Training intermediate fusion model (IMU + Audio → Concat → MLP)...")
    train(
        encoders, fusion, head,
        traindata, validdata,
        total_epochs=EPOCHS,
        optimtype=torch.optim.SGD,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        save='best_intermediate_fusion.pt',
        early_stop=True,
    )

    print("\nTesting best intermediate fusion model:")
    model = torch.load('best_intermediate_fusion.pt', map_location=device, weights_only=False).to(device)
    test(model, testdata, no_robust=True)
