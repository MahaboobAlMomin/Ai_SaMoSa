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
import torch.nn.functional as F
from sklearn.metrics import f1_score
from datasets.samosa.get_data import get_dataloader

DATA_DIR = 'datasets/samosa'

if __name__ == '__main__':
    _, _, testdata = get_dataloader(DATA_DIR, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    enc_imu   = torch.load('encoder_imu.pt',   map_location=device, weights_only=False).eval()
    head_imu  = torch.load('head_imu.pt',      map_location=device, weights_only=False).eval()
    enc_audio = torch.load('encoder_audio.pt', map_location=device, weights_only=False).eval()
    head_audio= torch.load('head_audio.pt',    map_location=device, weights_only=False).eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in testdata:
            imu, audio, labels = batch
            imu   = imu.float().to(device)
            audio = audio.float().to(device)

            logits_imu   = head_imu(enc_imu(imu))
            logits_audio = head_audio(enc_audio(audio))

            prob_imu   = F.softmax(logits_imu,   dim=1)
            prob_audio = F.softmax(logits_audio, dim=1)

            avg_prob = (prob_imu + prob_audio) / 2.0
            preds = avg_prob.argmax(dim=1).cpu()

            all_preds.append(preds)
            all_labels.append(labels)

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    accuracy = (all_preds == all_labels).float().mean().item()
    macro_f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='macro')

    print(f"\nE4 Late Fusion Results")
    print(f"  Test Accuracy : {accuracy * 100:.1f}%")
    print(f"  Macro F1      : {macro_f1:.4f}")
