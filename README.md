# Multimodal Activity Recognition on SAMoSA

Master's project — multimodal IoT data fusion using the SAMoSA dataset (audio + IMU).

Four experiments compare unimodal baselines against intermediate (feature-level) and
late (decision-level) fusion for 27-class activity recognition.

---

## Results summary

| Experiment | Model               | Test Accuracy | Macro-F1 |
|-----------|---------------------|--------------|----------|
| E1        | Unimodal IMU        | 10.5 %       | 0.047    |
| E2        | Unimodal Audio      | 33.3 %       | 0.276    |
| E3        | Intermediate Fusion | 59.3 %       | 0.555    |
| E4        | Late Fusion         | 36.4 %       | 0.298    |

Chance level (27 classes, uniform): ~3.7 %

---

## Requirements

Python 3.9+ recommended.

```
pip install -r requirements.txt
```

---

## Dataset setup

1. Obtain the SAMoSA dataset (`.pkl` files).
   Each file is named: `{subject}---{location}---{activity}---{trial}.pkl`

2. Place the raw files at:
   ```
   datasets/samosa/TrainingDataset/
   ```
   (The folder must be named exactly `TrainingDataset` — note the capital D and no space.)

3. Run the preprocessing script **once** from the repo root to generate `.npy` files:
   ```
   python datasets/samosa/preprocess_raw.py
   ```
   This creates:
   ```
   datasets/samosa/imu/train_data.npy      (1455, 100, 9)
   datasets/samosa/imu/test_data.npy       (162,  100, 9)
   datasets/samosa/audio/train_data.npy    (1455, 48000)
   datasets/samosa/audio/test_data.npy     (162,  48000)
   datasets/samosa/activity/train_data.npy (1455,)
   datasets/samosa/activity/test_data.npy  (162,)
   ```

> **Split note:** the train/test split is alphabetical by filename (not subject-disjoint).
> Some subjects appear in both splits. Reported accuracy is therefore an upper bound on
> generalisation to fully unseen subjects.

---

## Running experiments

All commands must be run from the `project_code/` directory.

```bash
# E1 — Unimodal IMU baseline
python experiments/e1_unimodal_imu.py

# E2 — Unimodal Audio baseline
python experiments/e2_unimodal_audio.py

# E3 — Intermediate (feature-level) fusion
python experiments/e3_intermediate_fusion.py

# E4 — Late (decision-level) fusion
#   NOTE: E4 loads checkpoints saved by E1 and E2.
#   Run E1 and E2 first, or ensure encoder_imu.pt, head_imu.pt,
#   encoder_audio.pt, and head_audio.pt are present in the working directory.
python experiments/e4_late_fusion.py
```

Each script reports test accuracy and macro-F1 at completion.
All experiments use `SEED = 42` for reproducibility.

---

## Project structure

```
project_code/
├── experiments/
│   ├── e1_unimodal_imu.py          # E1: IMU-only LeNet + MLP
│   ├── e2_unimodal_audio.py        # E2: Audio-only LeNet + MLP
│   ├── e3_intermediate_fusion.py   # E3: Concat fusion of both modalities
│   └── e4_late_fusion.py           # E4: Probability-average late fusion
├── datasets/
│   └── samosa/
│       ├── get_data.py             # DataLoader with z-score normalisation
│       └── preprocess_raw.py       # Converts raw .pkl files to .npy
├── training_structures/
│   ├── unimodal.py                 # Train/test loop for E1, E2
│   └── Supervised_Learning.py      # Train/test loop for E3 (MMDL wrapper)
├── unimodals/
│   └── common_models.py            # LeNet, MLP definitions
├── fusions/
│   └── common_fusions.py           # Concat fusion module
├── eval_scripts/
│   ├── performance.py              # accuracy, f1_score, AUPRC wrappers
│   ├── complexity.py               # Training complexity stubs
│   └── robustness.py              # Robustness evaluation stubs
├── utils/
│   └── AUPRC.py                   # Average precision utility
├── requirements.txt
└── README.md
```
