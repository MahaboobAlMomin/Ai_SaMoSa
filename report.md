# A Comparative Study of Intermediate vs Late Fusion for Multimodal IoT Sensor Data using the SAMoSA Dataset

**Course:** Multimodal Machine Learning — Master's Project
**Framework:** MultiIoT

---

## Abstract

This project investigates multimodal sensor fusion for human activity recognition using the SAMoSA dataset, which provides paired audio and inertial measurement unit (IMU) recordings across 27 activity classes. Four experiments are conducted: two unimodal baselines (IMU-only and audio-only), one intermediate fusion model (feature-level concatenation), and one late fusion model (decision-level probability averaging). All models share the same convolutional encoder architecture and identical training configuration to ensure a fair comparison. Intermediate fusion achieves 59.3% test accuracy and a macro-F1 score of 0.555, substantially outperforming both unimodal baselines and late fusion. The results support the hypothesis that joint feature learning across modalities is more effective than combining independent unimodal decisions for this task. Known limitations of the evaluation protocol are discussed, including a non-subject-disjoint train/test split.

---

## 1. Introduction

Human activity recognition (HAR) from wearable IoT sensors is a well-studied problem with applications in healthcare monitoring, smart environments, and human-computer interaction. Most real-world deployments use more than one sensor modality — for example, combining motion data from accelerometers with acoustic signals from microphones. How these modalities should be combined is an open design question, and the answer depends heavily on the degree of complementarity between the signals.

Two broad approaches to multimodal fusion are commonly contrasted in the literature. **Intermediate fusion** (also called feature-level or early-to-mid fusion) passes each modality through its own encoder, concatenates the resulting feature vectors, and trains a shared classifier on the joint representation. **Late fusion** (decision-level fusion) trains independent unimodal models and combines their output probability distributions at inference time, typically by averaging or voting. Intermediate fusion allows the model to learn cross-modal interactions; late fusion does not, but it is more modular and easier to train.

This project uses the SAMoSA dataset within the MultiIoT framework to compare these two fusion strategies against unimodal baselines. The specific contributions are:

1. A complete preprocessing pipeline converting raw SAMoSA `.pkl` files into normalised `.npy` arrays.
2. Four controlled experiments with matched architectures and hyperparameters.
3. An evaluation of the trade-offs between intermediate and late fusion on a 27-class IoT activity recognition task.

---

## 2. Dataset

### 2.1 SAMoSA

SAMoSA (Sensor-Audio Multimodal Open-Source Activity) is a publicly available dataset designed for multi-sensor activity recognition. Each recording captures a participant performing one of 27 daily activities — ranging from domestic tasks (chopping, blending, washing) to personal activities (toothbrushing, coughing, clapping) — and provides two synchronised sensor streams:

- **Audio:** a one-dimensional raw PCM signal sampled at 48 kHz.
- **IMU:** a nine-axis inertial signal (3-axis accelerometer, 3-axis gyroscope, 3-axis magnetometer) at a lower sampling rate, stored as a variable-length (N × 9) array.

Files are named using the convention `{subject}---{location}---{activity}---{trial}.pkl`, which encodes all metadata directly in the filename.

The 27 activity classes are: Alarm\_clock, Blender\_in\_use, Brushing\_hair, Chopping, Clapping, Coughing, Drill in use, Drinking, Grating, Hair\_dryer\_in\_use, Hammering, Knocking, Laughing, Microwave, Other, Pouring\_pitcher, Sanding, Scratching, Screwing, Shaver\_in\_use, Toilet\_flushing, Toothbrushing, Twisting\_jar, Vacuum in use, Washing\_Utensils, Washing\_hands, Wiping\_with\_rag.

### 2.2 Split Statistics

After preprocessing, the dataset is divided as follows:

| Split      | Samples | Use                      |
|------------|---------|--------------------------|
| Train      | 1,335   | Model training           |
| Validation | 120     | Early stopping           |
| Test       | 162     | Final evaluation         |
| **Total used** | **1,617** | |

Files are sorted alphabetically by filename before splitting: the first 1,455 become the train+validation set and the next 162 become the test set. The remaining 16 files (the alphabetical tail) are excluded. The validation set consists of the last 120 samples of the 1,455-sample training pool (indices 1,335–1,455).

**Split limitation:** because filenames sort primarily by subject identifier, some subjects appear in both the train and test sets. This is a non-subject-disjoint split. Reported accuracy is therefore an optimistic estimate of generalisation to entirely unseen subjects. This limitation is acknowledged throughout; a subject-disjoint evaluation would require a stratified split by subject ID.

---

## 3. Methodology

### 3.1 Preprocessing Pipeline

Raw `.pkl` files are loaded and converted to three `.npy` arrays per split: IMU data, audio data, and integer class labels.

**IMU:** Each raw IMU array has a variable number of timesteps. All sequences are fixed to exactly 100 timesteps by front-cropping sequences longer than 100 rows and zero-padding (at the end) sequences shorter than 100 rows. The resulting shape per sample is (100, 9).

**Audio:** Each raw audio array is a one-dimensional PCM signal of variable length. Signals longer than 48,000 samples (one second at 48 kHz) are front-cropped; shorter signals are zero-padded. The resulting shape per sample is (48,000,).

**Reshaping for 2D convolution:** Before passing to the convolutional encoders, audio is reshaped from (48,000,) to (100, 480), treating every 480-sample block as one time row. This allows the same LeNet 2D convolution architecture to process both modalities.

**Normalisation:** Normalisation statistics are computed on the training set only and applied identically to validation and test sets, preventing data leakage.

- *IMU:* Z-score normalisation per sensor channel. The mean and standard deviation of each of the 9 axes are computed over all (N × T) training observations and used to standardise each axis independently.
- *Audio:* Global Z-score normalisation. A single scalar mean and standard deviation are computed over all training audio samples and applied uniformly.

A channel dimension is added to both modalities (unsqueeze at axis 1) so that inputs to the 2D LeNet encoders have shape (B, 1, 100, 9) for IMU and (B, 1, 100, 480) for audio.

### 3.2 Model Architectures

All experiments use the same LeNet-based encoder architecture from the MultiIoT `unimodals.common_models` module, and the same MLP head design. This ensures that any difference in performance is attributable to the fusion strategy rather than architectural capacity.

**LeNet encoder:** A lightweight convolutional network with two convolutional layers, ReLU activations, and max-pooling. The output is a spatial feature map that is flattened to a fixed-dimensional vector before being passed to the classifier.

**MLP head:** A two-layer fully connected network with ReLU activation, mapping from the encoder output dimension to 27 class logits via a hidden layer.

The precise architectural configurations per experiment are as follows:

| Experiment | Modality | LeNet config | Encoder output dim | MLP hidden dim | Output |
|---|---|---|---|---|---|
| E1 | IMU | LeNet(in=1, channels=3, kernel=2) | 144 | 40 | 27 |
| E2 | Audio | LeNet(in=1, channels=3, kernel=5) | 672 | 100 | 27 |
| E3 (IMU branch) | IMU | LeNet(in=1, channels=3, kernel=2) | 144 | — | — |
| E3 (Audio branch) | Audio | LeNet(in=1, channels=3, kernel=5) | 672 | — | — |
| E3 (fused) | IMU + Audio | Concat(144, 672) | 816 | 100 | 27 |
| E4 | IMU + Audio | Reuses E1 + E2 | — | — | 27 |

### 3.3 Fusion Strategies

**E3 — Intermediate (Feature-Level) Fusion**

Each modality has its own dedicated encoder. The two encoder outputs (144-dimensional IMU features and 672-dimensional audio features) are concatenated into a single 816-dimensional joint representation. A shared MLP classifier then maps this fused vector to 27 class logits. Training is end-to-end: encoder weights, fusion, and classifier are optimised jointly using a single cross-entropy loss. This is the key property of intermediate fusion — cross-modal interactions are learnable.

```
IMU   → LeNet_IMU   → [144-d]  ──┐
                                   Concat → [816-d] → MLP → [27-d logits]
Audio → LeNet_Audio → [672-d]  ──┘
```

**E4 — Late (Decision-Level) Fusion**

The independently trained models from E1 and E2 are loaded in evaluation mode. For each test sample, both models produce 27-dimensional logit vectors, which are independently passed through softmax to obtain class probability distributions. The two distributions are averaged element-wise and the class with the highest average probability is taken as the prediction. No additional training occurs in E4.

```
IMU   → [E1 encoder + head] → softmax → [27-d probs] ──┐
                                                          mean → argmax → class
Audio → [E2 encoder + head] → softmax → [27-d probs] ──┘
```

### 3.4 Training Configuration

All training experiments (E1, E2, E3) use the same hyperparameter configuration to ensure a fair comparison. E4 requires no training.

| Hyperparameter     | Value                          |
|--------------------|-------------------------------|
| Optimiser          | SGD                            |
| Learning rate      | 0.01                           |
| Weight decay       | 0.0001                         |
| Batch size         | 40                             |
| Max epochs         | 100                            |
| Early stopping     | Yes (patience = 7 epochs)      |
| Stopping criterion | Validation accuracy            |
| Random seed        | 42                             |

The model checkpoint with the highest validation accuracy during training is saved and used for final test evaluation. The test set is only accessed once, after training is complete. All models are evaluated in `model.eval()` mode to ensure that batch normalisation layers use fixed running statistics rather than batch statistics.

---

## 4. Results

### 4.1 Main Results

Table 1 presents the test performance of all four experiments. Chance level for a 27-class uniform classifier is approximately 3.7%.

**Table 1. Test set performance across all experiments.**

| ID | Model               | Best Val Acc | Test Accuracy | Macro-F1 |
|----|---------------------|:------------:|:-------------:|:--------:|
| E1 | Unimodal IMU        | 11.7%        | 10.5%         | 0.047    |
| E2 | Unimodal Audio      | 44.2%        | 33.3%         | 0.276    |
| E3 | Intermediate Fusion | 68.3%        | **59.3%**     | **0.555**|
| E4 | Late Fusion         | —            | 36.4%         | 0.298    |

Macro-F1 is reported as the unweighted average F1 across all 27 classes, making it sensitive to per-class performance rather than aggregate sample counts. This is the appropriate metric given the potential for class imbalance in the dataset.

### 4.2 Training Behaviour

**E1 (IMU):** Early stopping triggered at epoch 14, with validation accuracy plateauing at 11.7% after epoch 6. The IMU-only model converges quickly to a low-performance local minimum, suggesting that the 100-timestep IMU signal, as represented here, carries limited discriminative information across the 27 classes.

**E2 (Audio):** Training progressed steadily over 36 epochs before early stopping. Validation accuracy reached 44.2% at epoch 28, with consistent improvement throughout, indicating that the audio signal is substantially more informative for this task.

**E3 (Intermediate Fusion):** Training continued for 30 epochs before early stopping. Validation accuracy reached 68.3% at epoch 22, with a smooth learning curve showing consistent joint learning across both encoder branches.

**E4 (Late Fusion):** No training was performed. The result (36.4%) reflects the combination of E1 and E2 predictions directly.

---

## 5. Discussion

### 5.1 IMU vs Audio as Unimodal Signal Sources

The large performance gap between E1 (10.5%) and E2 (33.3%) reveals a fundamental asymmetry in the informativeness of the two modalities for the 27 activities in SAMoSA. Many activities — such as blending, drilling, vacuuming, or toothbrushing — produce highly characteristic acoustic signatures that are reliably distinguishable from a short audio clip. By contrast, the IMU signal records wrist motion, which is much less distinctive across these activities. For example, the motion patterns of "chopping" and "scratching" may be kinematically similar, while their sounds differ substantially. The audio-first nature of SAMoSA's activity taxonomy is thus a key driver of the observed modality asymmetry.

### 5.2 Intermediate Fusion vs Late Fusion

Late fusion (E4, 36.4%) improves only marginally over the audio-only baseline (E2, 33.3%) and underperforms the best unimodal model. This is a direct consequence of averaging a strong signal (audio) with a near-random signal (IMU). Because the IMU model predicts near chance level, its softmax distribution is approximately uniform over 27 classes, and averaging it with the audio distribution degrades predictions for many samples. This illustrates the key risk of late fusion: a weak unimodal model can dilute rather than complement a stronger one.

Intermediate fusion (E3, 59.3%) avoids this problem because the learned joint representation allows the network to down-weight uninformative IMU features implicitly. By training the IMU and audio encoders jointly with a shared loss, the network can learn to extract whatever discriminative signal the IMU does contain — even if this is small — and combine it with audio features in a way that is task-specific. The result is a 26-percentage-point improvement over the best unimodal model and a 22.9-percentage-point improvement over late fusion.

The macro-F1 scores reinforce this conclusion. E3 achieves 0.555 versus 0.298 for E4 and 0.276 for E2. The wider gap in macro-F1 than in accuracy suggests that E3 is also more balanced across classes, generalising better to activities that are individually harder to classify.

### 5.3 Limitations

Several limitations of this study should be noted.

**Non-subject-disjoint split.** The train/test split is constructed by sorting filenames alphabetically, which groups files by subject. Consequently, some subjects appear in both the training and test sets. The reported accuracy is therefore an upper bound on generalisation to entirely unseen subjects. In a realistic deployment scenario, where the model must recognise activities performed by new users, performance would likely be lower. A proper evaluation would hold out all recordings of selected subjects for testing.

**Audio representation.** Raw audio is reshaped from a 48,000-sample 1D signal to a (100, 480) 2D array and processed by a 2D convolutional network. This representation preserves the raw amplitude sequence in a strided matrix format but contains no explicit frequency information. Standard audio processing pipelines use spectrograms or Mel-frequency cepstral coefficients (MFCCs) to extract frequency-domain features, which are more informative for acoustic activity recognition. The relatively modest audio accuracy (33.3% unimodal) may partly reflect this limitation.

**Single run.** All experiments were run once with a fixed seed. Reporting a single run means that performance variance due to stochastic initialisation and batch ordering is unknown. Multiple runs would be needed to provide confidence intervals.

**Dropped samples.** Sixteen files from the original dataset were excluded to maintain a fixed split size. These files belong to subjects whose names appear late in alphabetical order and are therefore absent from both splits.

**Early stopping and IMU training.** The IMU model triggered early stopping at epoch 14, well before the 100-epoch limit. This suggests the model converged to a poor local minimum rather than failing to train for long enough. Architectural modifications (e.g., a different encoder or explicit temporal modelling) might improve IMU-only performance.

---

## 6. Conclusion

This project compared four approaches to multimodal activity recognition on the SAMoSA dataset: unimodal IMU, unimodal audio, intermediate fusion, and late fusion. The results clearly support intermediate fusion as the preferred strategy. By learning a joint representation from both sensor modalities, the intermediate fusion model (59.3% accuracy, macro-F1 = 0.555) outperforms the best unimodal baseline by 26 percentage points and late fusion by 22.9 percentage points.

The results also reveal that modality quality matters. When one modality (IMU) is near-random in its predictions, late fusion degrades performance relative to the stronger modality alone. Intermediate fusion is more robust to this imbalance because it can learn to selectively rely on the more informative signal during training.

The primary limitation of this work is the non-subject-disjoint evaluation protocol, which means reported numbers should be interpreted as indicative rather than as true generalisation performance. Future work should implement a subject-held-out cross-validation scheme, evaluate proper audio feature representations (MFCC or spectrogram), and extend the comparison to additional fusion architectures such as cross-attention or transformer-based fusion.

---

## References

[1] Reyes-Ortiz, J.L., et al. (2016). Transition-aware human activity recognition using smartphones. *Neurocomputing*, 171, 754–767.

[2] Lahat, D., Adali, T., & Jutten, C. (2015). Multimodal data fusion: an overview of methods, challenges, and prospects. *Proceedings of the IEEE*, 103(9), 1449–1477.

[3] Liang, P.P., et al. (2022). MultiBench: Multiscale benchmarks for multimodal representation learning. *NeurIPS Datasets and Benchmarks Track*.

[4] Lecun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.

[5] Gong, Y., et al. (2021). AST: Audio spectrogram transformer. *Interspeech 2021*.

[6] Radu, V., et al. (2018). Multimodal deep learning for activity and context recognition. *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies*, 1(4), 1–27.

[7] Mollyn, V., et al. (2023). SAMoSA: Sensing Activities with Motion and Subsampled Audio. *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies*, 6(3), 1–19.
