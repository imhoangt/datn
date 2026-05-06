# CHARM-Net v7.1 — WiFi CSI Human Activity Recognition

**CHARM-Net**: Channel, Hardware, Activity Recognition Multi-stream Network
Đồ án tốt nghiệp — WiFi CSI HAR (Single-person, 8 classes, 3 datasets)

---

## Kiến trúc (Architecture)

```
X_amp [B, 350, 52, M=3, A=4]        X_dfs [B, 28, 128, M=3]
          ↓ per RX, shared weights             ↓
    AmpStem (FiLM scale×h+shift)    DfsStem (FiLM scale×h+shift)
    AmpEncoder (DropPath rate=0.1)  DfsEncoder (DropPath rate=0.1)
          └────── CrossModalFusion (gated cross-attention) ──────┘
                          Z_m [B, 350, D=80]
              ReceiverContextFusion (cross-RX residual, gate init=0)
              ReceiverFusion (mean+var attention → weighted sum)
              DualScaleReadout
                  ├── fast: v_mean + v_energy
                  └── slow: BlurPool stride=4 → depthwise conv → mean
              HierarchicalHeads
                  ├── head_unified → logit [B,8] → p_8class [B,8]  (main)
                  ├── head_occ    → p_occ  [B,1]                    (aux)
                  ├── head_dyn    → p_dyn  [B,1]                    (aux)
                  └── head_act    → p_act  [B,6]                    (aux)
```

### Thông số kỹ thuật

| Component | Params |
|---|---|
| amp_stem | ~17,200 |
| dfs_stem | ~5,200 |
| amp_encoder | ~185,000 |
| dfs_encoder | ~75,000 |
| cross_fusion | ~52,000 |
| recv_context | ~6,500 |
| recv_fusion | ~1,450 |
| readout | ~20,000 |
| heads | ~1,300 |
| **Total** | **~515K** |

---

## 3 Datasets

| Dataset | Thiết bị | Receivers | Subcarriers | Rate | Protocol |
|---|---|---|---|---|---|
| own | BCM4366c0 / Nexmon CSI | 3 × ASUS RT-AC86U | 52 | 350Hz | LOSO 8-fold |
| xrf55 | Intel 5300 / linux-80211n-csitool | 3 sites | 30 | ~200Hz | LOSO 8-fold |
| expose_csi | Broadcom 43684 / AX-CSI (802.11ax) | 3 collectors | 52 (of 2048) | ~150Hz | Blocked 5-fold |

### Classes per dataset

**Own dataset** (8 classes):

| ID | Class | Mô tả |
|---|---|---|
| 0 | walk | Đi bộ |
| 1 | run | Chạy |
| 2 | stand_up | Đứng lên |
| 3 | sit_down | Ngồi xuống |
| 4 | bend_to_pick | Cúi nhặt đồ |
| 5 | fall | Ngã |
| 6 | lay_still | Nằm yên |
| 7 | empty_room | Phòng trống |

**XRF55** (8 classes):

| ID | Class | XRF55 activity ID |
|---|---|---|
| 0 | walk | 39 |
| 1 | run | 35 |
| 2 | jump | 34 |
| 3 | stand_up | 37 |
| 4 | sit_down | 36 |
| 5 | bend_to_pick | 9 |
| 6 | fall | 33 |
| 7 | clap | 32 |

**Expose CSI** (8 classes, activity code letter):

| ID | Class | Code |
|---|---|---|
| 0 | walk | A |
| 1 | run | B |
| 2 | jump | C |
| 3 | sit_still | D |
| 4 | squat | K |
| 5 | lay_down | I |
| 6 | clap | H |
| 7 | empty | E |

---

## Cài đặt

```bash
pip install -r requirements.txt
pip install torch-ema seaborn
```

---

## Luồng sử dụng

### 1. Chuẩn bị dữ liệu

**Own (BCM4366c0 / Nexmon CSI):**
```bash
# Cấu trúc: data/raw/own/person_P{n}/room_{r}/setup_{s}/trajectory_Tr{t}/{act}_rep{:02d}_rx{0,1,2}.pcap
python scripts/preprocess_own.py --folds 0 1 2 3 4 5 6 7
```

**XRF55 (Intel 5300, .dat binary):**
```bash
# Cấu trúc: data/raw/xrf55/rx_01/{volunteer}/{class_id}_{vol}_{rep}.dat
#           data/raw/xrf55/rx_02/{volunteer}/{class_id}_{vol}_{rep}.dat
#           data/raw/xrf55/rx_03/{volunteer}/{class_id}_{vol}_{rep}.dat
python scripts/preprocess_xrf55.py --folds 0 1 2 3 4 5 6 7
```

**Expose CSI (AX-CSI, 802.11ax, .mat):**
```bash
# Cấu trúc: data/raw/expose_csi/rx_01/S7a_{X}.mat   X ∈ {A,B,C,D,K,I,H,E}
#           data/raw/expose_csi/rx_02/S7b_{X}.mat
#           data/raw/expose_csi/rx_03/S7c_{X}.mat
python scripts/preprocess_exposing.py --folds 0 1 2 3 4
```

> **Chưa có data?** Dùng mock data để test pipeline:
> ```python
> from src.utils.mock_generator import generate_mock_dataset_hdf5
> generate_mock_dataset_hdf5('data/processed/own/fold_00.h5')
> ```

### 2. Kiểm tra preprocessing

```bash
cd notebooks
python 02_preprocessing_verify.py
# → outputs/figures/dfs_verification.png
# → outputs/figures/amplitude_verification.png
```

### 3. Training

```bash
# Own dataset — LOSO 8 folds
python scripts/train.py --dataset own --fold all

# Single fold
python scripts/train.py --dataset own --fold 0

# XRF55
python scripts/train.py --dataset xrf55 --fold all

# Expose CSI (5-fold)
python scripts/train.py --dataset exposing --fold all

# Resume từ fold 3
python scripts/train.py --dataset own --fold all --resume 3
```

### 4. Evaluation

```bash
python scripts/evaluate.py --dataset own --fold all --plot
python scripts/evaluate.py --dataset xrf55 --fold all --plot
python scripts/evaluate.py --dataset exposing --fold all --plot
```

### 5. So sánh 3 datasets

```bash
cd notebooks && python 04_results_comparison.py
```

---

## Cấu trúc thư mục

```
DATN/
├── configs/
│   ├── base.yaml                  Shared defaults
│   ├── dataset_own.yaml           Own-data overrides
│   ├── dataset_xrf55.yaml         XRF55 overrides
│   └── dataset_exposing.yaml      Expose CSI overrides
├── src/
│   ├── models/
│   │   ├── stems.py               AmpStem, DfsStem (FiLM)
│   │   ├── encoders.py            AmpEncoder, DfsEncoder (DropPath)
│   │   ├── fusion.py              CrossModal, ReceiverContext, ReceiverFusion
│   │   ├── readout.py             DualScaleReadout (fast+slow)
│   │   ├── heads.py               HierarchicalHeads, TemperatureScaler
│   │   └── charm_net.py           CHARMNet v7.1 + build_model()
│   ├── preprocessing/
│   │   ├── common.py              AGC, Hampel, PCHIP, sliding window, stats
│   │   ├── pipeline_own.py        Own-data pipeline
│   │   ├── pipeline_xrf55.py      XRF55 pipeline (binary .dat parser)
│   │   ├── pipeline_exposing.py   Expose CSI pipeline (blocked 5-fold)
│   │   └── parsers/
│   │       ├── nexmon_parser.py   .pcap → CSI (own dataset)
│   │       ├── intel5300_parser.py .dat binary → CSI (XRF55)
│   │       └── axcsi_parser.py    .mat → CSI (expose_csi)
│   ├── datasets/
│   │   └── base_dataset.py        PyTorch Dataset + build_loaders()
│   ├── training/
│   │   ├── losses.py              compute_loss, train_one_epoch, evaluate_with_ema
│   │   └── loso_runner.py         run_loso(), TrainLogger, checkpointing
│   ├── evaluation/
│   │   └── metrics.py             compute_fold_metrics, benchmark table
│   └── utils/
│       ├── logger.py              CSV + TensorBoard
│       ├── visualization.py       Confusion matrix, class distribution
│       └── mock_generator.py      Mock HDF5 for pipeline testing
├── scripts/
│   ├── preprocess_own.py
│   ├── preprocess_xrf55.py
│   ├── preprocess_exposing.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   ├── 01_eda.py
│   ├── 02_preprocessing_verify.py
│   ├── 03_training_curves.py
│   └── 04_results_comparison.py
├── data/
│   ├── raw/                       Dữ liệu thô (không commit)
│   └── processed/                 HDF5 đã xử lý (không commit)
└── outputs/
    ├── checkpoints/               Model checkpoints per fold
    ├── logs/                      CSV + TensorBoard logs
    └── figures/                   Visualization outputs
```

---

## Pipeline v7.1 — Các bước chính

```
A1-A4  Parse raw file → validate → select subcarriers → detect gaps
B1     AGC normalization (per-packet, per-antenna)
B2     Hampel outlier detection + PCHIP Cartesian repair
B3     Multi-RX timestamp alignment (cross-correlation) [own only]
B4     PCHIP resample → common time grid
G1     Reference antenna selection (argmin temporal variance)
C      LOSO split [own/xrf55] / Blocked 5-fold [expose_csi]
Stats  Welford Z-score + Fisher subcarrier selection (single pass over train)
D      Sliding window T=350, stride=175 (~9 windows/recording at 350Hz)
F      Amplitude branch: |H| → LP 30Hz → median removal → Z-score
G      DFS branch: conj× → static remove → BP [0.5,80Hz] → STFT → log norm
H      Offline augmentation: E-stage (CSI) + H-stage (features), 3 aug copies
I      Write HDF5 (train: 4× data, val/test: original only)
```

---

## Training Phases

| Phase | Epochs | LR range | aug_level | EMA decay |
|---|---|---|---|---|
| 1 Warmup | 5 | 1e-5 → 1e-4 (linear) | no_aug | 0.999 |
| 2 Main cosine | 50 | 1e-4 → 1e-6 | full (+ CutMix-Time) | 0.999 |
| 3 Fine-tune | 20 | 1e-5 → 1e-7 | no_aug | 0.9999 |

Post-training: Temperature scaling calibration on validation set (LBFGS).

---

## Design Decisions

| Decision | Rationale |
|---|---|
| FiLM `scale×h+shift` (not `h+embed`) | Additive embed cannot modulate amplitude gain differences across hardware (Perez 2018) |
| DropPath rate 0.1 | Small dataset → higher regularization; Huang ECCV2016 recommends 0.1-0.2 |
| Weight sharing across receivers | Reduces overfitting; receivers see same modality (Set Transformer, Deep Sets theory) |
| ReceiverContextFusion gate init=0 | Safe warm-start: tanh(0)=0, no effect at epoch 0 |
| head_unified + F.log_softmax | Numerically stable L_main; avoids log(softmax(x)) double approximation |
| Blocked 5-fold for expose_csi | Single volunteer → no LOSO; chronological split prevents temporal leakage |
| CutMix-Time on-the-fly (batch-level) | Must be applied after DataLoader to pair different samples; offline aug is per-sample |

---

## Citation

Nếu sử dụng pipeline này, vui lòng cite:
- Widar3.0: Chi et al., MobiSys 2019
- XRF55: Wang et al., IMWUT 2024
- Exposing the CSI: Cominelli et al., PerCom 2023
- FiLM: Perez et al., AAAI 2018
- Stochastic Depth: Huang et al., ECCV 2016
