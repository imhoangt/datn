# CHARM-Net v6.6 — WiFi CSI Human Activity Recognition

**CHARM-Net**: Channel, Hardware, Activity Recognition Multi-stream Network  
Đồ án tốt nghiệp — WiFi CSI HAR (Single-person, Research)

---

## Cài đặt

```bash
pip install -r requirements.txt
pip install torch-ema seaborn
```

---

## Luồng sử dụng

### 1. Chuẩn bị dữ liệu

**Data tự thu (BCM4366c0 / Nexmon CSI):**
```bash
# Đặt file .pcap vào data/raw/own/
# Cấu trúc: data/raw/own/person_P0/room_01/setup_01/trajectory_Tr0/{activity}_rep00_rx{0,1,2}.pcap

python scripts/preprocess_own.py --folds 0 1 2 3 4 5 6 7
```

**XRF55 (Intel 5300):**
```bash
# Đặt file .npy/.mat vào data/raw/xrf55/{activity}/
python scripts/preprocess_xrf55.py --folds 0 1 2 3 4 5 6 7
```

**Exposing the CSI (AX-CSI, 802.11ax):**
```bash
# Đặt file vào data/raw/exposing/{activity}/
python scripts/preprocess_exposing.py --folds 0 1 2
```

> **Chưa có data?** Dùng mock data để test pipeline:
> ```python
> from src.utils.mock_generator import generate_mock_dataset_hdf5
> generate_mock_dataset_hdf5('data/processed/own/fold_00.h5')
> ```

### 2. Kiểm tra preprocessing (QUAN TRỌNG)

```bash
cd notebooks
python 02_preprocessing_verify.py
# → outputs/figures/dfs_verification.png   ← kiểm tra signature từng class
# → outputs/figures/amplitude_verification.png
```

### 3. Training

```bash
# LOSO tất cả 8 fold
python scripts/train.py --dataset own --fold all

# Single fold
python scripts/train.py --dataset own --fold 0

# XRF55
python scripts/train.py --dataset xrf55 --fold all

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
charm_net/
├── configs/             YAML configs
├── src/
│   ├── models/          CHARMNet v6.6 (stems, encoders, fusion, readout, heads)
│   ├── preprocessing/   Pipeline v6.6 + parsers (Nexmon/Intel5300/AX-CSI)
│   ├── datasets/        PyTorch Dataset + DataLoader builders
│   ├── training/        Loss functions, trainer, LOSO runner
│   ├── evaluation/      Metrics, benchmark table
│   └── utils/           Logger, visualization, mock generator
├── scripts/             Entry points (preprocess, train, evaluate)
├── notebooks/           01-04 analysis notebooks
├── data/                raw/ + processed/ (HDF5)
└── outputs/             checkpoints/ + logs/ + figures/
```

---

## Thông số kỹ thuật

| Component | Value |
|-----------|-------|
| Model params | ~495K |
| VRAM usage | ~50–150 MB |
| Input X_amp | [B, 350, 52, 3, 4] float32 |
| Input X_dfs | [B, 28, 128, 3] float32 |
| Output | P_8class [B, 8], sum=1 |
| Protocol | LOSO-Person 8-fold |
| Primary metric | Macro F1 (8 classes) |

---

## Pipeline v6.6 — Các bước chính

```
A1-A4  Parse .pcap → validate → select 52 subs → detect gaps
B1     AGC normalization (per-packet, per-antenna)
B2     Hampel outlier detection + PCHIP Cartesian repair
B3     Multi-RX timestamp alignment (cross-correlation)
B4     PCHIP resample → common time grid (FIX v6.6: temporal alignment)
G1     Reference antenna selection (argmin temporal variance)
C      LOSO split (recording-level, anti-leakage assertions)
Stats  Welford Z-score + Fisher subcarrier selection (single pass)
D      Sliding window T=350, stride=175 (~9 windows/recording)
F      Amplitude branch: |H| → LP 30Hz → median removal → Z-score
G      DFS branch: conj× → static remove → BP [0.5,80Hz] → Fisher select → STFT → log norm
H      Branch augmentation: CutMix-Time (H1b), masking (H1a, H2)
I      Sanity checks + write HDF5
```

---

## Citation

Nếu dùng pipeline này trong nghiên cứu, vui lòng cite:
- Widar3.0 [Chi et al., MobiSys 2019]
- XRF55 [Wang et al., IMWUT 2024]
- Exposing the CSI [Cominelli et al., PerCom 2023]
