# FL Framework Comparison: TFF vs Flower for Bug Prediction

## Study Overview

This replication package compares **TensorFlow Federated (TFF)** and **Flower** frameworks for federated bug prediction, evaluating:

### Technical Factors

1. **Communication Cost**
   - Total bytes sent/received
   - Per-round communication overhead

2. **Security (Poisoning Attack Robustness)**
   - **Model Poisoning**: Adding +100 to gradients from malicious clients
   - **Data Poisoning**: Flipping labels (buggy ↔ non-buggy) for malicious clients
   - **Metrics**: F1 Score, Accuracy, Precision, Recall, Loss (before/after attacks)

---

## Quick Start: Google Colab (Recommended)

### Prerequisites
- Google account for Colab access
- Dataset file: `dataset_pairs_1_.json`

### Step-by-Step Instructions

#### STEP 1: Run TFF Experiments

1. Open Google Colab: https://colab.research.google.com
2. Upload `01_TFF_Bug_Prediction.ipynb`
   - File → Upload notebook → Select the file
3. Enable GPU (recommended):
   - Runtime → Change runtime type → GPU → Save
4. Run all cells:
   - Runtime → Run all
5. When prompted, upload `dataset_pairs_1_.json`
6. Wait for all experiments to complete (~15-20 minutes)
7. Download the output file: `tff_results.json`

**Experiments Run:**
- Baseline (no attack)
- Model Poisoning at 10%, 20%, 30% malicious clients
- Data Poisoning at 10%, 20%, 30% malicious clients

---

#### STEP 2: Run Flower Experiments

1. Open a new Colab tab
2. Upload `02_Flower_Bug_Prediction.ipynb`
3. Enable GPU:
   - Runtime → Change runtime type → GPU → Save
4. Run all cells:
   - Runtime → Run all
5. When prompted, upload `dataset_pairs_1_.json`
6. Wait for all experiments to complete (~10-15 minutes)
7. Download the output file: `flower_results.json`

**Experiments Run:**
- Baseline (no attack)
- Model Poisoning at 10%, 20%, 30% malicious clients
- Data Poisoning at 10%, 20%, 30% malicious clients

---

#### STEP 3: Run Comparison Analysis

1. Open a new Colab tab
2. Upload `03_Comparison_Analysis.ipynb`
3. Run all cells:
   - Runtime → Run all
4. When prompted, upload both result files:
   - `tff_results.json`
   - `flower_results.json`
5. Download all output files:
   - `comparison_results.json`
   - `communication_comparison.png`
   - `model_poisoning_comparison.png`
   - `data_poisoning_comparison.png`
   - `security_comprehensive.png`

---

## Package Contents

```
fl_comparison_package/
├── README.md                          # This file
├── data/
│   └── dataset_pairs_1_.json          # Bug prediction dataset
├── notebooks/
│   ├── 01_TFF_Bug_Prediction.ipynb    # TFF experiments (STEP 1)
│   ├── 02_Flower_Bug_Prediction.ipynb # Flower experiments (STEP 2)
│   └── 03_Comparison_Analysis.ipynb   # Comparison (STEP 3)
├── comparison_metrics/
│   ├── technical_comparison.py        # Communication & security analysis
│   ├── flexibility_comparison.py      # Flexibility metrics
│   └── statistical_analysis.py        # Statistical tests
├── tff_implementation/                # TFF source code
├── flower_implementation/             # Flower source code
├── utils/                             # Shared utilities
└── requirements_*.txt                 # Dependencies
```

---

## Security Analysis Details

### Model Poisoning Attack
```
Attack: Add +100 to all gradient values from malicious clients

Implementation:
- Malicious clients compute normal gradients
- Before sending to server, add 100 to all gradient values
- Server aggregates poisoned + clean gradients via FedAvg

Effect: Corrupts global model by injecting large gradient values
```

### Data Poisoning Attack
```
Attack: Flip labels for malicious clients' data

Implementation:
- Malicious clients flip their training labels:
  - buggy (1) → non-buggy (0)
  - non-buggy (0) → buggy (1)
- Clients train on corrupted labels
- Server aggregates normally

Effect: Model learns incorrect patterns from poisoned data
```

### Metrics Measured

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Precision | True positives / (True positives + False positives) |
| Recall | True positives / (True positives + False negatives) |
| F1 Score | Harmonic mean of precision and recall |
| Loss | Binary cross-entropy loss |

### Malicious Client Fractions Tested
- 10% (1 out of 10 clients)
- 20% (2 out of 10 clients)
- 30% (3 out of 10 clients)

---

## Expected Output Format

### tff_results.json / flower_results.json
```json
{
  "framework": "TensorFlow Federated",
  "config": {
    "num_clients": 10,
    "num_rounds": 10,
    "malicious_fractions": [0.1, 0.2, 0.3]
  },
  "baseline_metrics": {
    "accuracy": 0.85,
    "f1_score": 0.82,
    "precision": 0.84,
    "recall": 0.80,
    "loss": 0.35
  },
  "communication": {
    "total_bytes": 50000000,
    "per_round_bytes": 5000000
  },
  "poisoning_analysis": {
    "model_poisoning": {
      "10pct": {
        "metrics_after": {...},
        "accuracy_drop": 0.05,
        "f1_drop": 0.06
      },
      "20pct": {...},
      "30pct": {...}
    },
    "data_poisoning": {
      "10pct": {...},
      "20pct": {...},
      "30pct": {...}
    }
  }
}
```

---

## Troubleshooting

### Memory Issues (OOM)
If you encounter out-of-memory errors:
1. Runtime → Restart runtime
2. Reduce `num_clients` from 10 to 5
3. Reduce `num_rounds` from 10 to 5

### Installation Errors
If package installation fails:
```python
# For TFF notebook, run:
!pip install tensorflow==2.14.0 tensorflow-federated==0.64.0 --quiet

# For Flower notebook, run:
!pip install flwr torch --quiet
```

### Dataset Upload Issues
- Ensure file is named exactly: `dataset_pairs_1_.json`
- File size should be ~2-3 MB
- Check JSON is valid (not corrupted)

### Slow Training
- Ensure GPU is enabled: Runtime → Change runtime type → GPU
- Check GPU allocation: `!nvidia-smi`

---

## Configuration Options

You can modify the experiment parameters in each notebook:

```python
CONFIG = {
    'num_clients': 10,           # Number of FL clients
    'num_rounds': 10,            # Training rounds
    'malicious_fractions': [0.1, 0.2, 0.3]  # Attack intensities
}
```

---

## Citation

If you use this replication package, please cite:
```
@misc{fl_comparison_2024,
  title={Comparing TensorFlow Federated and Flower for Federated Bug Prediction},
  author={[Author]},
  year={2024}
}
```

---

## Contact

For questions or issues, please open an issue in the repository.
