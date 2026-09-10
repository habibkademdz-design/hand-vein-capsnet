# Hand Dorsal Vein Recognition using Capsule Networks

This project implements an improved Capsule Network (CapsNet) architecture for biometric identification based on hand dorsal vein patterns. The system achieves high accuracy in person identification using vein recognition with dynamic routing.

## 📋 Dataset Structure

The dataset is organized as follows:

```
data/ROI_Directory/
├── person_001_db1_L1
├── person_001_db1_L2
├── person_001_db1_L3
├── person_001_db1_L4
├── person_002_db1_L1
├── ...
└── person_276_db1_L4
```

**Important:** L1, L2, L3, L4 represent 4 different acquisitions of the same person's hand vein, NOT different classes.

- **XXX** = Person identity (001 to 276)
- **db1** = Database 1
- **Z** = Acquisition number (1 to 4)
- **Classification label** = XXX only

## 🏗️ Improved Architecture

```
ROI grayscale (128 × 128 × 1)
        │
        ▼
   Data Augmentation
   (Rotation, Shift, Elastic Deformation)
        │
        ▼
   Normalization [0,1]
        │
        ▼
┌─────────────────────────────┐
│ Conv2D 5×5, 64 filters      │
│ + BatchNorm + ReLU          │
│ Output: 124 × 124 × 64      │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Conv2D 3×3, 128 filters     │
│ + BatchNorm + ReLU          │
│ + MaxPool 2×2               │
│ Output: 61 × 61 × 128       │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Conv2D 3×3, 256 filters     │
│ + BatchNorm + ReLU          │
│ Output: 59 × 59 × 256       │
└────────────┬────────────────┘
             │
             ▼
┌──────────────────────────────┐
│ Primary Capsules Layer       │
│ Conv 9×9, stride=2           │
│ 32 types × 8 dimensions      │
│ Squashing activation         │
│ Output: 20 × 20 × 256 (u)    │
└────────────┬─────────────────┘
             │
             ▼
    ┌─────────────────┐
    │ Dynamic Routing │
    │ (3-4 iterations)│
    └────────┬────────┘
             │
             ▼
┌──────────────────────────────┐
│ Class Capsules (Nc × 16D)    │
│ Nc = 276 (1 per subject)     │
│ Output: 276 × 16             │
└────────────┬─────────────────┘
             │
             ▼
    Capsule Lengths ||c_i||
    + Margin Loss + L2 Reg
             │
             ▼
    Predicted Identity ŷ
```

## ✨ Key Improvements

1. **Data Augmentation**
   - Rotation (±10°)
   - Shift (±5 pixels)
   - Elastic deformation
   - Zoom and flipping

2. **BatchNormalization**
   - Added after each Conv2D layer
   - Stabilizes training
   - Reduces overfitting

3. **Enhanced Architecture**
   - Additional Conv2D layer (256 filters)
   - MaxPooling for spatial reduction
   - Better feature extraction capability

4. **Advanced Regularization**
   - Margin loss with customizable thresholds
   - L2 weight regularization
   - Routing entropy regularization

5. **Proper Validation Strategy**
   - Leave-One-Acquisition-Out (LOAO) cross-validation
   - Utilizes all 4 acquisitions per person
   - More robust evaluation

## 📦 Requirements

```
tensorflow >= 2.10
numpy
scipy
scikit-learn
matplotlib
opencv-python
pandas
```

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/habibkademdz-design/hand-vein-capsnet.git
cd hand-vein-capsnet
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
# Place your ROI images in:
mkdir -p data/ROI_Directory/
# Copy person_XXX_db1_LZ.png files here
```

### 3. Train the Model

```bash
python train.py --config configs/default.yaml
```

### 4. Evaluate

```bash
python evaluate.py --model_path checkpoints/best_model.h5
```

## 📊 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 32-64 | Training batch size |
| `learning_rate` | 0.001 | Adam optimizer LR |
| `epochs` | 100-200 | Max training epochs |
| `margin_loss_m_plus` | 0.9 | Positive margin |
| `margin_loss_m_minus` | 0.1 | Negative margin |
| `lambda_` | 0.5 | Margin loss weight |
| `l2_regularization` | 1e-4 | L2 weight penalty |
| `routing_iterations` | 3-4 | Dynamic routing steps |
| `early_stopping_patience` | 15 | Epochs for early stop |

## 🔄 Cross-Validation Strategy

**Leave-One-Acquisition-Out (LOAO):**

```
Fold 1: Train on [L2, L3, L4] → Validate on [L1]
Fold 2: Train on [L1, L3, L4] → Validate on [L2]
Fold 3: Train on [L1, L2, L4] → Validate on [L3]
Fold 4: Train on [L1, L2, L3] → Validate on [L4]
```

## 📈 Expected Performance

- **Accuracy**: > 95% (depends on dataset quality)
- **Inference Time**: ~50-100ms per image
- **Model Size**: ~10-15 MB

## 🏗️ Project Structure

```
hand-vein-capsnet/
├── data/
│   └── ROI_Directory/          # Place your dataset here
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and augmentation
│   ├── capsnet_model.py        # CapsNet architecture
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── primary_capsules.py # Primary capsule layer
│   │   └── routing.py          # Dynamic routing algorithm
│   ├── losses.py               # Custom loss functions
│   └── utils.py                # Utility functions
├── configs/
│   └── default.yaml            # Default configuration
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── results_visualization.ipynb
├── requirements.txt
├── .gitignore
├── LICENSE (MIT)
└── README.md
```

## 🔬 Advanced Features

- **Visualization Module**: Display vein patterns and activations
- **Inference Pipeline**: Real-time person identification
- **Metrics Dashboard**: Training curves and performance statistics
- **Model Interpretability**: Capsule activation analysis

## 📚 References

- Sabour, S., Frosst, N., & Hinton, G. E. (2017). "Dynamic Routing Between Capsules"
- Hand vein biometrics literature
- Capsule Networks for medical imaging

## 👨‍💻 Author

Habib Kadem Dz (habibkademdz-design)

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact & Support

For questions or issues, please open an issue on GitHub.

---

**Last Updated**: September 2026
