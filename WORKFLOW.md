# Workflow for Training and Evaluation

## Overview
This document outlines the complete workflow for training and evaluating the Hand Vein CapsNet model.

## Prerequisites

1. **Dataset Preparation**
   ```bash
   # Organize dataset in the following structure:
   data/ROI_Directory/
   ├── person_001_db1_L1.png
   ├── person_001_db1_L2.png
   ├── person_001_db1_L3.png
   ├── person_001_db1_L4.png
   ├── person_002_db1_L1.png
   └── ...
   ```

2. **Environment Setup**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

## Step 1: Configuration

Edit `configs/default.yaml` to customize hyperparameters:

```yaml
data:
  data_dir: "data/ROI_Directory/"
  img_size: 128
  num_subjects: 276

training:
  batch_size: 32
  epochs: 200
  learning_rate: 0.001

regularization:
  l2_regularization: 0.0001
  margin_loss:
    m_plus: 0.9
    m_minus: 0.1
    lambda_: 0.5
```

## Step 2: Training

### Train Single Fold (LOAO)
```bash
# Train on fold 1 (validate on L1 images)
python train.py --config configs/default.yaml --fold 1
```

### Train All Folds
```bash
# Train on all 4 folds (Leave-One-Acquisition-Out)
python train.py --config configs/default.yaml
```

### With Specific GPU
```bash
python train.py --config configs/default.yaml --gpu 0
```

## Step 3: Evaluation

### Evaluate Single Fold
```bash
python evaluate.py --config configs/default.yaml --fold 1
```

### Evaluate All Folds
```bash
python evaluate.py --config configs/default.yaml
```

## Step 4: Inference

### Single Image Prediction
```bash
python inference.py --model checkpoints/fold_1_best.h5 --image path/to/image.png
```

### Batch Prediction
```bash
python inference.py --model checkpoints/fold_1_best.h5 --batch_dir data/test_images/
```

## Output Files

### Training Outputs
- `checkpoints/fold_X_best.h5`: Best model for fold X
- `checkpoints/fold_X_history.npy`: Training history
- `logs/fold_X/`: TensorBoard logs

### Evaluation Outputs
- `predictions/fold_X_y_true.npy`: True labels
- `predictions/fold_X_y_pred.npy`: Predictions
- `predictions/fold_X_probs.npy`: Probability scores
- `predictions/fold_X_cm.npy`: Confusion matrix

## Monitoring Training

### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir logs/

# Open browser at http://localhost:6006
```

## Performance Analysis

After training, analyze results:

```python
import numpy as np
from src.utils import plot_training_history, plot_confusion_matrix

# Plot training history
history = np.load('checkpoints/fold_1_history.npy', allow_pickle=True).item()
plot_training_history(history, save_path='plots/fold_1_history.png')

# Plot confusion matrix
y_true = np.load('predictions/fold_1_y_true.npy')
y_pred = np.load('predictions/fold_1_y_pred.npy')
plot_confusion_matrix(y_true, y_pred, save_path='plots/fold_1_cm.png')
```

## Cross-Validation Results

Expected LOAO results:

| Fold | Train Acc | Val Acc | Best Epoch |
|------|-----------|---------|------------|
| 1    | TBD       | TBD     | TBD        |
| 2    | TBD       | TBD     | TBD        |
| 3    | TBD       | TBD     | TBD        |
| 4    | TBD       | TBD     | TBD        |
| **Avg** | **TBD** | **TBD** | **TBD**    |

## Troubleshooting

### Out of Memory Error
- Reduce `batch_size` in config (e.g., 32 → 16)
- Reduce `img_size` (though 128 is recommended)

### Poor Training Performance
- Check data loading (verify image paths)
- Adjust learning rate (try 0.0001 or 0.0005)
- Increase epochs
- Enable data augmentation

### Model Convergence Issues
- Verify dataset balance
- Check for corrupted images
- Adjust margin loss parameters

## Advanced Usage

### Custom Model Architecture

Edit `src/capsnet_model.py` to modify:
- Number of convolutional layers
- Filter sizes
- Primary capsule dimensions
- Routing iterations

### Custom Loss Function

Implement in `src/losses.py`:
```python
class CustomCapsuleLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        # Custom loss implementation
        return loss
```

## References

- [Capsule Networks Paper](https://arxiv.org/abs/1710.09829)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Hand Vein Recognition Survey](https://arxiv.org/abs/1606.00101)
