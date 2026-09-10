# Quick Start Guide

## Installation

```bash
# Clone repository
git clone https://github.com/habibkademdz-design/hand-vein-capsnet.git
cd hand-vein-capsnet

# Create environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1. Prepare Dataset

Place your ROI images in `data/ROI_Directory/`:

```
data/ROI_Directory/
├── person_001_db1_L1.png
├── person_001_db1_L2.png
├── person_001_db1_L3.png
├── person_001_db1_L4.png
└── ...
```

### 2. Train Model

```bash
# Train on all LOAO folds
python train.py --config configs/default.yaml

# Or train single fold
python train.py --config configs/default.yaml --fold 1
```

### 3. Evaluate

```bash
# Evaluate all folds
python evaluate.py --config configs/default.yaml

# Evaluate single fold
python evaluate.py --config configs/default.yaml --fold 1
```

### 4. Inference

```bash
# Single image
python inference.py --model checkpoints/fold_1_best.h5 --image test_image.png

# Batch images
python inference.py --model checkpoints/fold_1_best.h5 --batch_dir data/test/
```

## Configuration

Edit `configs/default.yaml`:

```yaml
data:
  img_size: 128          # Image dimensions
  num_subjects: 276      # Number of subjects

training:
  batch_size: 32         # Batch size
  epochs: 200            # Max epochs
  learning_rate: 0.001   # Learning rate

regularization:
  l2_regularization: 1e-4
  margin_loss:
    m_plus: 0.9          # Positive margin
    m_minus: 0.1         # Negative margin
    lambda_: 0.5         # Loss weight
```

## Monitoring Training

```bash
# Start TensorBoard
tensorboard --logdir logs/
```

Open http://localhost:6006 in your browser.

## Expected Results

- **Accuracy**: > 95% on LOAO validation
- **Training Time**: ~30-60 minutes per fold (GPU)
- **Model Size**: ~15 MB
- **Inference Time**: ~50-100ms per image

## Troubleshooting

**Q: Out of memory error**
A: Reduce batch_size to 16 or 8

**Q: Training too slow**
A: Use GPU with `--gpu 0`, increase batch_size

**Q: Poor accuracy**
A: Check data quality, verify person IDs in filenames, increase epochs

See [WORKFLOW.md](WORKFLOW.md) for detailed guide.
