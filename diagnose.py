"""Diagnostic et troubleshooting pour le modèle CapsNet"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from src.data_loader import HandVeinDataLoader
from src.capsnet_model import build_capsnet, build_simple_capsnet
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

def diagnose_data():
    """Diagnostique les données du dataset."""
    print("\n" + "="*60)
    print("DATA DIAGNOSTIC")
    print("="*60)
    
    data_loader = HandVeinDataLoader()
    X, y, filenames = data_loader.load_dataset()
    
    print(f"\nDataset Statistics:")
    print(f"  Total images: {len(X)}")
    print(f"  Image shape: {X[0].shape}")
    print(f"  Image dtype: {X[0].dtype}")
    print(f"  Image value range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"  Unique subjects: {len(np.unique(y))}")
    print(f"  Images per subject: {len(X) // len(np.unique(y))}")
    
    # Check for empty or corrupted images
    empty_count = 0
    for i, img in enumerate(X):
        if img.std() < 0.01:  # Very low variance = likely empty/corrupted
            empty_count += 1
            if empty_count <= 3:
                print(f"    ⚠️ Image {i} ({filenames[i]}) has very low variance: {img.std():.6f}")
    
    if empty_count > 0:
        print(f"  Total images with low variance: {empty_count}")
    
    # Check data distribution per acquisition
    print("\nData Distribution by Acquisition:")
    for acq in ['L1', 'L2', 'L3', 'L4']:
        count = sum(1 for fn in filenames if acq in fn)
        print(f"  {acq}: {count} images")
    
    # Check unique subjects per acquisition
    print("\nUnique Subjects per Acquisition:")
    for acq in ['L1', 'L2', 'L3', 'L4']:
        subjects = set(fn.split('_')[1] for fn in filenames if acq in fn)
        print(f"  {acq}: {len(subjects)} subjects")
    
    return X, y, filenames


def test_model_forward_pass():
    """Test si le modèle peut faire une prédiction forward."""
    print("\n" + "="*60)
    print("MODEL FORWARD PASS TEST")
    print("="*60)
    
    try:
        model = build_capsnet(num_subjects=276)
        print("\n✅ Model built successfully")
        print(f"\nModel Summary:")
        model.summary()
        
        # Test with dummy data
        dummy_input = np.random.randn(4, 128, 128, 1).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)
        print(f"\n✅ Forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output value range: [{output.min():.4f}, {output.max():.4f}]")
        
        return model
    except Exception as e:
        print(f"\n❌ Model error: {e}")
        return None


def test_loss_function():
    """Test la fonction de loss."""
    print("\n" + "="*60)
    print("LOSS FUNCTION TEST")
    print("="*60)
    
    try:
        from src.losses import MarginLoss
        
        # Create dummy predictions and labels
        batch_size = 8
        num_subjects = 276
        
        y_true = tf.constant(
            [[1 if i == j else 0 for j in range(num_subjects)] 
             for i in range(batch_size)], 
            dtype=tf.float32
        )
        y_pred = tf.constant(
            np.random.uniform(0, 1, (batch_size, num_subjects)),
            dtype=tf.float32
        )
        
        loss_fn = MarginLoss(m_plus=0.9, m_minus=0.1, lambda_=0.5)
        loss = loss_fn(y_true, y_pred)
        
        print(f"\n✅ Loss computation successful")
        print(f"  Batch size: {batch_size}")
        print(f"  Num subjects: {num_subjects}")
        print(f"  Loss value: {loss.numpy():.6f}")
        
        if loss.numpy() > 100:
            print(f"  ⚠️ Warning: Loss seems very high")
        
    except Exception as e:
        print(f"\n❌ Loss function error: {e}")


def test_training_loop():
    """Test un small training loop."""
    print("\n" + "="*60)
    print("TRAINING LOOP TEST (10 batches)")
    print("="*60)
    
    try:
        # Load small dataset
        data_loader = HandVeinDataLoader()
        X_train, y_train, X_val, y_val = data_loader.load_loao_fold(fold_num=4)
        
        # Use only first 64 samples for speed
        X_train = X_train[:64]
        y_train = y_train[:64]
        X_val = X_val[:32]
        y_val = y_val[:32]
        
        y_train_onehot = tf.keras.utils.to_categorical(y_train, 276)
        y_val_onehot = tf.keras.utils.to_categorical(y_val, 276)
        
        # Build model
        model = build_simple_capsnet(num_subjects=276)
        
        # Train
        print(f"\nTraining on {len(X_train)} samples...")
        history = model.fit(
            X_train, y_train_onehot,
            validation_data=(X_val, y_val_onehot),
            epochs=5,
            batch_size=16,
            verbose=1
        )
        
        print(f"\n✅ Training successful")
        print(f"  Final train loss: {history.history['loss'][-1]:.6f}")
        print(f"  Final train acc: {history.history['accuracy'][-1]:.6f}")
        print(f"  Final val loss: {history.history['val_loss'][-1]:.6f}")
        print(f"  Final val acc: {history.history['val_accuracy'][-1]:.6f}")
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all diagnostics."""
    print("\n" + "="*70)
    print("HAND VEIN CAPSNET - DIAGNOSTIC SUITE")
    print("="*70)
    
    # Run tests
    X, y, filenames = diagnose_data()
    model = test_model_forward_pass()
    test_loss_function()
    test_training_loop()
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
