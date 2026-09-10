"""Training script for Hand Vein CapsNet"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
)

from src.data_loader import HandVeinDataLoader
from src.capsnet_model import build_capsnet
from src.utils import load_config, create_directories, plot_training_history


def train_loao_fold(fold_num, config):
    """Train model on single LOAO fold.
    
    Args:
        fold_num (int): Fold number (1-4)
        config (dict): Configuration dictionary
    """
    print(f"\n{'='*60}")
    print(f"Training LOAO Fold {fold_num}")
    print(f"{'='*60}")
    
    # Load data
    data_loader = HandVeinDataLoader(
        data_dir=config['data']['data_dir'],
        img_size=config['data']['img_size'],
        num_subjects=config['data']['num_subjects']
    )
    
    X_train, y_train, X_val, y_val = data_loader.load_loao_fold(fold_num)
    
    # Convert to one-hot
    y_train_onehot = keras.utils.to_categorical(y_train, config['data']['num_subjects'])
    y_val_onehot = keras.utils.to_categorical(y_val, config['data']['num_subjects'])
    
    # Create model
    model = build_capsnet(
        num_subjects=config['data']['num_subjects'],
        l2_reg=config['regularization']['l2_regularization']
    )
    
    print(f"\nModel Summary:")
    model.summary()
    
    # Create callbacks
    callbacks = []
    
    # Checkpoint
    if config['training']['checkpoint']['enabled']:
        checkpoint_dir = config['training']['checkpoint']['save_dir']
        create_directories([checkpoint_dir])
        checkpoint_path = os.path.join(checkpoint_dir, f'fold_{fold_num}_best.h5')
        callbacks.append(ModelCheckpoint(
            checkpoint_path,
            monitor=config['training']['checkpoint']['monitor'],
            save_best_only=config['training']['checkpoint']['save_best_only'],
            verbose=1
        ))
    
    # Early stopping
    if config['training']['early_stopping']['enabled']:
        callbacks.append(EarlyStopping(
            monitor=config['training']['early_stopping']['monitor'],
            patience=config['training']['early_stopping']['patience'],
            mode=config['training']['early_stopping']['mode'],
            verbose=1
        ))
    
    # Learning rate reduction
    callbacks.append(ReduceLROnPlateau(
        monitor='val_loss',
        factor=config['training']['learning_rate_decay'],
        patience=5,
        verbose=1
    ))
    
    # TensorBoard
    if config['logging']['tensorboard']['enabled']:
        log_dir = os.path.join(config['logging']['tensorboard']['log_dir'], f'fold_{fold_num}')
        create_directories([log_dir])
        callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))
    
    # Train model
    history = model.fit(
        X_train, y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        callbacks=callbacks,
        verbose=config['logging']['verbose']
    )
    
    # Save history
    history_file = os.path.join(checkpoint_dir, f'fold_{fold_num}_history.npy')
    np.save(history_file, history.history)
    
    return model, history


def train_all_folds(config):
    """Train model on all LOAO folds.
    
    Args:
        config (dict): Configuration dictionary
    """
    models = {}
    histories = {}
    
    for fold_num in range(1, 5):
        model, history = train_loao_fold(fold_num, config)
        models[fold_num] = model
        histories[fold_num] = history
    
    return models, histories


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Hand Vein CapsNet')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--fold', type=int, default=None,
                       help='Specific fold to train (1-4), default: all')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Load config
    config = load_config(args.config)
    
    # Create directories
    create_directories([
        config['training']['checkpoint']['save_dir'],
        config['logging']['tensorboard']['log_dir'],
        config['inference']['predictions_dir']
    ])
    
    print("\n" + "="*60)
    print("Hand Vein Recognition using Capsule Networks")
    print("="*60)
    print(f"Config file: {args.config}")
    print(f"Data directory: {config['data']['data_dir']}")
    print(f"Number of subjects: {config['data']['num_subjects']}")
    print(f"Image size: {config['data']['img_size']}x{config['data']['img_size']}")
    print("="*60 + "\n")
    
    # Train
    if args.fold:
        model, history = train_loao_fold(args.fold, config)
    else:
        models, histories = train_all_folds(config)
        
        # Print summary
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        for fold_num, history in histories.items():
            best_acc = max(history.history['val_accuracy'])
            print(f"Fold {fold_num}: Best Val Accuracy = {best_acc:.4f}")
        print("="*60 + "\n")


if __name__ == '__main__':
    main()
