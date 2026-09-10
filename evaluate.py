"""Evaluation script for Hand Vein CapsNet"""

import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import HandVeinDataLoader
from src.utils import load_config, print_metrics_report, plot_confusion_matrix


def evaluate_fold(model, fold_num, config):
    """Evaluate model on single fold.
    
    Args:
        model: Trained model
        fold_num (int): Fold number
        config (dict): Configuration
        
    Returns:
        dict: Evaluation metrics
    """
    # Load data
    data_loader = HandVeinDataLoader(
        data_dir=config['data']['data_dir'],
        img_size=config['data']['img_size'],
        num_subjects=config['data']['num_subjects']
    )
    
    _, _, X_val, y_val = data_loader.load_loao_fold(fold_num)
    
    # Predict
    y_pred_probs = model.predict(X_val, batch_size=config['inference']['batch_size'])
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
    accuracy = accuracy_score(y_val, y_pred)
    
    return {
        'fold': fold_num,
        'accuracy': accuracy,
        'y_true': y_val,
        'y_pred': y_pred,
        'y_pred_probs': y_pred_probs
    }


def evaluate_all_folds(models, config):
    """Evaluate models on all folds.
    
    Args:
        models (dict): Dictionary of trained models
        config (dict): Configuration
        
    Returns:
        list: List of evaluation results
    """
    results = []
    
    for fold_num in range(1, 5):
        print(f"\nEvaluating Fold {fold_num}...")
        result = evaluate_fold(models[fold_num], fold_num, config)
        results.append(result)
        print(f"Fold {fold_num} Accuracy: {result['accuracy']:.4f}")
    
    # Overall accuracy
    all_y_true = np.concatenate([r['y_true'] for r in results])
    all_y_pred = np.concatenate([r['y_pred'] for r in results])
    overall_acc = accuracy_score(all_y_true, all_y_pred)
    
    print(f"\n{'='*60}")
    print(f"Overall LOAO Accuracy: {overall_acc:.4f}")
    print(f"{'='*60}")
    
    return results, overall_acc


def save_results(results, config):
    """Save evaluation results.
    
    Args:
        results (list): Evaluation results
        config (dict): Configuration
    """
    results_dir = config['inference']['predictions_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # Save predictions
    for result in results:
        fold_num = result['fold']
        
        # Save arrays
        np.save(os.path.join(results_dir, f'fold_{fold_num}_y_true.npy'), result['y_true'])
        np.save(os.path.join(results_dir, f'fold_{fold_num}_y_pred.npy'), result['y_pred'])
        np.save(os.path.join(results_dir, f'fold_{fold_num}_probs.npy'), result['y_pred_probs'])
        
        # Save confusion matrix
        cm = confusion_matrix(result['y_true'], result['y_pred'])
        np.save(os.path.join(results_dir, f'fold_{fold_num}_cm.npy'), cm)
    
    print(f"\nResults saved to {results_dir}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Hand Vein CapsNet')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--model_dir', type=str, default='checkpoints/',
                       help='Directory with model checkpoints')
    parser.add_argument('--fold', type=int, default=None,
                       help='Specific fold to evaluate (1-4)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    print("\n" + "="*60)
    print("Hand Vein CapsNet - Evaluation")
    print("="*60 + "\n")
    
    # Load models
    models = {}
    if args.fold:
        model_path = os.path.join(args.model_dir, f'fold_{args.fold}_best.h5')
        models[args.fold] = tf.keras.models.load_model(model_path)
        results, overall_acc = evaluate_fold(models[args.fold], args.fold, config)
    else:
        for fold_num in range(1, 5):
            model_path = os.path.join(args.model_dir, f'fold_{fold_num}_best.h5')
            if os.path.exists(model_path):
                models[fold_num] = tf.keras.models.load_model(model_path)
        
        results, overall_acc = evaluate_all_folds(models, config)
    
    # Save results
    if config['inference']['save_predictions']:
        save_results(results if isinstance(results, list) else [results], config)


if __name__ == '__main__':
    main()
