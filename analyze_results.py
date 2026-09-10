"""Results Analysis Script for Hand Vein CapsNet"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
import seaborn as sns


def load_fold_results(fold_num, predictions_dir='predictions'):
    """Load results for a specific fold.
    
    Args:
        fold_num (int): Fold number (1-4)
        predictions_dir (str): Directory containing predictions
        
    Returns:
        dict: Results dictionary
    """
    fold_prefix = f'fold_{fold_num}'
    
    try:
        y_true = np.load(f'{predictions_dir}/{fold_prefix}_y_true.npy')
        y_pred = np.load(f'{predictions_dir}/{fold_prefix}_y_pred.npy')
        probs = np.load(f'{predictions_dir}/{fold_prefix}_probs.npy')
        
        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'probs': probs,
            'fold': fold_num
        }
    except FileNotFoundError:
        print(f"⚠️ Results for fold {fold_num} not found")
        return None


def compute_metrics(y_true, y_pred, probs=None):
    """Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        probs: Prediction probabilities (optional)
        
    Returns:
        dict: Metrics dictionary
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Top-k accuracy
    if probs is not None:
        for k in [5, 10]:
            top_k = np.sum(np.argsort(probs, axis=1)[:, -k:] == y_true[:, np.newaxis], axis=1)
            metrics[f'top_{k}_accuracy'] = np.mean(top_k)
    
    return metrics


def analyze_all_folds(predictions_dir='predictions', num_folds=4):
    """Analyze results across all folds.
    
    Args:
        predictions_dir (str): Directory containing predictions
        num_folds (int): Number of folds
        
    Returns:
        dict: Analysis results
    """
    results = {}
    all_metrics = []
    
    print("\n" + "="*70)
    print("HAND VEIN RECOGNITION - RESULTS ANALYSIS")
    print("="*70 + "\n")
    
    for fold_num in range(1, num_folds + 1):
        print(f"Processing Fold {fold_num}...")
        fold_results = load_fold_results(fold_num, predictions_dir)
        
        if fold_results is None:
            continue
        
        y_true = fold_results['y_true']
        y_pred = fold_results['y_pred']
        probs = fold_results['probs']
        
        # Compute metrics
        metrics = compute_metrics(y_true, y_pred, probs)
        
        results[f'fold_{fold_num}'] = {
            'results': fold_results,
            'metrics': metrics
        }
        all_metrics.append(metrics)
        
        # Print fold results
        print(f"\nFold {fold_num} Results:")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Precision:         {metrics['precision']:.4f}")
        print(f"  Recall:            {metrics['recall']:.4f}")
        print(f"  F1-Score:          {metrics['f1']:.4f}")
        if 'top_5_accuracy' in metrics:
            print(f"  Top-5 Accuracy:    {metrics['top_5_accuracy']:.4f}")
        if 'top_10_accuracy' in metrics:
            print(f"  Top-10 Accuracy:   {metrics['top_10_accuracy']:.4f}")
    
    # Compute averages
    if all_metrics:
        print("\n" + "-"*70)
        print("CROSS-VALIDATION RESULTS (Average across folds):")
        print("-"*70)
        
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        
        print(f"\nAccuracy:          {avg_metrics['accuracy']['mean']:.4f} ± {avg_metrics['accuracy']['std']:.4f}")
        print(f"Precision:         {avg_metrics['precision']['mean']:.4f} ± {avg_metrics['precision']['std']:.4f}")
        print(f"Recall:            {avg_metrics['recall']['mean']:.4f} ± {avg_metrics['recall']['std']:.4f}")
        print(f"F1-Score:          {avg_metrics['f1']['mean']:.4f} ± {avg_metrics['f1']['std']:.4f}")
        
        if 'top_5_accuracy' in avg_metrics:
            print(f"Top-5 Accuracy:    {avg_metrics['top_5_accuracy']['mean']:.4f} ± {avg_metrics['top_5_accuracy']['std']:.4f}")
        if 'top_10_accuracy' in avg_metrics:
            print(f"Top-10 Accuracy:   {avg_metrics['top_10_accuracy']['mean']:.4f} ± {avg_metrics['top_10_accuracy']['std']:.4f}")
        
        results['average_metrics'] = avg_metrics
    
    return results


def plot_fold_comparison(results, output_dir='plots'):
    """Plot comparison across folds.
    
    Args:
        results (dict): Analysis results
        output_dir (str): Output directory for plots
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Extract metrics for each fold
    fold_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for fold_num in range(1, 5):
        fold_key = f'fold_{fold_num}'
        if fold_key in results:
            metrics = results[fold_key]['metrics']
            fold_names.append(f'Fold {fold_num}')
            accuracies.append(metrics['accuracy'])
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1_scores.append(metrics['f1'])
    
    if not fold_names:
        print("No fold data to plot")
        return
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Hand Vein CapsNet - Cross-Validation Results', fontsize=16, fontweight='bold')
    
    # Accuracy
    ax = axes[0, 0]
    ax.bar(fold_names, accuracies, color='skyblue', edgecolor='navy')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim([0, 1])
    ax.set_title('Accuracy per Fold', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(accuracies):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Precision
    ax = axes[0, 1]
    ax.bar(fold_names, precisions, color='lightcoral', edgecolor='darkred')
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_ylim([0, 1])
    ax.set_title('Precision per Fold', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(precisions):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Recall
    ax = axes[1, 0]
    ax.bar(fold_names, recalls, color='lightgreen', edgecolor='darkgreen')
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_ylim([0, 1])
    ax.set_title('Recall per Fold', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(recalls):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # F1-Score
    ax = axes[1, 1]
    ax.bar(fold_names, f1_scores, color='plum', edgecolor='purple')
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_ylim([0, 1])
    ax.set_title('F1-Score per Fold', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(f1_scores):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fold_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/fold_comparison.png")
    plt.close()


def plot_confusion_matrices(results, output_dir='plots', max_classes=20):
    """Plot confusion matrices for each fold (sample of classes).
    
    Args:
        results (dict): Analysis results
        output_dir (str): Output directory for plots
        max_classes (int): Max classes to show in heatmap
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    for fold_num in range(1, 5):
        fold_key = f'fold_{fold_num}'
        if fold_key not in results:
            continue
        
        y_true = results[fold_key]['results']['y_true']
        y_pred = results[fold_key]['results']['y_pred']
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # For large matrices, only show top-predicted classes
        if cm.shape[0] > max_classes:
            # Find most confused classes
            top_classes = np.argsort(np.sum(cm, axis=1))[-max_classes:]
            cm_subset = cm[np.ix_(top_classes, top_classes)]
            title_suffix = f" (Top {max_classes} Classes)"
        else:
            cm_subset = cm
            title_suffix = ""
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm_subset, cmap='Blues', cbar=True, ax=ax,
                   xticklabels=False, yticklabels=False)
        ax.set_title(f'Fold {fold_num} Confusion Matrix{title_suffix}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/fold_{fold_num}_confusion_matrix.png', 
                   dpi=200, bbox_inches='tight')
        print(f"✅ Saved: {output_dir}/fold_{fold_num}_confusion_matrix.png")
        plt.close()


def plot_accuracy_distribution(results, output_dir='plots'):
    """Plot per-class accuracy distribution.
    
    Args:
        results (dict): Analysis results
        output_dir (str): Output directory for plots
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    all_accuracies = []
    fold_labels = []
    
    for fold_num in range(1, 5):
        fold_key = f'fold_{fold_num}'
        if fold_key not in results:
            continue
        
        y_true = results[fold_key]['results']['y_true']
        y_pred = results[fold_key]['results']['y_pred']
        
        # Per-class accuracy
        class_accuracies = []
        for class_id in np.unique(y_true):
            mask = y_true == class_id
            class_acc = np.mean(y_pred[mask] == y_true[mask])
            class_accuracies.append(class_acc)
        
        all_accuracies.extend(class_accuracies)
        fold_labels.extend([f'Fold {fold_num}'] * len(class_accuracies))
    
    # Plot distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax = axes[0]
    ax.hist(all_accuracies, bins=30, color='skyblue', edgecolor='navy', alpha=0.7)
    ax.axvline(np.mean(all_accuracies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_accuracies):.3f}')
    ax.axvline(np.median(all_accuracies), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(all_accuracies):.3f}')
    ax.set_xlabel('Per-Class Accuracy', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Per-Class Accuracy', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Statistics
    ax = axes[1]
    ax.axis('off')
    stats_text = f"""
    Per-Class Accuracy Statistics
    
    Mean:                {np.mean(all_accuracies):.4f}
    Median:              {np.median(all_accuracies):.4f}
    Std Dev:             {np.std(all_accuracies):.4f}
    Min:                 {np.min(all_accuracies):.4f}
    Max:                 {np.max(all_accuracies):.4f}
    
    Classes with 0% accuracy:    {np.sum(np.array(all_accuracies) == 0)}
    Classes with 100% accuracy:  {np.sum(np.array(all_accuracies) == 1.0)}
    Classes above 80%:           {np.sum(np.array(all_accuracies) >= 0.8)}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
           family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/accuracy_distribution.png")
    plt.close()


def save_summary_report(results, output_dir='results'):
    """Save summary report as JSON.
    
    Args:
        results (dict): Analysis results
        output_dir (str): Output directory
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Prepare summary
    summary = {
        'fold_results': {},
        'average_metrics': None
    }
    
    for fold_num in range(1, 5):
        fold_key = f'fold_{fold_num}'
        if fold_key in results:
            metrics = results[fold_key]['metrics']
            summary['fold_results'][f'fold_{fold_num}'] = {
                k: float(v) for k, v in metrics.items()
            }
    
    if 'average_metrics' in results:
        summary['average_metrics'] = {
            k: {
                'mean': float(v['mean']),
                'std': float(v['std']),
                'values': [float(x) for x in v['values']]
            }
            for k, v in results['average_metrics'].items()
        }
    
    # Save JSON
    with open(f'{output_dir}/results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Saved: {output_dir}/results_summary.json")


def main():
    """Main analysis function."""
    # Analyze all folds
    results = analyze_all_folds(predictions_dir='predictions', num_folds=4)
    
    # Create plots
    print("\nGenerating visualizations...")
    plot_fold_comparison(results, output_dir='plots')
    plot_confusion_matrices(results, output_dir='plots')
    plot_accuracy_distribution(results, output_dir='plots')
    
    # Save summary
    save_summary_report(results, output_dir='results')
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nOutput files:")
    print("  - plots/fold_comparison.png")
    print("  - plots/fold_*_confusion_matrix.png")
    print("  - plots/accuracy_distribution.png")
    print("  - results/results_summary.json")


if __name__ == '__main__':
    main()
