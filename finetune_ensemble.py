"""
Finetune ensemble weights and bias using annotated edge cases.
Loads ensemble_config.yaml, optimizes weights, and saves back to config.
"""
import numpy as np
import yaml
import os
from pathlib import Path
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import sys

# Add src to path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.model.text_models import HeuristicLexiconModel, ToxicityModel, ZeroShotExtremismNLI


# Label mapping
LABELS = ["non_extremist", "potentially_extremist", "extremist"]
LABEL_MAP = {"safe": 0, "uncertain": 1, "extremist": 2}


def load_ensemble_config(config_path):
    """Load ensemble configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_ensemble_config(config_path, config, weights, bias, metrics, model_names, weight_decay=None):
    """Save updated ensemble config with optimized weights and bias."""
    # Update ensemble section with optimized parameters
    ensemble_config = {
        'weights': {
            name: float(weight) for name, weight in zip(model_names, weights)
        },
        'bias': [float(b) for b in bias.flatten()],
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'f1_macro': float(metrics['f1_macro']),
            'f1_weighted': float(metrics['f1_weighted'])
        }
    }
    
    # Add weight decay parameter if provided
    if weight_decay is not None:
        ensemble_config['optimization'] = {
            'weight_decay': float(weight_decay),
            'description': 'Regularization strength towards equal weights (1/n) and zero bias'
        }
    
    config['ensemble'] = ensemble_config
    
    # Write back to file, preserving models config
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_annotations(annotations_path):
    """Load annotations from YAML file."""
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = yaml.safe_load(f)
    return annotations


def load_edge_cases(edge_cases_dir):
    """
    Load text from edge case files, split into lines.
    
    Returns:
        dict: {filename: [line1, line2, ...]}
    """
    texts = {}
    for i in range(1, len(os.listdir(edge_cases_dir)) + 1):  # text1.txt to text10.txt
        file_path = os.path.join(edge_cases_dir, f'text{i}.txt')
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                # Split into lines and filter out empty lines
                lines = [line.strip() for line in f.readlines() if line.strip()]
                texts[f'text{i}.txt'] = lines
    return texts


def instantiate_models_from_config(config):
    """
    Instantiate model objects from config file.
    
    Returns:
        tuple: (list of model instances, list of model names)
    """
    models = []
    model_names = []
    
    # Map class names to actual classes
    class_map = {
        'ToxicityModel': ToxicityModel,
        'ZeroShotExtremismNLI': ZeroShotExtremismNLI,
        'HeuristicLexiconModel': HeuristicLexiconModel,
    }
    
    for model_key, model_config in config.get('models', {}).items():
        class_name = model_config.get('class')
        args = model_config.get('args', {})
        
        if class_name in class_map:
            print(f"   - {model_key} ({class_name})")
            model_class = class_map[class_name]
            model_instance = model_class(**args)
            models.append(model_instance)
            model_names.append(model_key)
        else:
            print(f"   ⚠️ Skipping {model_key}: class {class_name} not available")
    
    return models, model_names


def collect_model_predictions(models, all_lines, line_to_file_map):
    """
    Collect predictions from all models for each line.
    
    Args:
        models: List of model instances
        all_lines: List of all text lines
        line_to_file_map: List mapping line index to (filename, line_num_in_file)
    
    Returns:
        np.array of shape (n_models, n_lines, n_labels)
    """
    n_models = len(models)
    n_lines = len(all_lines)
    n_labels = len(LABELS)
    
    predictions = np.zeros((n_models, n_lines, n_labels))
    
    for i, model in enumerate(models):
        print(f"   Getting predictions from {model.__class__.__name__}...")
        preds = model.predict(all_lines)
        predictions[i] = preds
    
    return predictions


def ensemble_with_params(model_predictions, weights, bias):
    """
    Apply ensemble with given weights and bias.
    
    Args:
        model_predictions: shape (n_models, n_texts, n_labels)
        weights: shape (n_models,)
        bias: shape (n_labels,) or scalar
    
    Returns:
        ensembled predictions of shape (n_texts, n_labels)
    """
    # Weighted average across models
    ensembled = np.average(model_predictions, axis=0, weights=weights)
    
    # Add bias
    ensembled += bias
    
    # Renormalize to probabilities
    ensembled = np.maximum(ensembled, 0)  # Clip negative values
    row_sums = ensembled.sum(axis=1, keepdims=True)
    ensembled = ensembled / (row_sums + 1e-9)
    
    return ensembled


def compute_loss(params, model_predictions, true_labels, weight_decay=0.1):
    """
    Compute loss for optimization.
    
    Args:
        params: flattened array of [weights..., bias...]
        model_predictions: shape (n_models, n_texts, n_labels)
        true_labels: shape (n_texts,) - integer labels
        weight_decay: strength of regularization towards equal weights (default: 0.1)
    
    Returns:
        loss value (cross-entropy)
    """
    n_models = model_predictions.shape[0]
    n_labels = model_predictions.shape[2]
    
    # Extract weights and bias from params
    weights = params[:n_models]
    bias = params[n_models:].reshape(1, n_labels)
    
    # Ensure weights are positive
    weights = np.abs(weights)
    
    # Get ensemble predictions
    ensemble_probs = ensemble_with_params(model_predictions, weights, bias)
    
    # Compute cross-entropy loss
    n_texts = len(true_labels)
    loss = 0.0
    for i in range(n_texts):
        prob = ensemble_probs[i, true_labels[i]]
        prob = np.clip(prob, 1e-9, 1.0)  # Avoid log(0)
        loss -= np.log(prob)
    
    loss /= n_texts
    
    # Add weight decay: penalize deviation from equal weights (1/n) and zero bias
    target_weight = 1.0 / n_models
    weight_penalty = weight_decay * np.sum((weights - target_weight) ** 2)
    bias_penalty = weight_decay * np.sum(bias ** 2)
    
    loss += weight_penalty + bias_penalty
    
    return loss


def evaluate_ensemble(ensemble_probs, true_labels):
    """Evaluate ensemble predictions."""
    predicted_labels = np.argmax(ensemble_probs, axis=1)
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1_macro = f1_score(true_labels, predicted_labels, average='macro')
    f1_weighted = f1_score(true_labels, predicted_labels, average='weighted')
    
    cm = confusion_matrix(true_labels, predicted_labels)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'predicted_labels': predicted_labels
    }


def main():
    """
    Main finetuning routine with weight decay regularization.
    
    Weight decay nudges optimization towards:
    - Equal weights: 1/n for each model (where n = number of models)
    - Zero bias: [0, 0, 0]
    
    Higher weight_decay = stronger preference for baseline (equal ensemble)
    Lower weight_decay = more freedom to deviate based on data
    
    Adjust the weight_decay parameter in step 7 to tune this tradeoff.
    """
    print("=" * 60)
    print("ENSEMBLE WEIGHT AND BIAS FINETUNING")
    print("=" * 60)
    
    # Paths
    config_path = os.path.join(PROJECT_ROOT, 'ensemble_config.yaml')
    annotations_path = os.path.join(PROJECT_ROOT, 'definition', 'annotations.yaml')
    edge_cases_dir = os.path.join(PROJECT_ROOT, 'definition', 'edge_cases')
    
    # Load ensemble configuration
    print("\n1. Loading ensemble configuration...")
    config = load_ensemble_config(config_path)
    print(f"   Loaded config from: {config_path}")
    
    # Load data
    print("\n2. Loading annotations and edge cases...")
    annotations = load_annotations(annotations_path)
    texts = load_edge_cases(edge_cases_dir)
    
    # Flatten all lines from all files and prepare ground truth
    print("\n3. Preparing line-level data...")
    all_lines = []
    true_labels = []
    line_to_file_map = []  # Track which file each line came from
    
    file_names = sorted(texts.keys())
    for fname in file_names:
        lines = texts[fname]
        labels = annotations[fname]
        
        if len(lines) != len(labels):
            print(f"   ⚠️ Warning: {fname} has {len(lines)} lines but {len(labels)} labels")
            # Truncate to shorter length
            min_len = min(len(lines), len(labels))
            lines = lines[:min_len]
            labels = labels[:min_len]
        
        for line_idx, (line, label) in enumerate(zip(lines, labels)):
            all_lines.append(line)
            true_labels.append(LABEL_MAP[label])
            line_to_file_map.append((fname, line_idx))
    
    true_labels = np.array(true_labels)
    
    print(f"   Loaded {len(file_names)} files with {len(all_lines)} total lines")
    print(f"   Label distribution: safe={np.sum(true_labels==0)}, uncertain={np.sum(true_labels==1)}, extremist={np.sum(true_labels==2)}")
    
    # Initialize models from config
    print("\n4. Instantiating models from config...")
    models, model_names = instantiate_models_from_config(config)
    print(f"   Initialized {len(models)} models")
    
    # Collect predictions
    print("\n5. Collecting model predictions...")
    model_predictions = collect_model_predictions(models, all_lines, line_to_file_map)
    print(f"   Prediction shape: {model_predictions.shape}")
    
    # Evaluate baseline (equal weights, no bias)
    print("\n6. Evaluating baseline ensemble (equal weights, no bias)...")
    baseline_weights = np.ones(len(models))
    baseline_bias = np.zeros(len(LABELS))
    baseline_ensemble = ensemble_with_params(model_predictions, baseline_weights, baseline_bias)
    baseline_metrics = evaluate_ensemble(baseline_ensemble, true_labels)
    
    print(f"   Baseline Accuracy: {baseline_metrics['accuracy']:.3f}")
    print(f"   Baseline F1 (macro): {baseline_metrics['f1_macro']:.3f}")
    print(f"   Baseline F1 (weighted): {baseline_metrics['f1_weighted']:.3f}")
    print(f"\n   Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Safe  Unc  Extr")
    cm = baseline_metrics['confusion_matrix']
    print(f"   Actual Safe  {cm[0,0]:3d}  {cm[0,1]:3d}  {cm[0,2]:3d}")
    print(f"          Unc   {cm[1,0]:3d}  {cm[1,1]:3d}  {cm[1,2]:3d}")
    print(f"          Extr  {cm[2,0]:3d}  {cm[2,1]:3d}  {cm[2,2]:3d}")
    
    # Optimize weights and bias
    print("\n7. Optimizing ensemble weights and bias...")
    
    # Hyperparameters
    weight_decay = 0.2  # Strength of regularization towards equal weights and zero bias
    print(f"   Weight decay: {weight_decay} (nudges towards equal weights 1/{len(models)} and zero bias)")
    
    # Initial parameters: equal weights + zero bias
    n_models = len(models)
    n_labels = len(LABELS)
    initial_params = np.concatenate([
        np.ones(n_models) / n_models,  # Start at target equal weights
        np.zeros(n_labels)  # Start at target zero bias
    ])
    
    # Optimization
    result = minimize(
        compute_loss,
        initial_params,
        args=(model_predictions, true_labels, weight_decay),
        method='L-BFGS-B',
        options={'maxiter': 1000, 'disp': True}
    )
    
    # Extract optimized parameters
    optimal_params = result.x
    optimal_weights = np.abs(optimal_params[:n_models])
    optimal_bias = optimal_params[n_models:].reshape(1, n_labels)
    
    print(f"\n   Optimization completed!")
    print(f"\n   Optimal weights (target: {1.0/n_models:.4f} each):")
    for name, weight in zip(model_names, optimal_weights):
        deviation = weight - (1.0/n_models)
        sign = "+" if deviation >= 0 else ""
        print(f"     {name}: {weight:.4f} ({sign}{deviation:.4f})")
    
    print(f"\n   Optimal bias (target: [0, 0, 0]):")
    print(f"     {optimal_bias.flatten()}")
    
    # Show total deviation from baseline
    weight_deviation = np.sqrt(np.mean((optimal_weights - 1.0/n_models)**2))
    bias_deviation = np.sqrt(np.mean(optimal_bias**2))
    print(f"\n   RMS deviation from baseline:")
    print(f"     Weights: {weight_deviation:.4f}")
    print(f"     Bias: {bias_deviation:.4f}")
    
    # Evaluate optimized ensemble
    print("\n8. Evaluating optimized ensemble...")
    optimized_ensemble = ensemble_with_params(model_predictions, optimal_weights, optimal_bias)
    optimized_metrics = evaluate_ensemble(optimized_ensemble, true_labels)
    
    print(f"   Optimized Accuracy: {optimized_metrics['accuracy']:.3f}")
    print(f"   Optimized F1 (macro): {optimized_metrics['f1_macro']:.3f}")
    print(f"   Optimized F1 (weighted): {optimized_metrics['f1_weighted']:.3f}")
    print(f"\n   Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Safe  Unc  Extr")
    cm = optimized_metrics['confusion_matrix']
    print(f"   Actual Safe  {cm[0,0]:3d}  {cm[0,1]:3d}  {cm[0,2]:3d}")
    print(f"          Unc   {cm[1,0]:3d}  {cm[1,1]:3d}  {cm[1,2]:3d}")
    print(f"          Extr  {cm[2,0]:3d}  {cm[2,1]:3d}  {cm[2,2]:3d}")
    
    # Show improvement
    print("\n9. Improvement Summary:")
    print(f"   Accuracy:     {baseline_metrics['accuracy']:.3f} → {optimized_metrics['accuracy']:.3f} "
          f"({optimized_metrics['accuracy'] - baseline_metrics['accuracy']:+.3f})")
    print(f"   F1 (macro):   {baseline_metrics['f1_macro']:.3f} → {optimized_metrics['f1_macro']:.3f} "
          f"({optimized_metrics['f1_macro'] - baseline_metrics['f1_macro']:+.3f})")
    print(f"   F1 (weighted): {baseline_metrics['f1_weighted']:.3f} → {optimized_metrics['f1_weighted']:.3f} "
          f"({optimized_metrics['f1_weighted'] - baseline_metrics['f1_weighted']:+.3f})")
    
    # Show per-file statistics
    print("\n10. Per-file accuracy:")
    print(f"{'File':<15} {'Lines':<7} {'Baseline':<12} {'Optimized':<12}")
    print("-" * 50)
    
    for fname in file_names:
        # Get indices for this file
        file_indices = [i for i, (f, _) in enumerate(line_to_file_map) if f == fname]
        
        if file_indices:
            file_true = true_labels[file_indices]
            file_baseline_pred = baseline_metrics['predicted_labels'][file_indices]
            file_optimized_pred = optimized_metrics['predicted_labels'][file_indices]
            
            baseline_acc = np.mean(file_true == file_baseline_pred)
            optimized_acc = np.mean(file_true == file_optimized_pred)
            
            print(f"{fname:<15} {len(file_indices):<7} {baseline_acc:>6.1%}       {optimized_acc:>6.1%}")
    
    # Show sample predictions (first few from each file)
    print("\n11. Sample line predictions (first 2 lines per file):")
    print(f"{'File':<12} {'Line':<4} {'True':<10} {'Baseline':<13} {'Optimized':<13} {'Text Preview':<40}")
    print("-" * 100)
    
    for fname in file_names:
        file_indices = [i for i, (f, _) in enumerate(line_to_file_map) if f == fname]
        # Show first 2 lines of each file
        for idx in file_indices[:2]:
            true_label = LABELS[true_labels[idx]]
            baseline_pred = LABELS[baseline_metrics['predicted_labels'][idx]]
            optimized_pred = LABELS[optimized_metrics['predicted_labels'][idx]]
            
            baseline_mark = "✓" if baseline_pred == true_label else "✗"
            optimized_mark = "✓" if optimized_pred == true_label else "✗"
            
            line_num = line_to_file_map[idx][1] + 1
            text_preview = all_lines[idx][:37] + "..." if len(all_lines[idx]) > 40 else all_lines[idx]
            
            print(f"{fname:<12} {line_num:<4} {true_label:<10} {baseline_pred:<10} {baseline_mark}  {optimized_pred:<10} {optimized_mark}  {text_preview}")
    
    # Save optimized parameters back to ensemble_config.yaml
    print("\n12. Saving optimized parameters to ensemble_config.yaml...")
    save_ensemble_config(
        config_path, 
        config, 
        optimal_weights, 
        optimal_bias, 
        optimized_metrics,
        model_names,
        weight_decay
    )
    print(f"   Updated: {config_path}")
    print(f"   ✓ Weights and bias saved under 'ensemble' key")
    print(f"   ✓ Models configuration preserved")
    print(f"   ✓ Optimization hyperparameters saved")
    print(f"   ✓ Trained on {len(all_lines)} lines from {len(file_names)} files")
    
    print("\n" + "=" * 60)
    print("FINETUNING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
