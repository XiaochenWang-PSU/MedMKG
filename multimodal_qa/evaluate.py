import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from typing import Dict, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import pandas as pd

class MedicalVQAEvaluator:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.model.eval()

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on test set for binary classification
        Returns:
            Dictionary containing evaluation metrics
        """
        all_logits = []
        all_labels = []
        all_predictions = []
        dataset_results = {
            'slake': {'logits': [], 'labels': []},
            'vqa_rad': {'logits': [], 'labels': []},
            'pathvqa': {'logits': [], 'labels': []}
        }
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                if batch is None:
                    continue
                
                
                batch = move_to_device(batch, self.device)
                


                
                # Get model predictions
                logits, _ = self.model(batch)
                
                # Convert answers to binary labels
                labels = batch['answer'].to(self.device)
                
                

                
                predictions = (logits > 0).float()
                
                # Store results
                all_logits.extend(logits.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                
                # Store dataset-specific results
                for idx, dataset_name in enumerate(batch['dataset']):
                    dataset_results[dataset_name]['logits'].append(logits[idx].cpu().numpy())
                    dataset_results[dataset_name]['labels'].append(labels[idx].cpu().numpy())
        
        # Calculate overall metrics
        metrics = self._calculate_metrics(
            np.array(all_labels),
            np.array(all_predictions),
            np.array(all_logits)
        )
        
        # Calculate per-dataset metrics
        for dataset_name, results in dataset_results.items():
            if results['logits']:
                dataset_metrics = self._calculate_metrics(
                    np.array(results['labels']),
                    (np.array(results['logits']) > 0).astype(float),
                    np.array(results['logits'])
                )
                metrics.update({
                    f'{dataset_name}_{k}': v 
                    for k, v in dataset_metrics.items()
                })
        
        return metrics

    def _calculate_metrics(self, labels: np.ndarray, predictions: np.ndarray, 
                         logits: np.ndarray) -> Dict[str, float]:
        """Calculate classification metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(labels, predictions)
        
        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        
        # AUC-ROC
        try:
            metrics['auc_roc'] = roc_auc_score(labels, logits)
        except:
            metrics['auc_roc'] = float('nan')
        
        return metrics

    def analyze_errors(self) -> pd.DataFrame:
        """
        Analyze model errors to identify patterns
        Returns:
            DataFrame containing error analysis information
        """
        error_data = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Analyzing errors"):
                if batch is None:
                    continue
                    
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get predictions
                logits, _ = self.model(batch)
                predictions = (logits > 0).float()
                
                # Convert answers to binary
                labels = torch.tensor([
                    1 if ans.lower().strip() in ['yes', 'true'] else 0 
                    for ans in batch['answer']
                ]).to(self.device)
                
                # Analyze each sample
                for idx, (pred, label) in enumerate(zip(predictions, labels)):
                    if pred != label:  # Only analyze errors
                        error_data.append({
                            'dataset': batch['dataset'][idx],
                            'question': batch['question'][idx],
                            'true_answer': batch['answer'][idx],
                            'predicted_answer': 'yes' if pred == 1 else 'no',
                            'confidence': torch.sigmoid(logits[idx]).item(),
                        })
        
        return pd.DataFrame(error_data)
def move_to_device(batch, device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
        elif isinstance(batch[k], dict):
            batch[k] = {sub_k: sub_v.to(device) if isinstance(sub_v, torch.Tensor) else sub_v
                        for sub_k, sub_v in batch[k].items()}
    return batch
def evaluate_model(model, test_loader, device) -> Dict[str, float]:
    """
    Convenience function for model evaluation
    Args:
        model: The medical VQA model
        test_loader: DataLoader for test set
        device: Device to run evaluation on
    Returns:
        Dictionary containing evaluation metrics
    """
    evaluator = MedicalVQAEvaluator(model, test_loader, device)
    metrics = evaluator.evaluate()
    
    # Print metrics
    print("\nEvaluation Results:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    return metrics