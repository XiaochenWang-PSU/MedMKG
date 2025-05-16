
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm
from tqdm import tqdm
import numpy as np
import torch
import os
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Evaluator:
    def __init__(self, model, dataset, num_entities, num_relations, all_triplets, output_folder='output', device="cpu", batch_size=128):
        self.model = model.to(device)
        self.dataset = dataset
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.all_triplets_set = set(all_triplets)  # If all_triplets is already a list of tuples
        # self.all_triplets_set = set(tuple(t.tolist()) for t in all_triplets)  # Convert to set for fast lookup
        self.output_folder = output_folder
        self.device = device
        self.batch_size = batch_size
        os.makedirs(self.output_folder, exist_ok=True)
        # self.valid_mask = self._precompute_valid_mask()
    def _precompute_valid_mask(self):
        """
        Precompute a sparse mask for valid triplets.
        """
        indices = torch.tensor(list(self.all_triplets_set), dtype=torch.long).T  # Shape: (3, num_triplets)
        values = torch.ones(indices.shape[1], dtype=torch.bool)
        mask = torch.sparse_coo_tensor(indices, values, size=(self.num_entities, self.num_relations, self.num_entities))
        return mask

#    def _generate_corrupted_triplets(self, pos_triplets, task):
#        """
#        Generate corrupted triplets in a vectorized manner, ensuring no real triplets exist in the corrupted set.
#        """
#        batch_size = pos_triplets.shape[0]
#        corrupted = []
#    
#        if task == "head":
#            candidates = torch.arange(self.num_entities, device=self.device)
#            for i in range(batch_size):
#                # Generate all corrupted triplets for the current positive triplet
#                corrupted_triplets = torch.stack(
#                    [candidates, pos_triplets[i, 1].expand(self.num_entities), pos_triplets[i, 2].expand(self.num_entities)],
#                    dim=1
#                )
#                # Filter using a set of valid triplets
#                mask = torch.tensor(
#                    [tuple(t.tolist()) not in self.all_triplets_set for t in corrupted_triplets],
#                    device=self.device
#                )
#                corrupted.append(corrupted_triplets[mask])
#        elif task == "tail":
#            candidates = torch.arange(self.num_entities, device=self.device)
#            for i in range(batch_size):
#                corrupted_triplets = torch.stack(
#                    [pos_triplets[i, 0].expand(self.num_entities), pos_triplets[i, 1].expand(self.num_entities), candidates],
#                    dim=1
#                )
#                mask = torch.tensor(
#                    [tuple(t.tolist()) not in self.all_triplets_set for t in corrupted_triplets],
#                    device=self.device
#                )
#                corrupted.append(corrupted_triplets[mask])
#        elif task == "link":
#            candidates = torch.arange(self.num_relations, device=self.device)
#            for i in range(batch_size):
#                corrupted_triplets = torch.stack(
#                    [pos_triplets[i, 0].expand(self.num_relations), candidates, pos_triplets[i, 2].expand(self.num_relations)],
#                    dim=1
#                )
#                mask = torch.tensor(
#                    [tuple(t.tolist()) not in self.all_triplets_set for t in corrupted_triplets],
#                    device=self.device
#                )
#                corrupted.append(corrupted_triplets[mask])
#        else:
#            raise ValueError("Task must be 'head', 'tail', or 'link'")
#    
#        # Concatenate all corrupted triplets into a single tensor
#        if corrupted:
#            corrupted = torch.cat(corrupted, dim=0)
#        else:
#            corrupted = torch.empty(0, 3, device=self.device)
#        return corrupted


    def evaluate(self, task="head"):
        """
        Perform evaluation on head, tail, or link prediction.

        Args:
            task: "head", "tail", or "link".

        Returns:
            A dictionary with evaluation metrics.
        """
        self.model.eval()

        ranks = []
        hits_at = {k: 0 for k in [3, 5, 10]}

        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {task} prediction"):
                pos_triplets = batch[0].to(self.device)  # Positive triplets: (batch_size, 3)
                batch_size = pos_triplets.shape[0]
                corrupted_triplets = self._generate_corrupted_triplets(pos_triplets, task)
                if corrupted_triplets.numel() == 0:
                    continue

                scores = self._compute_scores(corrupted_triplets, task)
                positive_scores = self._compute_scores_pos(pos_triplets, task)
                for i in range(batch_size):
                    # Rank of the positive triplet
                    positive_scores[i] = positive_scores[i].unsqueeze(0) if positive_scores[i].dim() == 0 else positive_scores[i]
                    # print(f"scores[i].shape: {scores[i].shape}, positive_scores[i].shape: {positive_scores[i].shape}")

                    all_scores = torch.cat((scores[i], positive_scores[i].unsqueeze(0)))
                    rank = torch.argsort(all_scores, descending=True).tolist().index(len(scores[i])) + 1
                    ranks.append(rank)

                    # Update Hits@k
                    for k in hits_at.keys():
                        hits_at[k] += (rank <= k)

        metrics = {
            "MR": np.mean(ranks),
            "MRR": np.mean(1.0 / np.array(ranks)),
            **{f"Hits@{k}": hits_at[k] / len(ranks) for k in hits_at},
        }
        self._save_results_to_csv(metrics, task)
        return metrics

    def _compute_scores_pos(self, triplets, task, chunk_size=8192):
        """
        Compute scores for triplets in chunks to avoid OOM errors.

        Args:
            triplets (torch.Tensor): Triplets to evaluate.
            task (str): The task ('head', 'tail', or 'link').
            chunk_size (int): Number of triplets to process per chunk.

        Returns:
            torch.Tensor: Scores for all triplets.
        """
        num_triplets = triplets.shape[0]
        scores = []

        # Process triplets in chunks
        for start in range(0, num_triplets, chunk_size):
            end = min(start + chunk_size, num_triplets)
            chunk = triplets[start:end].to(self.device)  # Move chunk to device

            # Compute scores for the current chunk
            chunk_scores = self.model._score_triplets(chunk)
            scores.append(chunk_scores.cpu())  # Move back to CPU to save GPU memory

        # Concatenate scores
        scores = torch.cat(scores, dim=0)
        return scores.to(self.device)

    def _save_results_to_csv(self, metrics, task):
        """
        Save evaluation metrics to a single CSV file for all tasks.
        """
        output_file = os.path.join(self.output_folder, f"{self.model.__class__.__name__}_evaluation.csv")
        file_exists = os.path.isfile(output_file)

        metrics["Task"] = task

        header = list(metrics.keys())
        row = [metrics[key] for key in header]

        with open(output_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)
        print("Saved to ", output_file)

#    def _evaluate_triplet_classification(self, threshold):
#        """
#        Evaluate the model on the triplet classification task.
#
#        Args:
#            threshold: Score threshold for classifying a triplet as true or false.
#
#        Returns:
#            A dictionary with accuracy, precision, recall, and F1-score.
#        """
#        true_labels = []
#        predicted_labels = []
#
#        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
#
#        with torch.no_grad():
#            for batch in tqdm(dataloader, desc="Evaluating triplet classification"):
#                triplets = batch[0].to(self.device)  # Triplets: (batch_size, 3)
#                labels = batch[1].to(self.device)  # True labels: (batch_size)
#
#                scores = self.model._score_triplets(triplets)
#                predictions = (scores >= threshold).float()
#
#                true_labels.extend(labels.cpu().numpy())
#                predicted_labels.extend(predictions.cpu().numpy())
#
#        # Compute metrics
#        accuracy = accuracy_score(true_labels, predicted_labels)
#        precision = precision_score(true_labels, predicted_labels)
#        recall = recall_score(true_labels, predicted_labels)
#        f1 = f1_score(true_labels, predicted_labels)
#
#        metrics = {
#            "Accuracy": accuracy,
#            "Precision": precision,
#            "Recall": recall,
#            "F1-Score": f1,
#        }
#
#        self._save_results_to_csv(metrics, "triplet_classification")
#        print("Triplet Classification Metrics:")
#        for key, value in metrics.items():
#            print(f"{key}: {value:.4f}")
#
#        return metrics
#
#    def _save_results_to_csv(self, metrics, task):
#        """
#        Save evaluation metrics to a single CSV file for all tasks.
#        """
#        output_file = os.path.join(self.output_folder, f"{self.model.__class__.__name__}_evaluation.csv")
#        file_exists = os.path.isfile(output_file)
#
#        metrics["Task"] = task
#
#        header = list(metrics.keys())
#        row = [metrics[key] for key in header]
#
#        with open(output_file, mode='a', newline='') as f:
#            writer = csv.writer(f)
#            if not file_exists:
#                writer.writerow(header)
#            writer.writerow(row)
#        print("Saved to ", output_file)
#
#
    def _generate_corrupted_triplets(self, pos_triplets, task):
        """
        Generate corrupted triplets by replacing heads, tails, or relations.
        """
        batch_size = pos_triplets.shape[0]

        if task == "head":
            corrupted = pos_triplets.unsqueeze(1).repeat(1, self.num_entities, 1)
            corrupted[:, :, 0] = torch.arange(self.num_entities, device=self.device).unsqueeze(0).repeat(batch_size, 1)
        elif task == "tail":
            corrupted = pos_triplets.unsqueeze(1).repeat(1, self.num_entities, 1)
            corrupted[:, :, 2] = torch.arange(self.num_entities, device=self.device).unsqueeze(0).repeat(batch_size, 1)
        elif task == "link":
            corrupted = pos_triplets.unsqueeze(1).repeat(1, self.num_relations, 1)
            corrupted[:, :, 1] = torch.arange(self.num_relations, device=self.device).unsqueeze(0).repeat(batch_size, 1)
        else:
            raise ValueError("Task must be 'head', 'tail', or 'link'")

        return corrupted.view(-1, 3)
#
    def _compute_scores(self, corrupted_triplets, task, chunk_size=8192):
        """
        Compute scores for corrupted triplets in chunks to avoid OOM errors.
        
        Args:
            corrupted_triplets (torch.Tensor): All corrupted triplets to evaluate.
            task (str): The task ('head', 'tail', or 'link').
            chunk_size (int): Number of triplets to process per chunk.
    
        Returns:
            torch.Tensor: Scores for all corrupted triplets reshaped to (batch_size, num_entities/relations).
        """
        num_corrupted = corrupted_triplets.shape[0]
        batch_size = num_corrupted // (self.num_entities if task in ["head", "tail"] else self.num_relations)
        scores = []
    
        # Process corrupted triplets in chunks
        for start in range(0, num_corrupted, chunk_size):
            end = min(start + chunk_size, num_corrupted)
            chunk = corrupted_triplets[start:end].to(self.device)  # Move chunk to device
    
            # Compute scores for the current chunk
            chunk_scores = self.model._score_triplets(chunk)
            scores.append(chunk_scores.cpu())  # Move back to CPU to save GPU memory
    
        # Concatenate scores and reshape
        scores = torch.cat(scores, dim=0)
        scores = scores.view(batch_size, -1)  # Reshape to (batch_size, num_entities/relations)
    
        return scores.to(self.device)

    def _get_positive_scores(self, pos_triplets, scores, task):
        """
        Extract positive scores corresponding to positive triplets.
        """
        if task == "head":
            indices = pos_triplets[:, 0]
        elif task == "tail":
            indices = pos_triplets[:, 2]
        elif task == "link":
            indices = pos_triplets[:, 1]
        else:
            raise ValueError("Invalid task.")
        
        return torch.gather(scores, 1, indices.unsqueeze(1)).squeeze(1)

    def _compute_mrr(self, ranks):
        """Compute Mean Reciprocal Rank."""
        return np.mean([1 / r for r in ranks if r > 0])

    def _update_hits_at_k(self, hits_at_k, rank, k_list):
        """Update Hits@k counters."""
        for k in k_list:
            if rank <= k:
                hits_at_k[k] += 1

    def _update_ndcg_at_k(self, ndcg_at_k, ranked_list, ground_truth, k_list):
        """Update NDCG@k scores."""
        for k in k_list:
            dcg = sum([1 / np.log2(i + 2) for i, idx in enumerate(ranked_list[:k]) if idx in ground_truth])
            idcg = sum([1 / np.log2(i + 2) for i in range(min(k, len(ground_truth)))])
            ndcg_at_k[k].append(dcg / idcg if idcg > 0 else 0.0)



    def _normalize_hits(self, hits_at_k, total):
        """Normalize Hits@k by the number of test triples."""
        return {k: hits_at_k[k] / total for k in hits_at_k}

    def _normalize_ndcg(self, ndcg_at_k):
        """Normalize NDCG@k by averaging scores."""
        return {k: np.mean(ndcg_at_k[k]) for k in ndcg_at_k}
    def _compute_ndcg(self, rank, k):
        """
        Compute NDCG@k for a single rank.
        """
        if rank <= k:
            return 1.0 / np.log2(rank + 1)
        return 0.0
        
class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        """
        Args:
            patience (int): How many epochs to wait for improvement.
            delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # No early stop
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Trigger early stop
            return False
