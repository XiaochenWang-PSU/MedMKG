import torch
from torch.utils.data import Dataset
import pandas as pd
import random
import hashlib

class KnowledgeGraphDataset(Dataset):
    def __init__(self, csv_path, num_entities, num_relations, neg_ratio=1):
        """
        Args:
            csv_path (str): Path to the CSV file containing the triplets.
            num_entities (int): Maximum number of entities (used for hash-based indexing).
            num_relations (int): Maximum number of relations (used for hash-based indexing).
            neg_ratio (int): Number of negative samples per positive sample.
        """
        self.csv_path = csv_path  # Store CSV path for later access
        # self.triplets = pd.concat([pd.read_csv(csv_path)[:100], pd.read_csv(csv_path)[-100:]]) 
        self.triplets = pd.read_csv(csv_path)
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.neg_ratio = neg_ratio
        self.triplets = self.triplets.apply(self._convert_to_ids, axis=1).values

    def _hash_id(self, input_str, mod_val):
        """
        Hash the input string to produce a numerical ID.
        """
        return int(hashlib.md5(input_str.encode()).hexdigest(), 16) % mod_val

    def _convert_to_ids(self, row):
        """
        Converts string triplets (Head, Relation, Tail) to numerical IDs using hashing.
        """
        head_id = self._hash_id(row['Head'], self.num_entities)
        relation_id = self._hash_id(row['Relation'], self.num_relations)
        tail_id = self._hash_id(row['Tail'], self.num_entities)
        return [head_id, relation_id, tail_id]

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        """
        Returns:
            pos_triplet: Positive triplet (head, relation, tail)
            neg_triplets: Flattened list of corrupted negative triplets
        """
        pos_triplet = self.triplets[idx]
        neg_triplets = self._generate_negative_samples(pos_triplet)
        
        # Flatten neg_triplets to shape [neg_ratio, 3]
        neg_triplets = torch.tensor(neg_triplets, dtype=torch.long).view(-1, 3)
    
        return (
            torch.tensor(pos_triplet, dtype=torch.long),
            neg_triplets
        )

    def _generate_negative_samples(self, triplet):
        """
        Generate corrupted triplets by replacing either the head or the tail entity.
        """
        neg_samples = []
        for _ in range(self.neg_ratio):
            corrupted_triplet = list(triplet)
            if random.random() < 0.5:  # Corrupt head
                corrupted_triplet[0] = random.randint(0, self.num_entities - 1)
            else:  # Corrupt tail
                corrupted_triplet[2] = random.randint(0, self.num_entities - 1)
            neg_samples.append(corrupted_triplet)
        return neg_samples
        
def build_graph_from_triplets(all_triplets):
    """
    all_triplets: np.ndarray of shape (num_samples, 3) -> [head, relation, tail].
    Returns:
        edge_index: torch.LongTensor of shape (2, num_edges)
        edge_type: torch.LongTensor of shape (num_edges,)
    """
    # Separate head, relation, tail columns
    src = all_triplets[:, 0]
    rel = all_triplets[:, 1]
    dst = all_triplets[:, 2]

    # edge_index: shape (2, num_edges)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    # edge_type: shape (num_edges,)
    edge_type = torch.tensor(rel, dtype=torch.long)
    return edge_index, edge_type