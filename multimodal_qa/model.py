import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from torch_geometric.nn import RGCNConv, RGATConv, GCNConv
from tqdm import tqdm
from PIL import Image
import random
from torch.nn import MultiheadAttention
from transformers.tokenization_utils_base import BatchEncoding
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import List, Dict, Tuple
import os



@dataclass
class KGTriplet:
    head: str  # Head concept
    head_name: str  # Head name/description
    relation: str  # type of relation
    tail: str  # Tail concept
    tail_name: str  # Tail name/description

class KGRetriever:
    def __init__(self, 
                 kg_path: str,
                 clip_model=None,
                 processor=None,
                 tokenizer = None, 
                 biomedclip: bool = False,
                 target_edges: int = 30000,  # Target number of edges
                 min_freq: int = 2):  # Minimum frequency for entities to keep
        self.clip_model = clip_model
        self.processor = processor
        self.tokenizer = tokenizer
        self.biomedclip = biomedclip
        self.target_edges = target_edges
        self.min_freq = min_freq
        
        # Load KG and create mappings
        self.triplets = self._load_kg(kg_path)
        self._create_mappings()
        
        # Load image mapping
        image_map_path = YOUR_PATH_OF_IMAGE_MAPPING_FILE
        df_image_map = pd.read_csv(image_map_path)
        self.image_id_to_path = dict(zip(df_image_map['IID'], df_image_map['Image_Path']))
    def build_edge_index(self):
        edge_index = []
        for src_id, relations in self.head_to_tails.items():
            src_idx = self.id_to_idx[src_id]
            for tails in relations.values():
                for tail in tails:
                    if tail in self.id_to_idx:
                        tgt_idx = self.id_to_idx[tail]
                        edge_index.append([src_idx, tgt_idx])
                        edge_index.append([tgt_idx, src_idx])
        return edge_index

    def build_alignment_triplet_pool(self, num_triplets=10000, per_image_samples=3):
        image_entities = [eid for eid in self.id_to_name if eid.startswith('I')]
        concept_entities = [eid for eid in self.id_to_name if not eid.startswith('I')]
    
        triplet_pool = []
        for img_id in image_entities:
            pos_ids = list(self.get_related_concepts(img_id, use_names=False))
            pos_ids = [pid for pid in pos_ids if not pid.startswith('I')]
            if not pos_ids:
                continue
    
            neg_candidates = list(set(concept_entities) - set(pos_ids))
            if not neg_candidates:
                continue
    
            for _ in range(per_image_samples):
                pos = random.choice(pos_ids)
                neg = random.choice(neg_candidates)
                triplet_pool.append((img_id, pos, neg))
    
            if len(triplet_pool) >= num_triplets:
                break
    
        self.triplet_pool = triplet_pool
        print(f"Triplet pool initialized with {len(triplet_pool)} triplets.")
    def _sample_kg(self, df: pd.DataFrame) -> pd.DataFrame:
        """Undersample knowledge graph to exactly target_edges"""
        # First, get most frequent heads to limit total nodes
        target_nodes = self.target_edges // 3  # Rough estimate to achieve target edges
        head_counts = df['Head'].value_counts()
        frequent_heads = head_counts.head(target_nodes).index.tolist()
        
        # Filter by frequent heads first
        df_filtered = df[df['Head'].isin(frequent_heads)]
        
        # If still too many edges, do random sampling while preserving relations
        if len(df_filtered) > self.target_edges:
            rel_groups = []
            for rel_type, group in df_filtered.groupby('Relation'):
                # Calculate number of edges to sample for this relation
                rel_fraction = len(group) / len(df_filtered)
                n_samples = int(self.target_edges * rel_fraction)
                sampled_group = group.sample(n=min(len(group), n_samples), random_state=42)
                rel_groups.append(sampled_group)
            
            df_filtered = pd.concat(rel_groups)
        
        # Final random sampling if still above target
        if len(df_filtered) > self.target_edges:
            df_filtered = df_filtered.sample(n=self.target_edges, random_state=42)
        
        print(f"Sampled KG stats:")
        print(f"Number of edges: {len(df_filtered)}")
        print(f"Number of unique heads: {df_filtered['Head'].nunique()}")
        print(f"Number of unique tails: {df_filtered['Tail'].nunique()}")
        print(f"Number of relations: {df_filtered['Relation'].nunique()}")
        
        return df_filtered
    def build_edge_types(self):
        edge_types = []
        for src, relations in self.head_to_tails.items():
            src_idx = self.id_to_idx[src]
            for rel, targets in relations.items():
                rel_idx = self.relation_to_idx[rel]
                for tgt in targets:
                    if tgt in self.id_to_idx:
                        edge_types.append(rel_idx)
                        edge_types.append(rel_idx)  # because you add both directions in build_edge_index
        return edge_types

    def _load_kg(self, kg_path: str) -> List[KGTriplet]:
        """Load knowledge graph from CSV file"""
        if not Path(kg_path).exists():
            raise FileNotFoundError(f"Knowledge graph file not found: {kg_path}")
            
        # Load full dataframe
        df = pd.read_csv(kg_path)
        
        # Filter for concept nodes if needed
        # df = df[df['Head'].str.startswith('C')]
        
        # Sample the knowledge graph
        df_sampled = df
        # df_sampled = self._sample_kg(df)
        # Convert to triplets
        return [
            KGTriplet(
                head=str(row['Head']),
                head_name=str(row['Head_Name']),
                relation=str(row['Relation']),
                tail=str(row['Tail']),
                tail_name=str(row['Tail_Name'])
            )
            for _, row in df_sampled.iterrows()
        ]
    def get_related_images(self, concept_ids: List[str], num_samples: int = 1) -> List[str]:
        """Get related image paths for given concepts"""
        image_candidates = set()
        # print('concept_ids received')
        for concept_id in concept_ids:
        #     print(concept_id)
            if concept_id in self.id_to_name:
               
                # Get related image IDs
                related = self.get_related_concepts(concept_id, use_names=False)
                # Filter for image IDs (those that have image paths)
                related_images = {rid for rid in related if rid in self.image_id_to_path}
                image_candidates.update(related_images)
        
        # Sample images if needed
        if len(image_candidates) > num_samples:
            image_candidates = random.sample(list(image_candidates), num_samples)
        
        # Get image paths
        return [self.image_id_to_path[img_id] for img_id in image_candidates]
    def _create_mappings(self):
        """Create mapping dictionaries for efficient retrieval"""
        self.id_to_name = {}
        self.name_to_ids = {}
        self.tail_to_heads = {}
        self.head_to_tails = {}
        self.relation_types = set()
        
        # Create ID index mapping
        all_ids = set()
        for triplet in self.triplets:
            all_ids.add(triplet.head)
            all_ids.add(triplet.tail)
        self.id_to_idx = {id_: idx for idx, id_ in enumerate(sorted(all_ids))}
        self.idx_to_id = {idx: id_ for id_, idx in self.id_to_idx.items()}
        
        for triplet in self.triplets:
            # ID-name mappings
            self.id_to_name[triplet.head] = triplet.head_name
            self.id_to_name[triplet.tail] = triplet.tail_name
            
            # Name-ID mappings
            for name, id_ in [(triplet.head_name, triplet.head), 
                            (triplet.tail_name, triplet.tail)]:
                if name not in self.name_to_ids:
                    self.name_to_ids[name] = set()
                self.name_to_ids[name].add(id_)
            
            # Relation mappings
            for source, target, mapping in [
                (triplet.tail, triplet.head, self.tail_to_heads),
                (triplet.head, triplet.tail, self.head_to_tails)
            ]:
                if source not in mapping:
                    mapping[source] = {}
                if triplet.relation not in mapping[source]:
                    mapping[source][triplet.relation] = set()
                mapping[source][triplet.relation].add(target)
            
            # Track relation types
            self.relation_types.add(triplet.relation)
            
        # Convert relation types to indexed dictionary
        self.relation_to_idx = {rel: idx for idx, rel in enumerate(sorted(self.relation_types))}
        
    def get_concepts_from_text(self, text: str) -> Set[str]:
        """Extract concepts from text"""
        matched_concepts = set()
        text_lower = text.lower()
        
        for concept_name in self.name_to_ids:
            if concept_name.lower() in text_lower:
                matched_concepts.update(self.name_to_ids[concept_name])
        # print(matched_concepts)
        return matched_concepts
        
    def get_related_concepts(self, concept: str, relation_type: str = None, use_names: bool = True) -> Set[str]:
        """Get related concepts for a given concept"""
        related = set()
        concept_ids = {concept} if not use_names else self.name_to_ids.get(concept, set())
        
        for concept_id in concept_ids:
            # Get concepts from head_to_tails
            if concept_id in self.head_to_tails:
                for rel_type, tails in self.head_to_tails[concept_id].items():
                    if relation_type is None or rel_type == relation_type:
                        if use_names:
                            related.update(self.id_to_name[t] for t in tails)
                        else:
                            related.update(tails)
                            
            # Get concepts from tail_to_heads
            if concept_id in self.tail_to_heads:
                for rel_type, heads in self.tail_to_heads[concept_id].items():
                    if relation_type is None or rel_type == relation_type:
                        if use_names:
                            related.update(self.id_to_name[h] for h in heads)
                        else:
                            related.update(heads)
        
        return related
    def get_entity_embedding_by_id(self, entity_id: str, device) -> Optional[torch.Tensor]:
        """Get CLIP embedding for an entity using its ID (text or image)"""
        if entity_id.startswith('I') and entity_id in self.image_id_to_path:
            image_path = self.image_id_to_path[entity_id]
            try:
                image = Image.open(image_path).convert('RGB')
                if self.biomedclip:
                    rel_img_features = self.processor(
                        images=image,
                        return_tensors="pt"
                    )['pixel_values'].to(device)
                else:
                    rel_img_features = self.processor(image).unsqueeze(0).to(device)

                return self.clip_model.encode_image(rel_img_features).unsqueeze(0)
            except Exception as e:
                print(f"Failed to load image {image_path} for entity {entity_id}: {e}")
                return None
        elif entity_id in self.id_to_name:
            concept_name = self.id_to_name[entity_id]
            return self.get_concept_embedding(concept_name, device)
        else:
            return None
    def get_concept_embedding(self, concept: str, device) -> Optional[torch.Tensor]:
        """Get CLIP text embedding for a concept"""
        if self.biomedclip:
            tokens = self.processor(
                text=concept,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(device)
        else:
            tokens = self.tokenizer(concept).to(device)
 
        # with torch.no_grad():
        return self.clip_model.encode_text(tokens)
    def build_graph_data(self, concepts: List[str]) -> Tuple[List[List[int]], List[int]]:
        """Build graph structure for a set of concept names"""
        edge_index = []  # (source_idx, target_idx) pairs
        edge_types = []  # Relation type indices
        
        # Get all concept IDs for the input concepts
        concept_ids = []
        for concept in concepts:
            if concept in self.name_to_ids:
                concept_ids.extend(list(self.name_to_ids[concept]))
                
        # Create ID to index mapping
        id_to_idx = {cid: i for i, cid in enumerate(concept_ids)}
        # Add edges
        for src_id in concept_ids:
            src_idx = id_to_idx[src_id]
            
            # Add head->tail edges
            if src_id in self.head_to_tails:
                for relation, tails in self.head_to_tails[src_id].items():
                    rel_idx = self.relation_to_idx[relation]
                    for tail in tails:
                        if tail in id_to_idx:
                            edge_index.append([src_idx, id_to_idx[tail]])
                            edge_types.append(rel_idx)
            
            # Add tail->head edges
            if src_id in self.tail_to_heads:
                for relation, heads in self.tail_to_heads[src_id].items():
                    rel_idx = self.relation_to_idx[relation]
                    for head in heads:
                        if head in id_to_idx:
                            edge_index.append([src_idx, id_to_idx[head]])
                            edge_types.append(rel_idx)
                            
        return edge_index, edge_types
        
    def get_num_relations(self) -> int:
        """Get total number of relation types"""
        return len(self.relation_types)
class MedicalVQAModel(nn.Module):
    def __init__(self, clip_model, projection_dim: int = 512):
        super().__init__()
        self.clip_model = clip_model
        
        # Get CLIP embedding dimension
        clip_dim = getattr(clip_model, 'embed_dim', 
                          getattr(clip_model, 'config.hidden_size', 512))
        
        # Projection layers
        self.image_projection = (nn.Linear(clip_dim, projection_dim) 
                               if clip_dim != projection_dim else nn.Identity())
        self.text_projection = (nn.Linear(clip_dim, projection_dim) 
                              if clip_dim != projection_dim else nn.Identity())
                              
        
        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, 1)  # Single output for binary classification
        )

    def encode_image(self, image):
        """Encode image using CLIP"""
        image_features = self.clip_model.encode_image(image)
        return self.image_projection(image_features)

    def encode_text(self, text_tokens):
        """Encode text using CLIP"""
        if isinstance(text_tokens, (dict, BatchEncoding)):  # Handle both dict and BatchEncoding
            text_features = self.clip_model.encode_text(text_tokens)
        else:  # Regular CLIP
            text_features = self.clip_model.encode_text(text_tokens.squeeze(1))
        return self.text_projection(text_features)
    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning binary logits
        Args:
            batch: Dictionary containing:
                - image: Image tensor
                - question_tokens: Tokenized question
                - answer: Binary labels (0 for "no", 1 for "yes")
        Returns:
            logits: Binary classification logits
            loss: Binary cross entropy loss if labels provided
        """
        # Process image
        if isinstance(batch['image'], dict):  # Biomedical CLIP
            image_features = self.encode_image(batch['image']['pixel_values'])
        else:
            image_features = self.encode_image(batch['image'])
        
        # Process question
        question_features = self.encode_text(batch['question_tokens'])
        
        
        combined_features = torch.cat([image_features, question_features], dim=-1)
        

        
        # Get binary logits
        logits = self.classifier(combined_features).squeeze(-1)
        
        # Convert answers to binary labels if provided
        loss = None

        loss = F.binary_cross_entropy_with_logits(logits, batch['answer'].float())
        return logits, loss
class EKGRLModel(MedicalVQAModel):
    def __init__(self, 
                 clip_model, 
                 kg_retriever,
                 projection_dim: int = 512):
        # Initialize parent class
        super().__init__(clip_model, projection_dim)
        
        self.kg_retriever = kg_retriever
        
        # Entity and relation projectors
        self.F_theta_e = nn.Sequential(  # For head entity case
            nn.Linear(2*projection_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(projection_dim, 1)
        )
        
        self.F_theta_r = nn.Sequential(  # For relation/tail case
            nn.Linear(2*projection_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(projection_dim, 1)
        )
        


        # Initialize entity embeddings
        # self._initialize_entity_embeddings()
        
    def _initialize_entity_embeddings(self):
        """Initialize entity embeddings using TransE-style approach"""
        print("Initializing entity embeddings...")
        self.eval()
        device = next(self.parameters()).device
        
        # Initialize embeddings for all entities
        with torch.no_grad():
            all_embeddings = {}
            for entity_id in tqdm(self.kg_retriever.id_to_name, desc="Computing entity embeddings"):
                # Get text embedding for entity
                entity_name = self.kg_retriever.id_to_name[entity_id]
                emb = self.kg_retriever.get_concept_embedding(entity_name, device)
                if emb is not None:
                    all_embeddings[entity_id] = emb.to(device).squeeze(1)
                    
            # Convert to tensor and store
            entity_embeddings = torch.stack([all_embeddings[eid] for eid in sorted(all_embeddings.keys())])
            self.register_buffer('entity_embeddings', entity_embeddings)
            
        print("Entity embeddings initialized!")
        self.train()

    def get_entity_features(self, concept_ids: List[str]) -> torch.Tensor:
        """Get pre-computed embeddings for given concept IDs"""
        indices = [self.kg_retriever.id_to_idx[cid] for cid in concept_ids 
                  if cid in self.kg_retriever.id_to_idx]
        if not indices:
            return None
        return self.entity_embeddings[indices].mean(dim=0, keepdim=True)


    def _is_head_entity(self, concept_id: str) -> bool:
        """Check if concept appears more as head entity in KG"""
        if concept_id not in self.kg_retriever.id_to_idx:
            return False
            
        head_count = sum(len(tails) for tails in 
                        self.kg_retriever.head_to_tails.get(concept_id, {}).values())
        tail_count = sum(len(heads) for heads in 
                        self.kg_retriever.tail_to_heads.get(concept_id, {}).values())
        
        return head_count >= tail_count
    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass implementing Formula (3) from paper
        Args:
            batch: Dictionary containing:
                - image: Image tensor
                - question_tokens: Tokenized question
                - answer: Binary labels (0 for "no", 1 for "yes")
        Returns:
            logits: Binary classification logits
            loss: Binary cross entropy loss if labels provided
        """
        # Process image
        if isinstance(batch['image'], dict):  # Biomedical CLIP
            image_features = self.encode_image(batch['image']['pixel_values'])
        else:
            image_features = self.encode_image(batch['image'])
        
        # Process question
        question_features = self.encode_text(batch['question_tokens'])
        
        # Combine image and question features
        combined_features = torch.cat([image_features, question_features], dim=-1)
        
        # Get concepts from questions and check if they're head entities
        is_head = torch.zeros(len(batch['question']), device=combined_features.device)
        for i, question in enumerate(batch['question']):
            concepts = self.kg_retriever.get_concepts_from_text(question)
            if any(self._is_head_entity(cid) for cid in concepts):
                is_head[i] = 1.0
        
        # Apply Formula (3): A(i_n, q_n) = F_?e(i_n, q_n) + F_?r(i_n, q_n)   (1 - H(q_n))
        head_logits = self.F_theta_e(combined_features)
        tail_logits = self.F_theta_r(combined_features)
        logits = head_logits + tail_logits * (1 - is_head).unsqueeze(1)
        
        logits = logits.squeeze(-1)  # Remove last dimension
        
        # Compute loss if labels provided
        loss = None
        if 'answer' in batch:
            loss = F.binary_cross_entropy_with_logits(logits, batch['answer'].float())
            
        return logits, loss

class KRISPMedVQAModel(MedicalVQAModel):
    def __init__(self, 
                 clip_model, 
                 kg_retriever, 
                 projection_dim=512, 
                 hidden_dim=512, 
                 embedding_cache_path="kg_embedding_cache.pt"):
        super().__init__(clip_model, projection_dim)
        self.kg_retriever = kg_retriever
        self.projection_dim = projection_dim
        self.embedding_cache_path = embedding_cache_path

        self.gcn = nn.ModuleList([
            GCNConv(projection_dim, projection_dim)
        ])

        self.symbolic_classifier = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(projection_dim, 1)
        )

        self._initialize_fixed_node_features()

    def _initialize_fixed_node_features(self):
        device = next(self.parameters()).device

        if os.path.exists(self.embedding_cache_path):
            print(f"Loading precomputed concept embeddings from {self.embedding_cache_path}...")
            self.X_fixed = torch.load(self.embedding_cache_path, map_location=device)
            return

        print("Precomputing fixed concept embeddings...")
        node_ids = list(self.kg_retriever.id_to_idx.keys())
        embeddings = []

        for node_id in node_ids:
            if node_id.startswith('I') and node_id in self.kg_retriever.image_id_to_path:
                image_path = self.kg_retriever.image_id_to_path[node_id]
                try:
                    image = Image.open(image_path).convert('RGB')
                    if self.kg_retriever.biomedclip:
                        rel_img_features = self.kg_retriever.processor(
                            images=image,
                            return_tensors="pt"
                        )['pixel_values'].to(device)
                    else:
                        rel_img_features = self.kg_retriever.processor(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = self.kg_retriever.clip_model.encode_image(rel_img_features)
                except Exception as e:
                    print(f"Failed to load image for {node_id}: {e}")
                    emb = torch.zeros((1, self.projection_dim), device=device)
            else:
                name = self.kg_retriever.id_to_name[node_id]
                if self.kg_retriever.biomedclip:
                    tokens = self.kg_retriever.processor(
                        text=name,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=77
                    ).to(device)
                    with torch.no_grad():
                        emb = self.kg_retriever.clip_model.encode_text(tokens)
                else:
                    tokens = self.kg_retriever.tokenizer(name).to(device)
                    with torch.no_grad():
                        emb = self.kg_retriever.clip_model.encode_text(tokens)

            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb)

        self.X_fixed = torch.cat(embeddings, dim=0)
        print(f"Saving embeddings to {self.embedding_cache_path}...")
        torch.save(self.X_fixed, self.embedding_cache_path)

    def extract_subgraph_batch(self, batch_subgraph_indices, batch_edge_lists, device):
        data_list = []
        for node_indices, edge_list in zip(batch_subgraph_indices, batch_edge_lists):
            if not node_indices:
                data_list.append(Data(
                    x=torch.zeros(1, self.projection_dim).to(device),
                    edge_index=torch.zeros(2, 0, dtype=torch.long).to(device)))
                continue

            x = self.X_fixed[torch.tensor(node_indices, dtype=torch.long, device=device)]
            # x = self.node_projection(x)

            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)
            data_list.append(Data(x=x, edge_index=edge_index))

        return Batch.from_data_list(data_list)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        y_implicit, base_loss = super().forward(batch)
        device = next(self.parameters()).device

        graph_batch = self.extract_subgraph_batch(
            batch['subgraph'],
            batch['subgraph_edges'],
            device
        )

        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch_vec = graph_batch.batch

        for conv in self.gcn:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.8, training=self.training)

        graph_features = global_mean_pool(x, batch=batch_vec)
        y_symbolic = self.symbolic_classifier(graph_features).squeeze(-1)

        logits = torch.max(y_implicit, y_symbolic)

        loss = None
        if 'answer' in batch:
            loss = F.binary_cross_entropy_with_logits(logits, batch['answer'].float().to(device))

        return logits, loss



class MKBNModel(MedicalVQAModel):
    def __init__(self, 
                 clip_model, 
                 kg_retriever,
                 projection_dim: int = 512,
                 hidden_dim: int = 512,
                 num_heads: int = 1,
                 embedding_cache_path: str = "kg_embedding_cache.pt"):
        # super().__init__()
        super().__init__(clip_model, projection_dim)
        self.clip_model = clip_model
        self.kg_retriever = kg_retriever
        self.projection_dim = projection_dim
        self.embedding_cache_path = embedding_cache_path

        self.gcn = nn.ModuleList([
            GCNConv(projection_dim, projection_dim)
        ])

        self.img_ques_attention = MultiheadAttention(projection_dim, num_heads, batch_first=True)
        self.ques_graph_attention = MultiheadAttention(projection_dim, num_heads, batch_first=True)
        self.img_graph_attention = MultiheadAttention(projection_dim, num_heads, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(projection_dim * 3, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(projection_dim, 1)
        )

        self._initialize_fixed_node_features()

    def _initialize_fixed_node_features(self):
        device = next(self.parameters()).device

        if os.path.exists(self.embedding_cache_path):
            print(f"Loading precomputed concept embeddings from {self.embedding_cache_path}...")
            self.X_fixed = torch.load(self.embedding_cache_path, map_location=device)
            return

        print("Precomputing fixed concept embeddings...")
        node_ids = list(self.kg_retriever.id_to_idx.keys())
        embeddings = []

        for node_id in node_ids:
            if node_id.startswith('I') and node_id in self.kg_retriever.image_id_to_path:
                image_path = self.kg_retriever.image_id_to_path[node_id]
                try:
                    image = Image.open(image_path).convert('RGB')
                    if self.kg_retriever.biomedclip:
                        rel_img_features = self.kg_retriever.processor(
                            images=image,
                            return_tensors="pt"
                        )['pixel_values'].to(device)
                    else:
                        rel_img_features = self.kg_retriever.processor(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = self.kg_retriever.clip_model.encode_image(rel_img_features)
                except Exception as e:
                    print(f"Failed to load image for {node_id}: {e}")
                    emb = torch.zeros((1, self.projection_dim), device=device)
            else:
                name = self.kg_retriever.id_to_name[node_id]
                if self.kg_retriever.biomedclip:
                    tokens = self.kg_retriever.processor(
                        text=name,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=77
                    ).to(device)
                    with torch.no_grad():
                        emb = self.kg_retriever.clip_model.encode_text(tokens)
                else:
                    # tokens = self.kg_retriever.tokenizer(name, return_tensors="pt", padding=True, truncation=True).to(device)
                    tokens = self.kg_retriever.tokenizer(name).to(device)
                    with torch.no_grad():
                        emb = self.kg_retriever.clip_model.encode_text(tokens)
                        # emb = self.kg_retriever.clip_model.encode_text(tokens["input_ids"])

            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb)

        self.X_fixed = torch.cat(embeddings, dim=0)

        print(f"Saving embeddings to {self.embedding_cache_path}...")
        torch.save(self.X_fixed, self.embedding_cache_path)



    def _initialize_graph_structure(self):
        edge_index = []
        for src_id, relations in self.kg_retriever.head_to_tails.items():
            src_idx = self.kg_retriever.id_to_idx[src_id]
            for tails in relations.values():
                for tail in tails:
                    if tail in self.kg_retriever.id_to_idx:
                        tail_idx = self.kg_retriever.id_to_idx[tail]
                        edge_index.append([src_idx, tail_idx])
                        edge_index.append([tail_idx, src_idx])
        self.graph_edge_index = torch.tensor(edge_index, dtype=torch.long).t()


    def extract_subgraph_batch(self, batch_subgraph_indices: List[List[int]], batch_edge_lists: List[List[List[int]]], device) -> Batch:
        data_list = []
        for node_indices, edge_list in zip(batch_subgraph_indices, batch_edge_lists):
            if not node_indices:
                data_list.append(Data(x=torch.zeros(1, self.projection_dim).to(device),
                                      edge_index=torch.zeros(2, 0, dtype=torch.long).to(device)))
                continue

            edge_index = torch.tensor(edge_list, dtype=torch.long).t().to(device)
            x = self.X_fixed[node_indices].to(device)
            data_list.append(Data(x=x, edge_index=edge_index))

        return Batch.from_data_list(data_list)



    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device

        image_features = self.clip_model.encode_image(batch['image'].to(device))
        # question_features = self.clip_model.encode_text(batch['question_tokens'])
        question_features = self.encode_text(batch['question_tokens'])
        graph_batch = self.extract_subgraph_batch(batch['subgraph'], batch['subgraph_edges'], device)

        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch_vec = graph_batch.batch
        for conv in self.gcn:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.8, training=self.training)
 
        graph_features = global_mean_pool(x, batch=batch_vec)

        img_ques_feat, _ = self.img_ques_attention(
            image_features.unsqueeze(1), question_features.unsqueeze(1), question_features.unsqueeze(1)
        )
        ques_graph_feat, _ = self.ques_graph_attention(
            question_features.unsqueeze(1), graph_features.unsqueeze(1), graph_features.unsqueeze(1)
        )
        img_graph_feat, _ = self.img_graph_attention(
            image_features.unsqueeze(1), graph_features.unsqueeze(1), graph_features.unsqueeze(1)
        )

        combined_features = torch.cat([
            img_ques_feat.squeeze(1),
            ques_graph_feat.squeeze(1),
            img_graph_feat.squeeze(1)
        ], dim=-1)


        logits = self.classifier(combined_features).squeeze(-1)

        loss = None
        if 'answer' in batch:
            loss = F.binary_cross_entropy_with_logits(logits, batch['answer'].float().to(device))

        return logits, loss



# MR-MKG

class GCNVQAModel(MedicalVQAModel):
    def __init__(self, 
                 clip_model, 
                 kg_retriever,
                 projection_dim: int = 64,
                 hidden_dim: int = 512,
                 lambda_weight: float = 0.1,
                 alpha: float = 0.2,
                 biomedclip: bool = False,
                 embedding_cache_path: str = "kg_embedding_cache.pt"):
        super().__init__(clip_model, projection_dim)
        self.clip_model = clip_model
        self.kg_retriever = kg_retriever
        self.projection_dim = projection_dim
        self.lambda_weight = lambda_weight
        self.alpha = alpha
        self.biomedclip = biomedclip
        self.embedding_cache_path = embedding_cache_path
        
        self.node_projection = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.gcn = nn.ModuleList([
            GCNConv(projection_dim, projection_dim)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(projection_dim * 3, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, 1)
        )

        self.kg_retriever.build_alignment_triplet_pool(num_triplets=1000)
        self._initialize_fixed_node_features()

    def _initialize_fixed_node_features(self):
        device = next(self.parameters()).device

        if os.path.exists(self.embedding_cache_path):
            print(f"Loading precomputed concept embeddings from {self.embedding_cache_path}...")
            self.X_fixed = torch.load(self.embedding_cache_path, map_location=device)
            return

        print("Precomputing fixed concept embeddings...")
        node_ids = list(self.kg_retriever.id_to_idx.keys())
        embeddings = []

        for node_id in node_ids:
            if node_id.startswith('I') and node_id in self.kg_retriever.image_id_to_path:
                image_path = self.kg_retriever.image_id_to_path[node_id]
                try:
                    image = Image.open(image_path).convert('RGB')
                    if self.kg_retriever.biomedclip:
                        rel_img_features = self.kg_retriever.processor(images=image, return_tensors="pt")['pixel_values'].to(device)
                    else:
                        rel_img_features = self.kg_retriever.processor(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = self.kg_retriever.clip_model.encode_image(rel_img_features)
                except Exception as e:
                    print(f"Failed to load image for {node_id}: {e}")
                    emb = torch.zeros((1, self.projection_dim), device=device)
            else:
                name = self.kg_retriever.id_to_name[node_id]
                if self.kg_retriever.biomedclip:
                    tokens = self.kg_retriever.processor(
                        text=name,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=77
                    ).to(device)
                    with torch.no_grad():
                        emb = self.kg_retriever.clip_model.encode_text(tokens)
                else:
                    tokens = self.kg_retriever.tokenizer(name).to(device)
                    with torch.no_grad():
                        emb = self.kg_retriever.clip_model.encode_text(tokens)

            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb)

        self.X_fixed = torch.cat(embeddings, dim=0)
        print(f"Saving embeddings to {self.embedding_cache_path}...")
        torch.save(self.X_fixed, self.embedding_cache_path)

    def extract_subgraph_batch(self, batch_subgraph_indices, batch_edge_lists, device):
        data_list = []
        for node_indices, edge_list in zip(batch_subgraph_indices, batch_edge_lists):
            if not node_indices:
                data_list.append(Data(
                    x=torch.zeros(1, self.projection_dim).to(device),
                    edge_index=torch.zeros(2, 0, dtype=torch.long).to(device)))
                continue

            x = self.X_fixed[torch.tensor(node_indices, dtype=torch.long, device=device)]
            x = self.node_projection(x)

            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)
            data_list.append(Data(x=x, edge_index=edge_index))

        return Batch.from_data_list(data_list)
    def sample_triplet_batch(self, device, num_samples: int = 16):
        triplets = random.sample(self.kg_retriever.triplet_pool, min(num_samples, len(self.kg_retriever.triplet_pool)))
        anchor_embs, pos_embs, neg_embs = [], [], []

        for img_id, pos_id, neg_id in triplets:
            anchor = self.kg_retriever.get_entity_embedding_by_id(img_id, device)
            pos = self.kg_retriever.get_entity_embedding_by_id(pos_id, device)
            neg = self.kg_retriever.get_entity_embedding_by_id(neg_id, device)

            if anchor is not None and pos is not None and neg is not None:
                anchor_embs.append(anchor.squeeze(0))
                pos_embs.append(pos.squeeze(0))
                neg_embs.append(neg.squeeze(0))

        if anchor_embs and pos_embs and neg_embs:
            return torch.stack(anchor_embs), torch.stack(pos_embs), torch.stack(neg_embs)
        return None, None, None

    def compute_alignment_loss(self, anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        d_pos = F.pairwise_distance(anchors, positives, p=2)
        d_neg = F.pairwise_distance(anchors, negatives, p=2)
        loss = F.relu(d_pos - d_neg + self.alpha)
        return loss.mean()

    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        image_features = self.encode_image(batch['image']['pixel_values'] if isinstance(batch['image'], dict) else batch['image'])
        question_features = self.encode_text(batch['question_tokens'])

        graph_batch = self.extract_subgraph_batch(
            batch['subgraph'],
            batch['subgraph_edges'],
            device
        )

        x = graph_batch.x
        for conv in self.gcn:
            x = conv(x, graph_batch.edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)

        graph_features = global_mean_pool(x, batch=graph_batch.batch)
        combined_features = torch.cat([image_features, question_features, graph_features], dim=-1)
        logits = self.classifier(combined_features).squeeze(-1)

        loss = None
        if 'answer' in batch:
            class_loss = F.binary_cross_entropy_with_logits(logits, batch['answer'].float().to(device))
            align_loss = torch.tensor(0.0).to(device)

            anchors, positives, negatives = self.sample_triplet_batch(device, num_samples=16)
            if anchors is not None:
                align_loss = self.compute_alignment_loss(anchors, positives, negatives)

            loss = class_loss + self.lambda_weight * align_loss

        return logits, loss
class KPathVQAModel(MedicalVQAModel):
    def __init__(self, 
                 clip_model, 
                 kg_retriever,
                 projection_dim: int = 512,
                 num_heads: int = 1,
                 biomedclip: bool = False):
        super().__init__(clip_model, projection_dim)
        
        self.kg_retriever = kg_retriever
        self.device = next(self.parameters()).device
        self.biomedclip = biomedclip
        self.projection_dim = projection_dim
        # Build concept to triplets dictionary during initialization
        self.concept_to_triplets = {}
        for triplet in self.kg_retriever.triplets:
            # Process head name
            head_name = triplet.head_name.lower()
            if head_name not in self.concept_to_triplets:
                self.concept_to_triplets[head_name] = []
            triplet_text = f"{triplet.head_name} {triplet.relation} {triplet.tail_name}"
            self.concept_to_triplets[head_name].append(triplet_text)
            
            # Process tail name
            tail_name = triplet.tail_name.lower()
            if tail_name not in self.concept_to_triplets:
                self.concept_to_triplets[tail_name] = []
            self.concept_to_triplets[tail_name].append(triplet_text)
        
        # Rest of initialization remains the same

        self.img_text_attention = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=num_heads,
            batch_first=True
        ).to(self.device)
        
        self.text_concept_attention = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=num_heads,
            batch_first=True
        ).to(self.device)
        
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(projection_dim, 1)
        ).to(self.device)

    def get_triplet_embeddings(self, concept_name: str) -> Optional[torch.Tensor]:
        """
        Get embedding for a single triplet containing the concept using lookup dictionary
        """
        concept_name = concept_name.lower()
        if concept_name not in self.concept_to_triplets or not self.concept_to_triplets[concept_name]:
            return None
            
        # Randomly sample one triplet text
        triplet_text = random.choice(self.concept_to_triplets[concept_name])
        
        # Get embedding for the triplet text
        if self.biomedclip:
            triplet_tokens = self.kg_retriever.processor(
                text=triplet_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device)
        else:
            triplet_tokens = self.kg_retriever.tokenizer(triplet_text).to(self.device)
        
        # Get embedding using text encoder and ensure shape consistency
        with torch.no_grad():
            triplet_emb = self.encode_text(triplet_tokens)
            return triplet_emb.squeeze()  # Remove any extra dimensions

    
    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get image and text features
        if isinstance(batch['image'], dict):
            image_features = self.encode_image(batch['image']['pixel_values'])
        else:
            image_features = self.encode_image(batch['image'])
        
        question_features = self.encode_text(batch['question_tokens'])
        
        # Get concepts and their triplet embeddings
        batch_size = len(batch['question'])
        concept_embeddings = []
        
        for i in range(batch_size):
            # Get concepts from question
            concept_names = set()
            for name in self.kg_retriever.name_to_ids.keys():
                if name.lower() in batch['question'][i].lower():
                    concept_names.add(name)
            
            if concept_names:
                # Randomly select one concept and get its triplet embedding
                concept_name = random.choice(list(concept_names))
                triplet_emb = self.get_triplet_embeddings(concept_name)
                
                if triplet_emb is not None:
                    # Ensure the embedding is 1D before adding to list
                    concept_embeddings.append(triplet_emb.squeeze())
                else:
                    concept_embeddings.append(torch.zeros(self.projection_dim, device=self.device))
            else:
                concept_embeddings.append(torch.zeros(self.projection_dim, device=self.device))
        
        # Stack the embeddings - now all should be shape [512]
        concept_embeddings = torch.stack(concept_embeddings)
        
        # Rest of the code remains the same...
        # Attention-based fusion
        img_text_fused, _ = self.img_text_attention(
            image_features.unsqueeze(1),
            question_features.unsqueeze(1),
            question_features.unsqueeze(1)
        )
        
        text_concept_fused, _ = self.text_concept_attention(
            question_features.unsqueeze(1),
            concept_embeddings.unsqueeze(1),
            concept_embeddings.unsqueeze(1)
        )
        
        combined_features = torch.cat([
            img_text_fused.squeeze(1),
            text_concept_fused.squeeze(1)
        ], dim=-1)
        
        logits = self.classifier(combined_features).squeeze(-1)
        
        loss = None
        if 'answer' in batch:
            loss = F.binary_cross_entropy_with_logits(
                logits,
                batch['answer'].float()
            )
            
        return logits, loss

