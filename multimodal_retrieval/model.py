import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pandas as pd
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
import numpy as np
from open_clip import create_model_from_pretrained, get_tokenizer
from pathlib import Path
from tqdm import tqdm 
from PIL import Image
import random
import time
import os
import pickle
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
        
        
        
        
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, AutoTokenizer, AutoModel
import pandas as pd
from typing import List, Dict, Optional, Set, Tuple
import numpy as np
from dataclasses import dataclass
import os

@dataclass
class KGTriplet:
    head: str  # Head ID
    head_name: str  # Head name/description
    relation: str  # type of relation
    tail: str  # Tail ID
    tail_name: str  # Tail name/description

class KnowledgeGraphRetriever:
   def __init__(self, kg_path: str, image_map_path: str = "image_mapping.csv", clip_model=None, processor=None,
                lambda_diversity: float = 0.8, biomedclip = False):
       self.clip_model = clip_model
       self.processor = processor
       self.lambda_diversity = lambda_diversity
       
       # Load image mapping
       df_image_map = pd.read_csv(image_map_path)
       self.image_id_to_path = dict(zip(df_image_map['IID'], df_image_map['Image_Path']))
       
       # Load KG and create mappings
       self.triplets = self._load_kg(kg_path)
       self._create_mappings()
       self.biomedclip = biomedclip
       
   def _load_kg(self, kg_path: str) -> List[KGTriplet]:
       if not Path(kg_path).exists():
           raise FileNotFoundError(f"Knowledge graph file not found: {kg_path}")
           
       df = pd.read_csv(kg_path)
       return [
           KGTriplet(
               head=str(row['Head']),
               head_name=str(row['Head_Name']),
               relation=str(row['Relation']),
               tail=str(row['Tail']),
               tail_name=str(row['Tail_Name'])
           )
           for _, row in df.iterrows()
           if row['Head'].startswith('I')
       ]

   def _create_mappings(self):
       self.id_to_name = {}
       self.name_to_ids = {}
       self.tail_to_heads = {}
       self.head_to_tails = {}
       
       for triplet in self.triplets:
           # ID-name mappings
           self.id_to_name[triplet.head] = triplet.head_name
           self.id_to_name[triplet.tail] = triplet.tail_name
           
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
               
   def compute_raw_similarity(self, query_image, candidate_path: str) -> float:
       try:
           candidate_image = Image.open(candidate_path).convert('RGB')
           candidate_processed = self.processor(candidate_image).unsqueeze(0)
           with torch.no_grad():
               sim = F.cosine_similarity(query_image, candidate_processed)
           return sim.item()
       except:
           return 0.0

   def _mmr_selection(self, candidates: Set[str], query_image, num_select: int = 1) -> List[str]:
    if not candidates:
        return []
        
    device = next(self.clip_model.parameters()).device
    query_image = query_image.to(device)

    # Batch process all images at once
    batch_size = 32
    all_embeddings = []
    valid_candidates = []
    
    candidate_list = list(candidates)
    for i in range(0, len(candidate_list), batch_size):
        batch_candidates = candidate_list[i:i + batch_size]
        batch_images = []
        batch_valid = []
        
        for candidate in batch_candidates:
            if candidate in self.image_id_to_path:
                try:
                    img = Image.open(self.image_id_to_path[candidate]).convert('RGB')
                    img = self.processor(img)
                    batch_images.append(img)
                    batch_valid.append(candidate)
                except:
                    continue
                    
        if batch_images:
            with torch.no_grad():
                batch_tensor = torch.stack(batch_images).to(device)
                batch_embeddings = self.clip_model.encode_image(batch_tensor)
                all_embeddings.append(batch_embeddings)
                valid_candidates.extend(batch_valid)

    if not valid_candidates:
        return []

    # Compute all similarities at once
    all_embeddings = torch.cat(all_embeddings)
    query_embedding = self.clip_model.encode_image(query_image.unsqueeze(0))
    similarities = F.cosine_similarity(query_embedding, all_embeddings)

    selected = []
    remaining = list(range(len(valid_candidates)))
    
    while len(selected) < num_select and remaining:
        if not selected:
            idx = remaining[similarities[remaining].argmax().item()]
            selected.append(idx)
            remaining.remove(idx)
        else:
            # Compute diversity penalty
            penalties = torch.max(F.cosine_similarity(
                all_embeddings[remaining].unsqueeze(1), 
                all_embeddings[selected].unsqueeze(0)
            ), dim=1)[0]
            
            scores = self.lambda_diversity * similarities[remaining] - \
                    (1 - self.lambda_diversity) * penalties
                    
            idx = remaining[scores.argmax().item()]
            selected.append(idx)
            remaining.remove(idx)

    return [valid_candidates[i] for i in selected]

   def get_related_concepts(self, concept: str, relation_type: str = None, use_names: bool = True) -> Set[str]:
       """Get related heads (images) for a given tail (concept)"""
       related = set()
       concept_ids = {concept} if not use_names else self.name_to_ids.get(concept, set())
       
       for concept_id in concept_ids:
           # For each concept (tail), get all related images (heads)
           if concept_id in self.tail_to_heads:
               for rel_type, heads in self.tail_to_heads[concept_id].items():
                   if relation_type is None or rel_type == relation_type:
                       if use_names:
                           related.update(self.id_to_name[h] for h in heads)
                       else:
                           related.update(heads)
       
       return related

   def _batch_mmr_selection(self, candidates_list, query_images, num_select: int = 1):
    device = next(self.clip_model.parameters()).device
    
    # Process all candidates at once
    all_candidate_embeddings = {}
    valid_candidates_map = {}
    
    # Unique candidates across all queries
    unique_candidates = set().union(*candidates_list)
    
    # Batch process candidate images
    batch_size = 32
    candidate_list = list(unique_candidates)
    # print(candidate_list)
    for i in range(0, len(candidate_list), batch_size):
        batch = candidate_list[i:i + batch_size]
        with torch.no_grad():
            batch_imgs = []
            valid_ids = []
            for cand in batch:
                if cand in self.image_id_to_path:
                    if self.biomedclip: 
                        # try:
                            # image = Image.open(image_path).convert('RGB')
                            img = Image.open(self.image_id_to_path[cand]).convert('RGB')
                            processed_img = self.processor(
                            text=None, 
                            images=img, 
                            return_tensors="pt", 
                            padding=True,
                            truncation=True,
                            max_length=512
                            )['pixel_values']
                            batch_imgs.append(processed_img.to(query_images.device))
                            valid_ids.append(cand)
                        # except:
                        #    continue
                    else:
                        # try:
                            img = Image.open(self.image_id_to_path[cand]).convert('RGB')
                            batch_imgs.append(self.processor(img).to(query_images.device))
                            valid_ids.append(cand)
                        # except:
                        #     continue
            
            if batch_imgs:
                batch_tensor = torch.stack(batch_imgs).squeeze(1).to(device)
                # embeddings = batch_tensor
                embeddings = self.clip_model.encode_image(batch_tensor)
                for j, cand_id in enumerate(valid_ids):
                    all_candidate_embeddings[cand_id] = embeddings[j]

    # Process each query
    results = []
    # query_embeddings = query_images
    query_embeddings = self.clip_model.encode_image(query_images)

      
    
    for query_idx, (query_embedding, candidates) in enumerate(zip(query_embeddings, candidates_list)):
        valid_candidates = [c for c in candidates if c in all_candidate_embeddings]
        if not valid_candidates:
            results.append([])
            continue
            
        embeddings = torch.stack([all_candidate_embeddings[c] for c in valid_candidates])
        similarities = F.cosine_similarity(query_embedding.unsqueeze(0), embeddings)
        
        selected = []
        remaining = list(range(len(valid_candidates)))
        
        while len(selected) < num_select and remaining:
            if not selected:
                idx = remaining[similarities[remaining].argmax().item()]
            else:
                penalties = torch.max(F.cosine_similarity(
                    embeddings[remaining].unsqueeze(1),
                    embeddings[selected].unsqueeze(0)
                ), dim=1)[0]
                
                scores = self.lambda_diversity * similarities[remaining] - \
                        (1 - self.lambda_diversity) * penalties
                idx = remaining[scores.argmax().item()]
                
            selected.append(idx)
            remaining.remove(idx)
            
        results.append([valid_candidates[i] for i in selected])
        
    return results

   
   def retrieve_batch(self, query_texts, query_images=None, relation_type="POSITIVE", num_results=1):
    cache_path = "mimic_retrieved_candidates_cache.pkl"
    
    # Try to load from cache first
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}
    
    all_candidates = []
    
    for text in query_texts:
        # If cached, use the stored candidates
        if text in cache:
            candidates = cache[text]
        else:
            # Otherwise, compute as before
            matched_concepts = {
                concept_id 
                for name in self.name_to_ids
                if name.lower() in text.lower()
                for concept_id in self.name_to_ids[name]
            }
            if matched_concepts:
                matched_concepts = set(random.sample(list(matched_concepts), 1))
            candidates = set()
            for concept in matched_concepts:
                related = self.get_related_concepts(concept, relation_type, use_names=False)
                candidates.update(related)
            
            if len(candidates) > 1:
                candidates = set(random.sample(list(candidates), 1))
            
            # Save to cache
            cache[text] = candidates
        
        all_candidates.append(candidates)
    
    # Save the updated cache
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    
    return all_candidates
    # print("all_candidates:", all_candidates)
    if query_images is None:
        return [list(c)[:num_results] for c in all_candidates]
        
    return self._batch_mmr_selection(all_candidates, query_images, num_results)
   def retrieve(self, query_text: str, query_image=None, relation_type: str = "positive", num_results: int = 1) -> List[str]:
       matched_concepts = set()
       query_lower = query_text.lower()
       p_time = time.time()
       # Match concepts
       for concept_name in self.name_to_ids:
           if concept_name.lower() in query_lower:
               matched_concepts.update(self.name_to_ids[concept_name])
       # print(time.time() - p_time)
       p_time = time.time()
       # Get candidate images
       image_candidates = set()
       for concept in matched_concepts:
           related = self.get_related_concepts(concept, relation_type, use_names=False)
           image_candidates.update(related)
       # print(time.time() - p_time)
       p_time = time.time()
       # Randomly sample 10 candidates if too many
       if len(image_candidates) > 3:
           image_candidates = set(random.sample(list(image_candidates), 3))
           
       if query_image is None or not image_candidates:
           return list(image_candidates)[:num_results]
       print(time.time() - p_time)
       p_time = time.time()
       result = self._mmr_selection(image_candidates, query_image, num_results)
       print(time.time() - p_time)
       return result

class OriginalCLIP(nn.Module):
    def __init__(self, model_base):
        super().__init__()
        self.model = model_base
    def forward(self, inputs, texts):
        image_features =  self.model.get_image_features(pixel_values=inputs['pixel_values'])
        text_features = self.model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        return image_features, text_features
    def encode_image(self, image):
        return self.model.get_image_features(pixel_values=image)
    def encode_text(self, text):
        return self.model.get_text_features(input_ids=text['input_ids'].to(self.model.device), attention_mask=text['attention_mask'].to(self.model.device))

class FashionKLIP(nn.Module):
   def __init__(self, clip_model, kg_retriever, projection_dim: int = 512,
                num_prototype_images: int = 1):
       super().__init__()
       self.clip_model = clip_model
       self.kg_retriever = kg_retriever
       self.num_prototype_images = num_prototype_images
       
       clip_dim = getattr(clip_model, 'embed_dim', 
                 getattr(clip_model, 'config.hidden_size', 512))
       
       self.image_projection = nn.Linear(clip_dim, projection_dim) \
                             if clip_dim != projection_dim else nn.Identity()
       self.text_projection = nn.Linear(clip_dim, projection_dim) \
                             if clip_dim != projection_dim else nn.Identity()
   
   def encode_image(self, image):
       image_features = self.clip_model.encode_image(image)
       return self.image_projection(image_features)
   
   def encode_text(self, text_tokens):
       text_features = self.clip_model.encode_text(text_tokens)
       return self.text_projection(text_features)

   def forward(self, batch):
    images = batch['image'].squeeze(1)
    if type(batch['text_tokens']) == dict:
        text_tokens = batch['text_tokens']
        texts = batch['text']
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_tokens)
    else:
    
        text_tokens = batch['text_tokens'].squeeze(1)
        texts = batch['text']
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_tokens)
    
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    
    similarity_itc = torch.matmul(text_features, image_features.T)
    
    itc_i2t = -torch.diagonal(
           F.log_softmax(similarity_itc / 0.07, dim=1)
       ).mean()
    itc_t2i = -torch.diagonal(
           F.log_softmax(similarity_itc.T / 0.07, dim=1)
       ).mean()
    loss_cva = (itc_i2t + itc_t2i) / 2
    
#    # Batch retrieve prototypes
    prototype_images = self.kg_retriever.retrieve_batch(texts, images)
    loss_cva = self._compute_cva_loss(
        prototype_images, image_features, text_features, images.device
    )
    
    
    
    return similarity_itc, loss_cva
   def _compute_cva_loss(self, prototype_images, image_features, text_features, device):
     loss_cva = torch.zeros(1, device=device)
     batch_size = len(image_features)
     for i, (img_feat, prototypes) in enumerate(zip(image_features, prototype_images)):
         # print(len(prototypes))
         if not prototypes:
             continue
             
         # Load and encode prototypes
         prototype_tensors = []
         for img_id in prototypes:
             if img_id in self.kg_retriever.image_id_to_path:
                 if self.kg_retriever.biomedclip:
                     # try:
                         img = Image.open(self.kg_retriever.image_id_to_path[img_id]).convert('RGB')
                         img = self.kg_retriever.processor(
                            images=img, 
                            return_tensors="pt", 
                            max_length=512
                            )['pixel_values'].to(device)
                         prototype_tensors.append(img.squeeze(1))
                     # except:
                     #     continue                      
                 else:
                 
                     # try:
                         img = Image.open(self.kg_retriever.image_id_to_path[img_id]).convert('RGB')
                         img = self.kg_retriever.processor(img).to(device)
                         prototype_tensors.append(img)
                     # except:
                     #     continue
                     
         if not prototype_tensors:
             continue
             
         prototype_tensors = torch.stack(prototype_tensors).squeeze(1)
         prototype_features = F.normalize(self.encode_image(prototype_tensors), dim=-1)
         
         weights = F.cosine_similarity(
             img_feat.unsqueeze(0).expand_as(prototype_features),
             prototype_features
         )
         
         prototype_text_sim = torch.matmul(prototype_features, text_features.T)
         
         for j, weight in enumerate(weights):
             pos_sim = prototype_text_sim[j, i]
             neg_sim = prototype_text_sim[j, torch.arange(batch_size) != i]
             loss_cva -= weight * (pos_sim - torch.logsumexp(torch.cat([pos_sim.unsqueeze(0), neg_sim]), dim=0))
  
         loss_cva /= len(prototype_features)
         
     return loss_cva / batch_size
class CombinedLoss(nn.Module):
   def __init__(self, temperature: float = 0.07, cva_weight: float = 0.5):
       super().__init__()
       self.temperature = temperature
       self.cva_weight = cva_weight
       
   def forward(self, similarity_itc: torch.Tensor, loss_cva: torch.Tensor) -> torch.Tensor:
       itc_i2t = -torch.diagonal(
           F.log_softmax(similarity_itc / self.temperature, dim=1)
       ).mean()
       itc_t2i = -torch.diagonal(
           F.log_softmax(similarity_itc.T / self.temperature, dim=1)
       ).mean()
       loss_itc = (itc_i2t + itc_t2i) / 2
       
       # return loss_itc    
       return loss_itc
       # return (1 - self.cva_weight) * loss_itc + self.cva_weight * loss_cva
        
        
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from PIL import Image


from torch_geometric.nn import MessagePassing

class GNN(nn.Module):
    """PyG-compatible GNN for subgraph training"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.num_layers):
            row, col = edge_index
            # Aggregate neighbor features: sum aggregation
            agg_messages = torch.zeros_like(x)
            agg_messages.index_add_(0, row, x[col])

            x_new = F.relu(self.convs[i](agg_messages))
            if i > 0:
                x_new = x_new + agg_messages  # Residual connection
            x = x_new
        
        return self.projection(x)














class KnowledgeClip(nn.Module):
    def __init__(self, clip_model, clip_name, kg_dataset, projection_dim: int = 512, gnn_hidden_dim: int = 512, num_neighbors: int = 10):
        super().__init__()
        self.clip_model = clip_model
        self.clip_name = clip_name
        # Device
        self.device = next(clip_model.parameters()).device
        
        # Dimensions
        clip_dim = getattr(clip_model, 'embed_dim', 512)
        self.feature_dim = projection_dim
        
#        # Projection layers
#        self.image_projection = (nn.Linear(clip_dim, projection_dim) if clip_dim != projection_dim else nn.Identity()).to(self.device)
#        self.text_projection = (nn.Linear(clip_dim, projection_dim) if clip_dim != projection_dim else nn.Identity()).to(self.device)
        
        # Fusion encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=projection_dim, nhead=8, dim_feedforward=512).to(self.device)
        self.multimodal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1).to(self.device)
        
        # GNN
        self.gnn = GNN(input_dim=projection_dim, hidden_dim=gnn_hidden_dim).to(self.device)
        
        # Relation classifier
        num_relations = kg_dataset.get_num_relations()
        self.relation_classifier = nn.Sequential(
            nn.Linear(2 * projection_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(projection_dim, num_relations)
        ).to(self.device)

#        # Cache features (fixed)
#        self.entity_features = self._initialize_entity_features(kg_dataset, clip_name = self.clip_name)  # (num_entities, projection_dim)
#        
#        # Build the full graph structure for NeighborLoader
#        edge_index = kg_dataset.get_edge_index().to(self.device)  # should be (2, num_edges)
#        
#        # self.graph_data = {'x': self.entity_features, 'edge_index': edge_index}
#        self.graph_data = Data(x=self.entity_features, edge_index=edge_index)
#        self.graph_edge_index = edge_index
#
#
#        
#        # Neighbor sampler
#        self.neighbor_sampler = NeighborLoader(
#            self.graph_data,
#            num_neighbors=[num_neighbors] * 2,  # 2-hop neighbors
#            input_nodes=None,
#            batch_size=128,
#            shuffle=True
#        )

    def _initialize_entity_features(self, kg_dataset, clip_name: str, cache_dir: str = "./entity_cache"):
        """Precompute entity features from images/texts, with cache"""
        print("Initializing entity features...")
    
        os.makedirs(cache_dir, exist_ok=True)
    
        # Sanitize clip name: replace slashes with hyphens
        safe_clip_name = clip_name.replace('/', '-')
    
        cache_path = os.path.join(cache_dir, f"entity_features_{safe_clip_name}.pt")
    
        # If cache exists, load it
        if os.path.exists(cache_path):
            print(f"Loading entity features from cache: {cache_path}")
            all_features = torch.load(cache_path, map_location=self.device)
            self.train()
            return all_features
    
        # Otherwise, compute
        self.eval()
    
        with torch.no_grad():
            adj_matrix = kg_dataset.get_adj_matrix().to(self.device)
            num_entities = adj_matrix.size(0)
            all_features = torch.zeros((num_entities, self.feature_dim)).to(self.device)
            valid_mask = torch.zeros(num_entities, dtype=torch.bool).to(self.device)
    
            # Separate image/text indices
            image_indices, text_indices = [], []
            for idx, entity in kg_dataset.idx_to_entity.items():
                if entity.startswith('I'):
                    image_indices.append(idx)
                else:
                    text_indices.append(idx)
    
            batch_size = 1024
    
            # Process text entities
            for i in range(0, len(text_indices), batch_size):
                batch_idx = text_indices[i:i + batch_size]
                batch_entities = [kg_dataset.idx_to_entity[idx] for idx in batch_idx]
                batch_features = kg_dataset._process_text_batch(batch_entities)
    
                # Handle valid entries
                valid_features = []
                valid_indices = []
                for idx, feat in zip(batch_idx, batch_features):
                    if feat is not None:
                        valid_features.append(feat)
                        valid_indices.append(idx)
    
                if valid_features:
                    if isinstance(valid_features[0], dict):
                        combined_features = {
                            'input_ids': torch.cat([f['input_ids'] for f in valid_features]).to(self.device),
                            'attention_mask': torch.cat([f['attention_mask'] for f in valid_features]).to(self.device)
                        }
                        encoded_features = self.encode_text(combined_features)
                    else:
                        stacked_features = torch.stack(valid_features).to(self.device)
                        encoded_features = self.encode_text(stacked_features)
    
                    for j, idx in enumerate(valid_indices):
                        all_features[idx] = encoded_features[j]
                        valid_mask[idx] = True
    
            # Process image entities
            for i in range(0, len(image_indices), batch_size):
                batch_idx = image_indices[i:i + batch_size]
                batch_entities = [kg_dataset.idx_to_entity[idx] for idx in batch_idx]
                batch_features = kg_dataset._process_image_batch(batch_entities)
    
                valid_features = []
                valid_indices = []
                for idx, feat in zip(batch_idx, batch_features):
                    if feat is not None:
                        valid_features.append(feat)
                        valid_indices.append(idx)
    
                if valid_features:
                    stacked_features = torch.stack(valid_features).to(self.device)
                    encoded_features = self.encode_image(stacked_features)
    
                    for j, idx in enumerate(valid_indices):
                        all_features[idx] = encoded_features[j]
                        valid_mask[idx] = True
    
            # Handle missing features
            mean_features = all_features[valid_mask].mean(0)
            all_features[~valid_mask] = mean_features
    
            # Save to cache
            torch.save(all_features, cache_path)
            print(f"Saved entity features to cache: {cache_path}")
    
        self.train()
        return all_features

    def encode_image(self, image):
        image = image.to(self.device)
        image_features = self.clip_model.encode_image(image.squeeze(1))
        return image_features
        # return self.image_projection(image_features)
    
    def encode_text(self, text_tokens):
        if isinstance(text_tokens, dict):
            text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}
            text_features = self.clip_model.encode_text(text_tokens)
        else:
            text_tokens = text_tokens.squeeze(1).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
        return text_features
        # return self.text_projection(text_features)

    def forward(self, batch, mode='finetune'):
        if mode == 'finetune':
            return self.forward_finetune(batch)
        elif mode == 'pretrain':
            return self.forward_pretrain(batch)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward_finetune(self, batch):
        images = batch['image'].to(self.device)
        if isinstance(batch['text_tokens'], dict):
            text_tokens = {k: v.to(self.device) for k, v in batch['text_tokens'].items()}
        else:
            text_tokens = batch['text_tokens'].squeeze(1).to(self.device)

        image_features = self.encode_image(images)
        text_features = self.encode_text(text_tokens)
        
        return self._compute_contrastive_loss(image_features, text_features)

    def forward_pretrain(self, batch):
        # Image head / text tail
        head_features = self.encode_image(batch['head'].to(self.device))
        if isinstance(batch['tail'], dict):
            tail_features = self.encode_text(batch['tail'])
        else:
            tail_features = self.encode_text(batch['tail'].squeeze(1).to(self.device))
        if isinstance(batch['relation_text'], dict):
            relation_features = self.encode_text(batch['relation_text'])
        else:
            relation_features = self.encode_text(batch['relation_text'].squeeze(1).to(self.device))
#        print(head_features.shape)
#        print(tail_features.shape)
#        print(batch['relation_text'].shape)
        
#        relation_features = self.encode_text(batch['relation_text'].squeeze(1).to(self.device)) if not isinstance(batch['relation_text'], dict) else self.encode_text(batch['relation_text'])
        
        seq_features = torch.cat([head_features.unsqueeze(1), relation_features.unsqueeze(1)], dim=1)
        fused_features = self.multimodal_encoder(seq_features)
        head_rel_features = fused_features[:, 0]
        
        loss_e2e = self._compute_contrastive_loss(head_rel_features, tail_features)
        
        entity_pairs = torch.cat([head_features, tail_features], dim=1)
        relation_logits = self.relation_classifier(entity_pairs)
        loss_e2r = F.cross_entropy(relation_logits, batch['relation_label'].to(self.device))

#        # Sample GNN embeddings
#        gnn_embeddings = self.get_gnn_embeddings(batch['head_idx'], batch['tail_idx'])
#        gnn_head, gnn_tail = gnn_embeddings
#        
#        loss_g2e = (self._compute_contrastive_loss(gnn_head, head_features) +
#                    self._compute_contrastive_loss(gnn_tail, tail_features)) / 2

        return loss_e2e + loss_e2r#  + loss_g2e

    def get_gnn_embeddings(self, head_indices, tail_indices):
        # Move indices to CPU before passing into NeighborLoader
        input_nodes = torch.cat([head_indices, tail_indices]).unique().cpu()
    
        subgraph_batch = self.neighbor_sampler(input_nodes)  # No error now!
    
        # Move subgraph to device after sampling
        subgraph_batch = subgraph_batch.to(self.device)
    
        # Run GNN
        x = subgraph_batch.x
        edge_index = subgraph_batch.edge_index
        gnn_output = self.gnn(x, edge_index)
    
        # Re-map to requested indices
        id_mapping = {old.item(): new_idx for new_idx, old in enumerate(subgraph_batch.batch)}
        gnn_head = torch.stack([gnn_output[id_mapping[i.item()]] for i in head_indices])
        gnn_tail = torch.stack([gnn_output[id_mapping[i.item()]] for i in tail_indices])
        return gnn_head, gnn_tail

    def _compute_contrastive_loss(self, features_a, features_b, temperature=0.07):
        features_a = F.normalize(features_a, dim=-1)
        features_b = F.normalize(features_b, dim=-1)
        
        similarity = torch.matmul(features_a, features_b.T) / temperature
        labels = torch.arange(len(features_a)).to(features_a.device)
        
        loss = F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.T, labels)
        return loss / 2

