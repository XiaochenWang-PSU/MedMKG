import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
import json
from bs4 import BeautifulSoup
import pydicom
import csv
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch.utils.data as data
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from torch_geometric.data import Data
import torch.nn.functional as F
import random



class OpenIDataset(Dataset):
   def __init__(self, text_path, image_path, clip_processor, clip_tokenizer):
       self.processor = clip_processor
       self.tokenizer = clip_tokenizer
       self.samples = []

       for text_file in tqdm(os.listdir(text_path)):
           text_path_patient_xml = os.path.join(text_path, text_file)
           with open(text_path_patient_xml, 'r', encoding='utf-8') as f:
               xml = f.read()
           soup = BeautifulSoup(xml, 'xml')
           
           report = 'COMPARIOSN: ' + soup.find('AbstractText', Label='COMPARISON').get_text() + '\n' + \
                    'INDICATION: ' + soup.find('AbstractText', Label='INDICATION').get_text() + '\n' + \
                    'FINDINGS: ' + soup.find('AbstractText', Label='FINDINGS').get_text() + '\n' + \
                    'IMPRESSION: ' + soup.find('AbstractText', Label='IMPRESSION').get_text() + '\n'
           
           # Store individual image paths instead of concatenating
           for image_id in soup.find_all('parentImage'):
               image_path_single = os.path.join(image_path, image_id['id'] + '.png')
               if os.path.exists(image_path_single):
                   self.samples.append((image_path_single, report))

   def __getitem__(self, index):
       image_path, report = self.samples[index]
       if self.tokenizer:
           try:
               image = Image.open(image_path).convert('RGB')
               processed_img = self.processor(image)
           except Exception as e:
               print(f"Error loading image {image_path}: {e}")
               raise
               
           tokenized_text = self.tokenizer(report)
           
           return {
               'image': processed_img,
               'text': report,
               'text_tokens': tokenized_text
           }
       else:
           try:
               image = Image.open(image_path).convert('RGB')
               processed_img = self.processor(
                images=image, 
                return_tensors="pt"
                )
               tokenized_text = self.processor(
                text=report, 
                images=None, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=77
                )
           except Exception as e:
               print(f"Error loading image {image_path}: {e}")
               raise
               
           return {
               'image': processed_img,
               'text': report,
               'text_tokens': tokenized_text
           }

   def __len__(self):
       return len(self.samples)
class MIMICDataset(Dataset):
    def __init__(self, image_base_path, text_base_path, metadata_csv_path, 
                 clip_processor, clip_tokenizer, kg_image_path_file, sample_size=10000, seed=42):
        """
        Args:
            image_base_path: Base path to MIMIC-CXR images (".../mimic-cxr-jpg/2.0.0/files/")
            text_base_path: Base path to MIMIC-CXR reports (".../mimic-cxr/2.0.0/files/")
            metadata_csv_path: Path to mimic-cxr-2.0.0-metadata.csv
            clip_processor: CLIP image processor
            clip_tokenizer: CLIP text tokenizer
            kg_image_path_file: Path to knowledge graph image mapping CSV
            sample_size: Number of samples to include
            seed: Random seed for sampling
        """
        self.processor = clip_processor
        self.tokenizer = clip_tokenizer
        
        # Load KG image paths to exclude
        print("Loading knowledge graph image paths...")
        kg_images_df = pd.read_csv(kg_image_path_file)
        self.kg_image_paths = set(kg_images_df['Image_Path'].values)
        
        # Load metadata
        print("Loading MIMIC-CXR metadata...")
        metadata_df = pd.read_csv(metadata_csv_path)
        
        # Build samples list
        print("Building dataset...")
        self.samples = []
        
        # Group by study_id to get all images for each study
        for _, group in tqdm(metadata_df.groupby(['subject_id', 'study_id'])):
            # Get patient and study info
            subject_id = str(group['subject_id'].iloc[0])
            study_id = str(group['study_id'].iloc[0])
            
            # Construct paths
            p_folder = f'p{subject_id[:2]}'
            p_id = f'p{subject_id}'
            s_id = f's{study_id}'
            
            # Construct text path
            text_path = os.path.join(text_base_path, p_folder, p_id, f"{s_id}.txt")
            
            # Only proceed if text file exists
            if not os.path.exists(text_path):
                continue
            
            # Process each image in the study
            for _, row in group.iterrows():
                dicom_id = row['dicom_id']
                
                # Construct image path
                img_path = os.path.join(image_base_path, p_folder, p_id, s_id, f"{dicom_id}.jpg")
                
                # Skip if image is in knowledge graph or doesn't exist
                if img_path in self.kg_image_paths or not os.path.exists(img_path):
                    continue
                    
                self.samples.append((img_path, text_path))
        
        print(f"Found {len(self.samples)} total samples")
        
        # Subsample if needed
        if sample_size and sample_size < len(self.samples):
            random.seed(seed)
            self.samples = random.sample(self.samples, sample_size)
            print(f"Subsampled to {sample_size} samples")
    
    def __getitem__(self, index):
        image_path, text_path = self.samples[index]
        
        # Read and format report
        try:
            with open(text_path, 'r') as f:
                report = f.read()
        except Exception as e:
            print(f"Error reading text file {text_path}: {e}")
            raise
        
        if self.tokenizer:  # Regular CLIP
            try:
                image = Image.open(image_path).convert('RGB')
                processed_img = self.processor(image)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                raise
                
            tokenized_text = self.tokenizer(report)
            
            return {
                'image': processed_img,
                'text': report,
                'text_tokens': tokenized_text
            }
        else:  # Biomedical CLIP
            try:
                image = Image.open(image_path).convert('RGB')
                processed_img = self.processor(
                    images=image, 
                    return_tensors="pt"
                )
                tokenized_text = self.processor(
                    text=report, 
                    images=None, 
                    return_tensors="pt", 
                    padding=True,
                    truncation=True,
                    max_length=77
                )
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                raise
                
            return {
                'image': processed_img,
                'text': report,
                'text_tokens': tokenized_text
            }
    
    def __len__(self):
        return len(self.samples)
def collate_fn_normal(batch):
    return {
        
        'image': torch.stack([item['image'] for item in batch]),
        'text': [item['text'] for item in batch],
        'text_tokens': torch.stack([item['text_tokens'] for item in batch])
    }
def collate_fn_pub(batch):
    images = torch.stack([item['image']['pixel_values'] for item in batch])
    texts = [item['text'] for item in batch]

    # Handling variable length of input_ids
    input_ids = [item['text_tokens']['input_ids'].squeeze(0) for item in batch]
    attention_masks = [item['text_tokens']['attention_mask'].squeeze(0) for item in batch]

    # Padding text_tokens here ensures uniformity
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)

    return {
        'image': images,
        'text': texts,
        'text_tokens': {
            'input_ids': input_ids_padded,
            'attention_mask': attention_masks_padded
        }
    }



def kg_collate_fn(batch):
    """Custom collate function for KG data
    Args:
        batch: List containing single PyG Data object
    Returns:
        The first (and only) item in the batch
    """
    # Since we're working with a single graph, just return the first item
    return batch[0]
    
def subsample_knowledge_graph(kg_df: pd.DataFrame, sample_size: int, seed: int = 42) -> pd.DataFrame:
    """
    Randomly sample rows from the knowledge graph DataFrame
    Args:
        kg_df: Original knowledge graph DataFrame
        sample_size: Number of rows to sample
        seed: Random seed for reproducibility
    Returns:
        Subsampled DataFrame
    """
    # Make sure sample_size isn't larger than the DataFrame
    sample_size = min(sample_size, len(kg_df))
    
    # Sample rows randomly
    return kg_df.sample(n=sample_size, random_state=seed)

class KGDataset(Dataset):
    def __init__(self, kg_path: str, image_map_path: str, 
                 processor=None, tokenizer=None, is_biomed: bool = False,
                 sample_size: Optional[int] = None, seed: int = 42):
        """Initialize Knowledge Graph dataset
        Args:
            kg_path: Path to knowledge graph CSV
            image_map_path: Path to image mapping CSV
            processor: Image/text processor
            tokenizer: Text tokenizer
            is_biomed: Whether using biomedical CLIP
            sample_size: If provided, subsample the knowledge graph to this size
            seed: Random seed for sampling
        """
        # Load KG data
        self.kg_df = pd.read_csv(kg_path)
        
        # Apply initial filtering
        self.kg_df = self.kg_df[self.kg_df['Head'].str.startswith('I')]
        self.kg_df['Tail'] = self.kg_df['Tail_Name']
        # print(self.kg_df['Tail'])
        
        # Subsample if requested
        if sample_size is not None:
            print(f"Subsampling knowledge graph to {sample_size} edges...")
            orig_size = len(self.kg_df)
            self.kg_df = subsample_knowledge_graph(self.kg_df, sample_size, seed)
            print(f"Sampled {len(self.kg_df)} edges from {orig_size} edges")
        
        self.processor = processor
        self.is_biomed = is_biomed
        self.tokenizer = tokenizer
        
        # Load image mapping
        image_map_df = pd.read_csv(image_map_path)
        
        # Only keep relevant images after sampling
        if sample_size is not None:
            relevant_images = set(self.kg_df['Head']) | set(self.kg_df['Tail'])
            image_map_df = image_map_df[image_map_df['IID'].isin(relevant_images)]
        
        self.image_id_to_path = dict(zip(image_map_df['IID'], image_map_df['Image_Path']))
        
        # Create entity vocabulary
        all_entities = set(self.kg_df['Head'].unique()) | set(self.kg_df['Tail'].unique())
        self.entity_to_idx = {ent: idx for idx, ent in enumerate(sorted(all_entities))}
        self.idx_to_entity = {idx: ent for ent, idx in self.entity_to_idx.items()}
        
        # Create relation vocabulary
        all_relations = set(self.kg_df['Relation'].unique())
        self.relation_to_idx = {rel: idx for idx, rel in enumerate(sorted(all_relations))}
        self.idx_to_relation = {idx: rel for rel, idx in self.relation_to_idx.items()}
        
#        # Create adjacency matrix
        self.adj_matrix = self._create_adj_matrix()
        
        # Filter out invalid entries
        self.valid_indices = self._get_valid_indices()
#        self.edge_index = self._create_edge_index()
        print(f"Dataset initialized with {len(self.valid_indices)} valid samples")
        print(f"Number of unique entities: {len(self.entity_to_idx)}")
        print(f"Number of unique relations: {len(self.relation_to_idx)}")
    def _create_edge_index(self) -> torch.Tensor:
        """Create and cache edge_index tensor [2, num_edges] from KG."""
        edge_list = []
        for _, row in self.kg_df.iterrows():
            head_idx = self.entity_to_idx[row['Head']]
            tail_idx = self.entity_to_idx[row['Tail']]
            # Undirected graph: add both directions
            edge_list.append((head_idx, tail_idx))
            edge_list.append((tail_idx, head_idx))
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # shape [2, num_edges]
        return edge_index

    def get_edge_index(self) -> torch.Tensor:
        """Return cached edge_index tensor."""
        return self.edge_index
    def _create_adj_matrix(self) -> torch.Tensor:
        """Create normalized adjacency matrix for the knowledge graph"""
        num_entities = len(self.entity_to_idx)
        adj = torch.zeros((num_entities, num_entities))
        
        # Add edges from knowledge graph
        for _, row in self.kg_df.iterrows():
            head_idx = self.entity_to_idx[row['Head']]
            tail_idx = self.entity_to_idx[row['Tail']]
            adj[head_idx, tail_idx] = 1
            adj[tail_idx, head_idx] = 1  # Make undirected
            
        # Add self-loops
        adj = adj + torch.eye(num_entities)
        
        # Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_inv_sqrt[torch.isnan(deg_inv_sqrt)] = 0
        
        normalized_adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(0)
        
        return normalized_adj
    
    def _process_image(self, image_id: str) -> Optional[torch.Tensor]:
        """Process image with safety checks"""
        try:
            # Check if image path exists
            if image_id not in self.image_id_to_path:
                return None
                
            # Set PIL Image size limit
            Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb check
            
            # Load and convert image
            image_path = self.image_id_to_path[image_id]
            if not Path(image_path).exists():
                return None
                
            img = Image.open(image_path).convert('RGB')
            

            
            # Process based on model type
            if self.is_biomed:
                processed = self.processor(
                    images=img,
                    return_tensors="pt",
                    max_length=512
                )['pixel_values'].squeeze(0)
            else:
                processed = self.processor(img)
                
            return processed
            
        except Exception as e:
            print(f"Warning: Failed to process image {image_id}: {str(e)}")
            return None
            
    def _process_text(self, text: str) -> Optional[torch.Tensor]:
        """Process text input"""
        try:
            if self.is_biomed:
                processed = self.processor(
                    text=text,
                    return_tensors="pt",
                    padding=True,
                truncation=True,
                max_length=77
                )
            else:
                processed = self.tokenizer(text)
            return processed
            
        except Exception as e:
            print(f"Warning: Failed to process text: {str(e)}")
            return None


    def _process_image_batch(self, image_ids: List[str]) -> List[Optional[torch.Tensor]]:
        """Process a batch of images efficiently"""
        processed_features = [None] * len(image_ids)
        valid_images = []
        valid_indices = []
        
        # Prepare all valid image paths first
        image_paths = [(i, self.image_id_to_path[img_id]) 
                      for i, img_id in enumerate(image_ids) 
                      if img_id in self.image_id_to_path]
        
        if not image_paths:
            return processed_features
        
        # Load images in parallel using ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor
        
        def load_image(args):
            idx, path = args
            try:
                return idx, Image.open(path).convert('RGB')
            except Exception as e:
                print(f"Failed to load image {path}: {e}")
                return idx, None
        
        # Load images in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(load_image, image_paths))
        
        # Process valid images
        valid_images = []
        valid_indices = []
        for idx, img in results:
            if img is not None:
                valid_images.append(img)
                valid_indices.append(idx)
        
        if not valid_images:
            return processed_features
        
        try:
            # Process all valid images at once
            if self.is_biomed:
                batch_processed = self.processor(
                    images=valid_images,
                    return_tensors="pt",
                    max_length=512
                )['pixel_values']
            else:
                # Process in sub-batches to avoid memory issues
                sub_batch_size = 32
                all_processed = []
                for i in range(0, len(valid_images), sub_batch_size):
                    sub_batch = valid_images[i:i + sub_batch_size]
                    sub_processed = torch.stack([
                        self.processor(img) for img in sub_batch
                    ])
                    all_processed.append(sub_processed)
                batch_processed = torch.cat(all_processed)
            
            # Assign processed features
            for orig_idx, features in zip(valid_indices, batch_processed):
                processed_features[orig_idx] = features
                
        except Exception as e:
            print(f"Batch image processing failed: {e}")
        
        return processed_features



    def _process_text_batch(self, texts: List[str]) -> List[Optional[torch.Tensor]]:
        """Process a batch of texts
        Args:
            texts: List of text strings to process
        Returns:
            List of processed text tensors (None for failed processes)
        """
        try:
            if self.is_biomed:
                # Process all texts in batch
                processed = self.processor(
                    text=texts,
                    return_tensors="pt",
                    max_length=77,
                    padding=True,
                    truncation=True
                )
                
                # Split into individual tensors
                return [
                    {
                        'input_ids': processed['input_ids'][i].unsqueeze(0),
                        'attention_mask': processed['attention_mask'][i].unsqueeze(0)
                    }
                    for i in range(len(texts))
                ]
            else:
                processed = []
                for text in texts:
                    try:
                        text_tensor = self.tokenizer(text)
                        if isinstance(text_tensor, torch.Tensor):
                            # Ensure tensor is in the expected shape, say [1, 77]
                            
                            processed.append(text_tensor)
                        else:
                            processed.append(None)
                    except Exception as e:
                        print(f"Failed to process text '{text}': {e}")
                        processed.append(None)
                        
                return processed
                
        except Exception as e:
            print(f"Batch text processing failed: {e}")
            return [None] * len(texts)
    def _get_valid_indices(self) -> List[int]:
        """Pre-check which indices are valid"""
        valid_indices = []
        
        for idx in range(len(self.kg_df)):
            row = self.kg_df.iloc[idx]
            
            # Check if entities exist in mappings
            if row['Head'] not in self.entity_to_idx or row['Tail'] not in self.entity_to_idx:
                continue
                
            # Check if relation exists in mappings
            if row['Relation'] not in self.relation_to_idx:
                continue
                
            # Check if images exist for image entities
            head_is_image = str(row['Head']).startswith('I')
            tail_is_image = str(row['Tail']).startswith('I')

            is_valid = True
            
            if head_is_image and row['Head'] not in self.image_id_to_path:
                is_valid = False
            if tail_is_image and row['Tail'] not in self.image_id_to_path:
                is_valid = False
                
            if is_valid:
                valid_indices.append(idx)
                
        if not valid_indices:
            raise ValueError("No valid samples found in the dataset")
            
        return valid_indices
    
    def __len__(self) -> int:
        return len(self.valid_indices)
        
    def __getitem__(self, idx: int) -> Dict:
        """Get a single triplet"""
        # Get the actual index from valid indices
        actual_idx = self.valid_indices[idx]
        row = self.kg_df.iloc[actual_idx]
        
        # Get entity indices
        head_idx = self.entity_to_idx[row['Head']]
        tail_idx = self.entity_to_idx[row['Tail']]
        
        # Process entities
        head_is_image = str(row['Head']).startswith('I')
        tail_is_image = str(row['Tail']).startswith('I')
        
        # Process head
        if head_is_image:
            head_data = self._process_image(row['Head'])
            if head_data is None:
                print('without image')
                head_data = torch.zeros((3, 224, 224))
            head_type = "image"
        else:
            head_data = self._process_text(row['Head_Name'])
            head_type = "text"
            
        # Process tail
        if tail_is_image:
            tail_data = self._process_image(row['Tail'])
            if tail_data is None:
                tail_data = torch.zeros((3, 224, 224))
            tail_type = "image"
        else:
            tail_data = self._process_text(row['Tail_Name'])
            tail_type = "text"
            
        # Create one-hot relation
        relation_idx = self.relation_to_idx[row['Relation']]
        relation_label = torch.tensor(relation_idx, dtype=torch.long)  # For classification
        relation_text = self._process_text(row['Relation'])  # For multimodal fusion

        return {
        'head': head_data,
        'head_type': head_type,
        'tail': tail_data,
        'tail_type': tail_type,
        'relation_text': relation_text,  # For fusion
        'relation_label': relation_label,  # For classification
        'head_idx': head_idx,
        'tail_idx': tail_idx,
        'text': row['Head'] if not head_is_image else row['Tail_Name']
        }
        
    def get_adj_matrix(self) -> torch.Tensor:
        return self.adj_matrix
        
    def get_num_entities(self) -> int:
        return len(self.entity_to_idx)
        
    def get_num_relations(self) -> int:
        return len(self.relation_to_idx)

def collate_kg_batch(batch):
    """Custom collate function for KG batches with biomedical CLIP outputs"""
    # Initialize containers
    batch_dict = {
        'head': [],
        'head_type': [],
        'tail_type': [],
        'head_idx': [],
        'tail_idx': [],
        'relation_label': [],
        'text': [],
        'tail_input_ids': [],
        'tail_attention_mask': [],
        'relation_input_ids': [],
        'relation_attention_mask': []
    }
    
    max_tail_len = max(sample['tail']['input_ids'].size(1) for sample in batch)
    max_rel_len = max(sample['relation_text']['input_ids'].size(1) for sample in batch)
    
    for sample in batch:
        # Handle head images
        batch_dict['head'].append(sample['head'])
        
        # Handle tail text with padding
        tail_len = sample['tail']['input_ids'].size(1)
        if tail_len < max_tail_len:
            pad_len = max_tail_len - tail_len
            tail_ids = F.pad(sample['tail']['input_ids'], (0, pad_len), value=0)
            tail_mask = F.pad(sample['tail']['attention_mask'], (0, pad_len), value=0)
        else:
            tail_ids = sample['tail']['input_ids']
            tail_mask = sample['tail']['attention_mask']
        batch_dict['tail_input_ids'].append(tail_ids)
        batch_dict['tail_attention_mask'].append(tail_mask)
        
        # Handle relation text with padding
        rel_len = sample['relation_text']['input_ids'].size(1)
        if rel_len < max_rel_len:
            pad_len = max_rel_len - rel_len
            rel_ids = F.pad(sample['relation_text']['input_ids'], (0, pad_len), value=0)
            rel_mask = F.pad(sample['relation_text']['attention_mask'], (0, pad_len), value=0)
        else:
            rel_ids = sample['relation_text']['input_ids']
            rel_mask = sample['relation_text']['attention_mask']
        batch_dict['relation_input_ids'].append(rel_ids)
        batch_dict['relation_attention_mask'].append(rel_mask)
        
        # Collect other fields
        batch_dict['head_type'].append(sample['head_type'])
        batch_dict['tail_type'].append(sample['tail_type'])
        batch_dict['head_idx'].append(sample['head_idx'])
        batch_dict['tail_idx'].append(sample['tail_idx'])
        batch_dict['relation_label'].append(sample['relation_label'])
        batch_dict['text'].append(sample['text'])
    
    # Stack all tensors
    batch_dict['head'] = torch.stack(batch_dict['head'])
    batch_dict['head_idx'] = torch.tensor(batch_dict['head_idx'])
    batch_dict['tail_idx'] = torch.tensor(batch_dict['tail_idx'])
    batch_dict['relation_label'] = torch.stack(batch_dict['relation_label'])
    
    # Stack and reshape text inputs
    tail_input_ids = torch.cat(batch_dict.pop('tail_input_ids'), dim=0)
    tail_attention_mask = torch.cat(batch_dict.pop('tail_attention_mask'), dim=0)
    rel_input_ids = torch.cat(batch_dict.pop('relation_input_ids'), dim=0)
    rel_attention_mask = torch.cat(batch_dict.pop('relation_attention_mask'), dim=0)
    
    batch_dict['tail'] = {
        'input_ids': tail_input_ids,
        'attention_mask': tail_attention_mask
    }
    batch_dict['relation_text'] = {
        'input_ids': rel_input_ids,
        'attention_mask': rel_attention_mask
    }
    
    return batch_dict