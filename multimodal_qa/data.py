import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm
import pickle
from model import KGRetriever  # Ensure this import path matches your project structure




class MedicalVQADataset(Dataset):
    def __init__(self, 
                 dataset_name: str,
                 split: str,
                 processor=None,
                 tokenizer=None,
                 max_length: int = 77,
                 val_split: float = 0.1,
                 kg_retriever=None,
                 cache_path: str = None):

        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.kg_retriever = kg_retriever
        self.cache_path = cache_path or f"cache/{dataset_name}_{split}_graph_cache.pkl"

        if dataset_name.lower() == 'slake':
            self._load_slake(split)
        elif dataset_name.lower() == 'vqa_rad':
            self._load_vqa_rad(split, val_split)
        elif dataset_name.lower() == 'pathvqa':
            self._load_pathvqa(split)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                graph_data = pickle.load(f)
                for i, sample in enumerate(self.samples):
                    sample['subgraph'] = graph_data[i]['subgraph']
                    sample['subgraph_edges'] = graph_data[i]['subgraph_edges']
                    sample['subgraph_relations'] = graph_data[i]['subgraph_relations']  # NEW
        else:
            print("Precomputing subgraphs and edge indices with 1-hop expansion...")
            full_edge_index = torch.tensor(self.kg_retriever.build_edge_index(), dtype=torch.long).t()  # [2, E]
            full_edge_type = torch.tensor(self.kg_retriever.build_edge_types(), dtype=torch.long)       # [E]   new requirement
            graph_data = []
        
            for sample in tqdm(self.samples):
                concept_ids = self.kg_retriever.get_concepts_from_text(sample['question'])
                initial_indices = [self.kg_retriever.id_to_idx[cid] for cid in concept_ids if cid in self.kg_retriever.id_to_idx]
        
                one_hop_indices = set(initial_indices)
                for idx in initial_indices:
                    for i in range(full_edge_index.size(1)):
                        src, tgt = full_edge_index[:, i].tolist()
                        if src == idx or tgt == idx:
                            one_hop_indices.add(src)
                            one_hop_indices.add(tgt)
        
                one_hop_indices = sorted(list(one_hop_indices))
                index_map = {old: new for new, old in enumerate(one_hop_indices)}
        
                sub_edges = []
                sub_relations = []
                for i in range(full_edge_index.size(1)):
                    src, tgt = full_edge_index[:, i].tolist()
                    if src in index_map and tgt in index_map:
                        sub_edges.append([index_map[src], index_map[tgt]])
                        sub_relations.append(full_edge_type[i].item())
        
                graph_data.append({
                    'subgraph': one_hop_indices,
                    'subgraph_edges': sub_edges,
                    'subgraph_relations': sub_relations  # NEW
                })
                sample['subgraph'] = one_hop_indices
                sample['subgraph_edges'] = sub_edges
                sample['subgraph_relations'] = sub_relations  # NEW
        
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(graph_data, f)

#        if os.path.exists(self.cache_path):
#            with open(self.cache_path, 'rb') as f:
#                graph_data = pickle.load(f)
#                for i, sample in enumerate(self.samples):
#                    sample['subgraph'] = graph_data[i]['subgraph']
#                    sample['subgraph_edges'] = graph_data[i]['subgraph_edges']
#        else:
#            print("Precomputing subgraphs and edge indices with 1-hop expansion...")
#            full_edge_index = torch.tensor(self.kg_retriever.build_edge_index(), dtype=torch.long).t()  # [2, E]
#            graph_data = []
#            for sample in tqdm(self.samples):
#                concept_ids = self.kg_retriever.get_concepts_from_text(sample['question'])
#                initial_indices = [self.kg_retriever.id_to_idx[cid] for cid in concept_ids if cid in self.kg_retriever.id_to_idx]
#                one_hop_indices = set(initial_indices)
#                for idx in initial_indices:
#                    for i in range(full_edge_index.size(1)):
#                        src, tgt = full_edge_index[:, i].tolist()
#                        if src == idx:
#                            one_hop_indices.add(tgt)
#                        elif tgt == idx:
#                            one_hop_indices.add(src)
#                one_hop_indices = sorted(list(one_hop_indices))
#                index_map = {old: new for new, old in enumerate(one_hop_indices)}
#                sub_edges = []
#                for i in range(full_edge_index.size(1)):
#                    src, tgt = full_edge_index[:, i].tolist()
#                    if src in index_map and tgt in index_map:
#                        sub_edges.append([index_map[src], index_map[tgt]])
#                graph_data.append({
#                    'subgraph': one_hop_indices,
#                    'subgraph_edges': sub_edges
#                })
#                sample['subgraph'] = one_hop_indices
#                sample['subgraph_edges'] = sub_edges
#
#            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
#            with open(self.cache_path, 'wb') as f:
#                pickle.dump(graph_data, f)

        print(f"Loaded {len(self.samples)} samples from {dataset_name} for {split} split")

    def _process_answer(self, answer: str):
        answer = str(answer).lower().strip()
        if answer in ['yes', 'true']:
            return 1
        elif answer in ['no', 'false']:
            return 0
        return None

    def _load_slake(self, split: str):
        base_path = YOUR_SLAKE_PATH
        split_files = {'train': 'train.json', 'val': 'validate.json', 'test': 'test.json'}
        json_path = os.path.join(base_path, split_files[split])
        with open(json_path, 'r') as f:
            data = json.load(f)
        for item in data:
            if item.get("q_lang") == "en":
                binary_answer = self._process_answer(item['answer'])
                if binary_answer is not None:
                    image_path = os.path.join(base_path, 'imgs', item['img_name'])
                    if os.path.exists(image_path):
                        self.samples.append({
                            'image_path': image_path,
                            'question': item['question'],
                            'answer': binary_answer,
                            'answer_text': str(item['answer']).lower().strip(),
                            'dataset': 'slake'
                        })

    def _load_vqa_rad(self, split: str, val_split: float):
        base_path = YOUR_RAD_PATH
        json_path = os.path.join(base_path, 'trainset.json' if split != 'test' else 'testset.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
        yes_no_samples = []
        for item in data:
            binary_answer = self._process_answer(item['answer'])
            if binary_answer is not None:
                image_path = os.path.join(base_path, 'images', item['image_name'])
                if os.path.exists(image_path):
                    yes_no_samples.append({
                        'image_path': image_path,
                        'question': item['question'],
                        'answer': binary_answer,
                        'answer_text': str(item['answer']).lower().strip(),
                        'dataset': 'vqa_rad'
                    })
        if split != 'test':
            random.seed(42)
            split_idx = int(len(yes_no_samples) * (1 - val_split))
            random.shuffle(yes_no_samples)
            self.samples.extend(yes_no_samples[:split_idx] if split == 'train' else yes_no_samples[split_idx:])
        else:
            self.samples.extend(yes_no_samples)

    def _load_pathvqa(self, split: str):
        hf_split = {'train': 'train', 'val': 'validation', 'test': 'test'}[split]
        dataset = load_dataset("flaviagiammarino/path-vqa")[hf_split]
        for item in dataset:
            binary_answer = self._process_answer(item['answer'])
            if binary_answer is not None:
                image = item['image']
                if image.mode == 'CMYK':
                    image = image.convert('RGB')
                self.samples.append({
                    'image': image,
                    'question': item['question'],
                    'answer': binary_answer,
                    'answer_text': str(item['answer']).lower().strip(),
                    'dataset': 'pathvqa'
                })
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
    
        # Load image
        if 'image_path' in sample:
            image = Image.open(sample['image_path']).convert('RGB')
        else:
            image = sample['image']
    
        # With HuggingFace-style tokenizer
        if self.tokenizer:
            processed_img = self.processor(image)
            question_tokens = self.tokenizer(sample['question'])
    
            return {
                'image': processed_img.squeeze(0),
                'question': sample['question'],
                'question_tokens': question_tokens,
                'answer': sample['answer'],
                'answer_text': sample['answer_text'],
                'dataset': sample['dataset'],
                'subgraph': sample['subgraph'],
                'subgraph_edges': sample['subgraph_edges'],
                'subgraph_relations': sample['subgraph_relations']  # <-- added
            }
    
        # With vision-language processor (e.g., CLIPProcessor)
        else:
            processed_img = self.processor(images=image, return_tensors="pt")['pixel_values']
            question_tokens = self.processor.tokenizer(
                sample['question'],
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors=None
            )
            input_ids = torch.tensor(question_tokens['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(question_tokens['attention_mask'], dtype=torch.long)
    
            return {
                'image': processed_img.squeeze(0),
                'question': sample['question'],
                'question_tokens': {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                },
                'answer': sample['answer'],
                'answer_text': sample['answer_text'],
                'dataset': sample['dataset'],
                'subgraph': sample['subgraph'],
                'subgraph_edges': sample['subgraph_edges'],
                'subgraph_relations': sample['subgraph_relations']  # <-- added
            }

#    def __getitem__(self, idx: int) -> Dict:
#        sample = self.samples[idx]
#        if 'image_path' in sample:
#            image = Image.open(sample['image_path']).convert('RGB')
#        else:
#            image = sample['image']
#        if self.tokenizer:
#            processed_img = self.processor(image)
#            question_tokens = self.tokenizer(sample['question'])
#            return {
#                'image': processed_img.squeeze(0),
#                'question': sample['question'],
#                'question_tokens': question_tokens,
#                'answer': sample['answer'],
#                'answer_text': sample['answer_text'],
#                'dataset': sample['dataset'],
#                'subgraph': sample['subgraph'],
#                'subgraph_edges': sample['subgraph_edges']
#            }
#        else:
#            processed_img = self.processor(images=image, return_tensors="pt")['pixel_values']
#            question_tokens = self.processor.tokenizer(
#                sample['question'],
#                padding='max_length',
#                truncation=True,
#                max_length=self.max_length,
#                return_tensors=None
#            )
#            input_ids = torch.tensor(question_tokens['input_ids'], dtype=torch.long)
#            attention_mask = torch.tensor(question_tokens['attention_mask'], dtype=torch.long)
#            return {
#                'image': processed_img.squeeze(0),
#                'question': sample['question'],
#                'question_tokens': {
#                    'input_ids': input_ids,
#                    'attention_mask': attention_mask
#                },
#                'answer': sample['answer'],
#                'answer_text': sample['answer_text'],
#                'dataset': sample['dataset'],
#                'subgraph': sample['subgraph'],
#                'subgraph_edges': sample['subgraph_edges']
#            }

    def __len__(self):
        return len(self.samples)

    def get_statistics(self) -> Dict:
        return {
            'total_samples': len(self.samples),
            'answer_distribution': Counter(s['answer'] for s in self.samples),
            'answer_text_distribution': Counter(s['answer_text'] for s in self.samples)
        }



def collate_fn_med_vqa(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    if type(batch[0]['question_tokens']) == dict:
        return {
            'question': [item['question'] for item in batch],
            'answer': torch.tensor([item['answer'] for item in batch], dtype=torch.long),
            'answer_text': [item['answer_text'] for item in batch],
            'dataset': [item['dataset'] for item in batch],
            'image': torch.stack([item['image'] for item in batch]),
            'subgraph': [item['subgraph'] for item in batch],
            'subgraph_edges': [item['subgraph_edges'] for item in batch],  # ? ADD THIS
            'subgraph_relations': [item['subgraph_relations'] for item in batch],  # ? ADD THIS
            'question_tokens': {
                'input_ids': torch.stack([item['question_tokens']['input_ids'] for item in batch]),
                'attention_mask': torch.stack([item['question_tokens']['attention_mask'] for item in batch])
            }
        }
    else:
        return {
            'question': [item['question'] for item in batch],
            'answer': torch.tensor([item['answer'] for item in batch], dtype=torch.long),
            'answer_text': [item['answer_text'] for item in batch],
            'dataset': [item['dataset'] for item in batch],
            'image': torch.stack([item['image'] for item in batch]),
            'subgraph': [item['subgraph'] for item in batch],
            'subgraph_edges': [item['subgraph_edges'] for item in batch],  # ? ADD THIS
            'subgraph_relations': [item['subgraph_relations'] for item in batch],  # ? ADD THIS
            'question_tokens': torch.stack([item['question_tokens'] for item in batch])
        }


if __name__ == "__main__":
    # Demo to verify data loading
    from transformers import CLIPProcessor
    
    print("Testing data loading...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Test each dataset individually
    datasets = ['slake', 'vqa_rad', 'pathvqa']
    splits = ['train', 'val', 'test']
    
    for dataset_name in datasets:
        print(f"\nTesting {dataset_name} dataset:")
        
        for split in splits:
            print(f"\n{split} split:")
            dataset = MedicalVQADataset(
                dataset_name=dataset_name,
                split=split,
                processor=processor,
                tokenizer=None  # Using biomedical CLIP style processing
            )
            
            # Print statistics
            stats = dataset.get_statistics()
            print(f"Total samples: {stats['total_samples']}")
            print("\nBinary answer distribution:")
            print(stats['answer_distribution'])
            
            # Test dataloader
            print("\nTesting DataLoader...")
            dataloader = DataLoader(
                dataset,
                batch_size=256,
                shuffle=True,
                collate_fn=collate_fn_med_vqa,
                num_workers=2
            )
            
            # Get first batch

            for batch in dataloader:
                print("\nBatch contents:")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"{key}: Tensor of shape {value.shape}")
                    elif isinstance(value, dict):
                        print(f"{key}: Dict with keys {value.keys()}")
                    elif isinstance(value, list):
                        print(f"{key}: List of length {len(value)}")
                if type(batch['question_tokens']) == dict:
                    print("\nSample question shape:", batch['question_tokens']["input_ids"].shape)
                else:
                    print("\nSample question shape:", batch['question_tokens'].shape)
                print("\nSample image shape:", batch['image'].shape)
                print("Sample answer shape:", batch['answer'].shape)

            
            print("\n" + "="*50)