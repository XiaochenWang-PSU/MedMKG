import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from pathlib import Path
from open_clip import create_model_from_pretrained, get_tokenizer, create_model_and_transforms
from transformers import CLIPProcessor, CLIPModel, CLIPConfig
import json
import os

from model import *
from data import MedicalVQADataset, collate_fn_med_vqa
from evaluate import evaluate_model
from train import MedicalVQATrainer

def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_clip_model(model_name: str, device: str):
    """
    Load CLIP model and processor/tokenizer based on model name
    """
    
    print('Loading', model_name)
    if 'ViT-' in model_name:
        clip_model, _, processor = create_model_and_transforms(model_name, pretrained='openai')
        tokenizer = get_tokenizer('ViT-B-16')
    elif 'fla' in model_name:
        processor = CLIPProcessor.from_pretrained(model_name)
        config = CLIPConfig.from_pretrained(model_name)
        config.max_position_embeddings = 512
        clip_model = CLIPModel.from_pretrained(model_name)
        clip_model.config.max_position_embeddings = 512
        clip_model = ModifiedCLIP(clip_model)
        tokenizer = None
    else:
        clip_model, processor = create_model_from_pretrained(model_name)
        tokenizer = get_tokenizer(model_name)
        
    return clip_model.to(device), processor, tokenizer

def main(config):
    # Set up environment
    set_random_seed(config['seed'])
    device = torch.device(f"cuda:{config['gpu_id']}" if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    output_dir = Path(config['checkpoint_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Load CLIP model
    clip_model, processor, tokenizer = load_clip_model(config['clip_model_name'], device)
    
    # Initialize datasets for each specified dataset
    splits = ['train', 'val', 'test']
    dataloaders = {split: [] for split in splits}
    
    
    kg_retriever = KGRetriever(
        kg_path=YOUR_KG_PATH,  # Your KG path
        clip_model=clip_model,
        processor=processor,
        tokenizer=tokenizer,
        biomedclip='fla' in config['clip_model_name'],  # Check if using biomedical CLIP
        target_edges=None,  # Target number of edges
    min_freq=2  # Minimum entity frequency
    )
    
    for dataset_name in config['datasets']:
        print(f"\nLoading {dataset_name} dataset...")
        dataset_path = config['data_paths'][dataset_name]
        
        for split in splits:
            dataset = MedicalVQADataset(
                dataset_name=dataset_name,
                split=split,
                processor=processor,
                tokenizer=tokenizer,
                max_length=config.get('max_length', 77),
                val_split=config.get('val_split', 0.1),
                kg_retriever=kg_retriever,  # Pass retriever for subgraph extraction
                # cache_path=f"cache/{dataset_name}_{split}_graph_cache.pkl"
                cache_path=f"cache/{dataset_name}_{split}_graph_cache_with_edge.pkl"
            )
            
            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=config['batch_size'],
                num_workers=config['num_workers'],
                shuffle=(split == 'train'),
                collate_fn=collate_fn_med_vqa
            )
            
            dataloaders[split].append(dataloader)
    
##    # Create model
#    model = MedicalVQAModel(
#        clip_model=clip_model,
#        projection_dim=config.get('projection_dim', 512)
#    ).to(device)
    

    
##    # Then create KRISP model
#    model = KRISPMedVQAModel(
#        clip_model=clip_model,
#        kg_retriever=kg_retriever,
#        projection_dim=config.get('projection_dim', 512),
#        hidden_dim=config.get('hidden_dim', 512),
#        embedding_cache_path = 'cache/' + config['clip_model_name'].split('/')[-1] + '_kg_embedding.pt'
#    ).to(device)
#        
    # Then create EKGRL model
#    model = EKGRLModel(
#        clip_model=clip_model,
#        kg_retriever=kg_retriever,
#        projection_dim=config.get('projection_dim', 512)
#    ).to(device)

# )

	# MR-MKG
    model = GCNVQAModel(
    clip_model=clip_model,
    kg_retriever=kg_retriever,
    projection_dim=config.get('projection_dim', 64),
    hidden_dim=config.get('hidden_dim', 512),
    # num_heads=config.get('num_heads', 4),
    alpha=config.get('alpha', 0.2),
    lambda_weight=config.get('lambda_weight', 0.1),
    biomedclip='fla' in config['clip_model_name'],
    embedding_cache_path = 'cache/' + config['clip_model_name'].split('/')[-1] + '_kg_embedding.pt'
).to(device)

#    model = MKBNModel(
#    clip_model=clip_model,
#    kg_retriever=kg_retriever,
#    projection_dim=config.get('projection_dim', 512),
#    hidden_dim=config.get('hidden_dim', 512),
#    embedding_cache_path = 'cache/' + config['clip_model_name'].split('/')[-1] + '_kg_embedding.pt'
#).to(device)

#    model = KPathVQAModel(
#    clip_model=clip_model,
#    kg_retriever=kg_retriever,
#    projection_dim=config.get('projection_dim', 512),
#    num_heads=config.get('num_heads', 1),
#    biomedclip='fla' in config['clip_model_name']
#).to(device)
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
        patience=4,
        verbose=True,
        min_lr=1e-6
    )
    
    # Create trainer
    trainer = MedicalVQATrainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=output_dir / 'checkpoints',
        experiment_name=config.get('experiment_name', 'med_vqa')
    )
    
    # Train model
    trainer.train(
        num_epochs=config['num_epochs'],
        patience=config.get('patience', 5)
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    all_test_metrics = {}
    for idx, test_loader in enumerate(dataloaders['test']):
        dataset_name = config['datasets'][idx]
        print(f"\nEvaluating {dataset_name}...")
        test_metrics = evaluate_model(model, test_loader, device)
        all_test_metrics[dataset_name] = test_metrics
    
    # Save test metrics
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(all_test_metrics, f, indent=4)
    
    print("\nTraining completed!")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    # Define CLIP models to evaluate
    clip_models = [
         "flaviagiammarino/pubmed-clip-vit-base-patch32",
         "ViT-B-16",
        "hf-hub:xcwangpsu/MedCSP_clip",
         "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    ]
    
    # Define datasets and their paths
    dataset_configs = [
        {'name': 'slake', 'path': YOUR_SLAKE_PATH},
        {'name': 'vqa_rad', 'path': YOUR_RAD_PATH},
        {'name': 'pathvqa', 'path': YOUR_PATH_VQA_PATH}
    ]
    
    # Base configuration
    base_config = {
        'seed': 42,
        'gpu_id': 1,
        'val_split': 0.1,
        'batch_size': 64,
        'num_workers': 47,
        'num_epochs': 30,
        'learning_rate': 2e-5,#3e-5,
        'weight_decay': 0.2,#0.2,
        'patience': 5,
        'projection_dim': 512,
        'max_length': 77
    }
    
    # Run experiments for each combination

        
    for dataset in dataset_configs:
        for clip_model in clip_models:
            clip_name = clip_model.split('/')[-1]  # Get last part of model path
            print(f"\n{'='*80}")
            print(f"Running experiment with:")
            print(f"CLIP model: {clip_model}")
            print(f"Dataset: {dataset['name']}")
            print(f"{'='*80}\n")
            
            # Create specific configuration for this run
            config = base_config.copy()
            config.update({
                'clip_model_name': clip_model,
                'datasets': [dataset['name']],
                'data_paths': {dataset['name']: dataset['path']},
                'checkpoint_dir': f"checkpoints_0/{clip_name}/{dataset['name']}",
                'experiment_name': f"med_vqa_{clip_name}_{dataset['name']}"
            })
            
            
            try:
            	main(config)
            except Exception as e:
                print(f"\nError running experiment with {clip_model} on {dataset['name']}:")
                print(f"Error message: {str(e)}")
                print("Continuing with next configuration...\n")
                continue
