import torch
from torch.utils.data import DataLoader, random_split
import json
import random
import numpy as np
from pathlib import Path
from open_clip import create_model_from_pretrained, get_tokenizer, create_model_and_transforms
from PIL import Image
from bs4 import BeautifulSoup
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import CLIPProcessor, CLIPModel, CLIPConfig
from train import *
from model import *
from data import *
from evaluator import *



   

def main(config):
    set_random_seed(config['seed'])
    device = torch.device(f"cuda:{config['gpu_id']}" if torch.cuda.is_available() else 'cpu')
    
    
    
    if 'ViT-' in config['clip_model_name']:
        print('load original clip')
        clip_model, _, processor = create_model_and_transforms(config['clip_model_name'], pretrained='openai') 
        tokenizer = get_tokenizer('ViT-B-16')
        collate_fn = collate_fn_normal
    elif 'fla' in config['clip_model_name']:
        print('load pubmed clip')
        processor = CLIPProcessor.from_pretrained(config['clip_model_name'])
        config_ = CLIPConfig.from_pretrained(config['clip_model_name'])
        config_.max_position_embeddings = 512  # Modify the desired configuration attribute
        
        # Load the model with the modified configuration
        # clip_model = CLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32", config=config_)
        clip_model = CLIPModel.from_pretrained(config['clip_model_name'])
        clip_model.config.max_position_embeddings = 512
        # clip_model.resize_token_embeddings(new_num_tokens=512)
        clip_model = OriginalCLIP(clip_model)
        tokenizer = None
        collate_fn = collate_fn_pub
        # collate_fn = collate_fn_pubmed
    else:
        print('load customized clip')
        clip_model, processor = create_model_from_pretrained(config['clip_model_name'])
        tokenizer = get_tokenizer(config['clip_model_name'])
        collate_fn = collate_fn_normal
    # clip_model, processor = create_model_from_pretrained('hf-hub:xcwangpsu/MedCSP_clip')
    clip_model = clip_model.to(device)
    
    kg_retriever = KnowledgeGraphRetriever(
       kg_path=YOUR_KG_PATH,
       clip_model=clip_model,
       processor=processor,
       biomedclip=not tokenizer
   )
    
    dataset = OpenIDataset(
        text_path=config['text_path'],
        image_path=config['image_path'],
        clip_processor=processor,
        clip_tokenizer=tokenizer
    )

#    Uncommet these lines for MIMIC dataset
#    dataset = MIMICDataset(
#    image_base_path=YOUR_BASE_IMAGE_PATH_FOR_MIMIC_JPG,
#    text_base_path=YOUR_BASE_TEXT_PATH_FOR_MIMIC_REPORT,
#    metadata_csv_path=YOUR_MIMIC_JPG_META_FILE,
#    clip_processor=processor,
#    clip_tokenizer=tokenizer,
#    kg_image_path_file=YOUR_IMAGE_MAPPING_FILE,
#    sample_size=10000
#)

    dataset_size = len(dataset)
    train_size = int((1 - config['val_ratio'] - config['test_ratio']) * dataset_size)
    val_size = int(config['val_ratio'] * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
    dataset, 
    [train_size, val_size, test_size],
    generator=generator
)
    
    # if "BiomedCLIP-PubMedBERT_256" in config['clip_model_name']:
    #   collate_fn = collate_fn_pubmed
    
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
        model = KnowledgeClip(
        # clip_model_name=config['clip_model_name'],
        clip_model=clip_model,
        kg_retriever=kg_retriever
    ).to(device)
    
#    Uncomment for FashionKLIP
#    model = FashionKLIP(
#        # clip_model_name=config['clip_model_name'],
#        clip_model=clip_model,
#        kg_retriever=kg_retriever
#    ).to(device)
#    for name, param in model.named_parameters():
#      print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")

    trainer = FashionKLIPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=config['checkpoint_dir'],
        criterion = CombinedLoss()
    )
    
    trainer.train(num_epochs=config['num_epochs'])
    
    results = evaluate(model, test_loader, device)
    print("Precision at k:", results["precision"])
    print("Recall at k:", results["recall"])
    save_dir = "original_mimic_1e4"
    os.makedirs(save_dir, exist_ok=True)
    clip_model_part = config['clip_model_name'].split('/')[-1]  # just the model name
    image_path_part = os.path.basename(config['image_path'].rstrip('/'))  # just the last folder name

    filename = f"{clip_model_part}_{image_path_part}_metrics.json"
    save_path = os.path.join(save_dir, filename)

    # Save results to JSON
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Metrics saved to {save_path}")
    return results

if __name__ == "__main__":
    config = {
        'seed': 42,
        'gpu_id': 0,
        'clip_model_name': "flaviagiammarino/pubmed-clip-vit-base-patch32",
        'kg_path': YOUR_PATH_OF_KG,
        'text_path': YOUR_PATH_OF_TEXT,
        'image_path': YOUR_PATH_OF_IMAGE,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'batch_size': 128,
        'num_workers': 47,
        'num_epochs': 30,
        'checkpoint_dir': 'checkpoints'
    }
    
    main(config)
#    

    config = {
        'seed': 42,
        'gpu_id': 0,
        'clip_model_name': 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        'kg_path': YOUR_PATH_OF_KG,
        'text_path': YOUR_PATH_OF_TEXT,
        'image_path': YOUR_PATH_OF_IMAGE,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'batch_size': 128,
        'num_workers': 47,
        'num_epochs': 30,
        'checkpoint_dir': 'checkpoints'
    }
    
    main(config)
    config = {
        'seed': 42,
        'gpu_id': 0,
        'clip_model_name': 'hf-hub:xcwangpsu/MedCSP_clip',
        'kg_path': YOUR_PATH_OF_KG,
        'text_path': YOUR_PATH_OF_TEXT,
        'image_path': YOUR_PATH_OF_IMAGE,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'batch_size': 128,
        'num_workers': 47,
        'num_epochs': 30,
        'checkpoint_dir': 'checkpoints'
    }
    
    main(config)
    
    config = {
        'seed': 42,
        'gpu_id': 0,
        'clip_model_name': 'ViT-B-16',
        'kg_path': YOUR_PATH_OF_KG,
        'text_path': YOUR_PATH_OF_TEXT,
        'image_path': YOUR_PATH_OF_IMAGE,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'batch_size': 128,
        'num_workers': 47,
        'num_epochs': 30,
        'checkpoint_dir': 'checkpoints'
    }
    
    main(config)
