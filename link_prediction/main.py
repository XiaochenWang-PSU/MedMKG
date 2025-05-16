import torch
from torch.utils.data import DataLoader, random_split
from model import * 
from train import train
from data_loader import KnowledgeGraphDataset, build_graph_from_triplets
from evaluator import Evaluator, EarlyStopping  # Import the evaluator
import random
import numpy as np
import gc

def set_random_seed(seed=42):
    """
    Set random seeds for reproducibility.
    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)               # Python's built-in random module
    np.random.seed(seed)            # NumPy
    torch.manual_seed(seed)         # PyTorch
    torch.cuda.manual_seed(seed)    # PyTorch CUDA (if available)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU environments
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.benchmark = False
    
def main():
    """
    Main function to train and evaluate multiple baseline models on the knowledge graph dataset.
    """
    
    
    
    # Randomness control
    set_random_seed(seed=42)  # Fix random seed for reproducibility
    
    
    # Parameters
    DATA_CSV = YOUR_KG_PATH  # Path to the input CSV file
    NUM_ENTITIES = 8017            # Total number of entities in the dataset
    NUM_RELATIONS = 262              # Total number of relations
    EMBEDDING_DIM = 64             # Embedding dimension for all models
    RELATION_DIM = 64              # Relation-specific dimension for TransR/TuckER
    TENSOR_DIM = 10                 # Tensor dimension for NTN
    BATCH_SIZE = 8192               # Batch size for DataLoader
    NUM_EPOCHS = 500                 # Number of training epochs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Training device

    VALIDATION_RATIO = 0.1         # 10% for validation
    TEST_RATIO = 0.1                # 10% for testing
    TRAIN_RATIO = 1.0 - VALIDATION_RATIO - TEST_RATIO  # Remaining for training

    # Initialize Dataset
    dataset = KnowledgeGraphDataset(DATA_CSV, NUM_ENTITIES, NUM_RELATIONS, neg_ratio=1)
    dataset_size = len(dataset)
    train_size = int(TRAIN_RATIO * dataset_size)
    val_size = int(VALIDATION_RATIO * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # all_triplets_array = dataset.triplets  # shape (N, 3)
    # edge_index, edge_type = build_graph_from_triplets(all_triplets_array)
    
    # Define all baseline models
    models = {
        "TransE": TransE(NUM_ENTITIES, NUM_RELATIONS, EMBEDDING_DIM, margin=1.0),
        "TransD": TransD(NUM_ENTITIES, NUM_RELATIONS, EMBEDDING_DIM, margin=1.0),
        "TransR": TransR(NUM_ENTITIES, NUM_RELATIONS, EMBEDDING_DIM, RELATION_DIM, margin=1.0),
        "TransH": TransH(NUM_ENTITIES, NUM_RELATIONS, EMBEDDING_DIM, margin=1.0),
        "DistMult": DistMult(NUM_ENTITIES, NUM_RELATIONS, EMBEDDING_DIM),
        "ComplEx": ComplEx(NUM_ENTITIES, NUM_RELATIONS, EMBEDDING_DIM),
        "SimplE": SimplE(NUM_ENTITIES, NUM_RELATIONS, EMBEDDING_DIM),
        "ConvE": ConvE(NUM_ENTITIES, NUM_RELATIONS, EMBEDDING_DIM),
        "ConvR": ConvR(NUM_ENTITIES, NUM_RELATIONS, EMBEDDING_DIM),
        "HypER": HypER(NUM_ENTITIES, NUM_RELATIONS, EMBEDDING_DIM),     
        "AttH": AttH(NUM_ENTITIES, NUM_RELATIONS, EMBEDDING_DIM),
        "RotatE": RotatE(NUM_ENTITIES, NUM_RELATIONS, EMBEDDING_DIM),  
        "MurP": MurP(NUM_ENTITIES, NUM_RELATIONS, EMBEDDING_DIM),
        "MurE": MurE(NUM_ENTITIES, NUM_RELATIONS, EMBEDDING_DIM),    
        "TuckER": TuckER(NUM_ENTITIES, NUM_RELATIONS, EMBEDDING_DIM),
        "RESCAL": RESCAL(NUM_ENTITIES, NUM_RELATIONS, EMBEDDING_DIM),
        "NTN": NTN(NUM_ENTITIES, NUM_RELATIONS, EMBEDDING_DIM, tensor_dim=TENSOR_DIM),
    }

    # Train and evaluate each model
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    
        if isinstance(model, RGCN):
            best_val_loss = train(model, train_loader, val_loader, optimizer,
                                  num_epochs=NUM_EPOCHS, device=DEVICE, patience=5,
                                  edge_index=edge_index, edge_type=edge_type)
        else:
            best_val_loss = train(model, train_loader, val_loader, optimizer,
                                  num_epochs=NUM_EPOCHS, device=DEVICE, patience=5)
        
        # Train the model with early stopping
        # best_val_loss = train(model, train_loader, val_loader, optimizer, num_epochs=NUM_EPOCHS, device=DEVICE, patience=5)
    
        print(f"Best Validation Loss for {model_name}: {best_val_loss:.4f}")
    
        # Evaluate on the test set
        print(f"\nEvaluating {model_name} on Test Set...")
        evaluator = Evaluator(model, test_dataset, NUM_ENTITIES, NUM_RELATIONS, dataset,  device=DEVICE, batch_size=8192)
    
        # Evaluate for Head Prediction
        head_metrics = evaluator.evaluate(task="head")
        print(f"Head Prediction Metrics for {model_name}: {head_metrics}")
    
        # Evaluate for Tail Prediction
        tail_metrics = evaluator.evaluate(task="tail")
        print(f"Tail Prediction Metrics for {model_name}: {tail_metrics}")
    
#        # Evaluate for Link Prediction
        if isinstance(model, RGCN):
            link_metrics = evaluator.evaluate(task="link", edge_index=edge_index, edge_type=edge_type)
        else:
            link_metrics = evaluator.evaluate(task="link")

#        link_metrics = evaluator.evaluate(task="link")
#        print(f"Link Prediction Metrics for {model_name}: {link_metrics}")
        
        print(f"Clearing CUDA memory for {model_name}...")
        del model, optimizer, evaluator  # Delete large objects
        torch.cuda.empty_cache()         # Clear cached memory
        gc.collect()                     # Run garbage collection

if __name__ == "__main__":
    main()
