import torch
from torch.utils.data import DataLoader
from data_loader import KnowledgeGraphDataset
from model import *
import torch.optim as optim
from tqdm import tqdm  # Import tqdm for progress bar
import os

def train(model, train_dataloader, val_dataloader, optimizer, num_epochs=10, device="cpu", patience=3):
    """
    Training loop with validation loss monitoring, early stopping, and checkpoint saving.

    Args:
        model (nn.Module): The baseline model (e.g., TransE, SimplE, etc.).
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        optimizer: Optimizer for model training.
        num_epochs (int): Number of training epochs.
        device (str): Device to train on ('cpu' or 'cuda').
        patience (int): Number of epochs to wait for improvement before stopping.

    Returns:
        best_val_loss (float): The best validation loss achieved during training.
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs("ckpt", exist_ok=True)
    checkpoint_path = f"ckpt/{model.__class__.__name__}.pt"  # Dynamic checkpoint path

    model = model.to(device)
    model.train()

    early_stopping_counter = 0
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        total_train_loss = 0
        
        # Training phase
        model.train()
        train_progress = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=True)
        for pos_triplets, neg_triplets in train_progress:
            pos_triplets, neg_triplets = pos_triplets.to(device), neg_triplets.view(-1, 3).to(device)

            optimizer.zero_grad()
            loss = model(pos_triplets, neg_triplets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_progress.set_postfix(train_loss=loss.item())

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Average Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for pos_triplets, neg_triplets in val_dataloader:
                pos_triplets, neg_triplets = pos_triplets.to(device), neg_triplets.view(-1, 3).to(device)
                val_loss = model(pos_triplets, neg_triplets)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}")

        # Check for improvement
        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved: {best_val_loss:.4f} -> {avg_val_loss:.4f}. Saving model checkpoint.")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)  # Save the best model checkpoint
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"No improvement. Early stopping counter: {early_stopping_counter}/{patience}")

        # Trigger early stopping
        if early_stopping_counter >= patience:
            print("Early stopping triggered!")
            break

    print(f"Training completed. Best Validation Loss: {best_val_loss:.4f}")
    
    # Load the best model checkpoint
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded best model checkpoint from {checkpoint_path}.")
    return best_val_loss