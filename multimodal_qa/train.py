import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch.nn.functional as F

class MedicalVQATrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: List[DataLoader],
        val_loader: List[DataLoader],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        experiment_name: str = "medical_vqa"
    ):
        self.model = model.to(device)
        self.train_loaders = train_loader
        self.val_loaders = val_loader
        self.optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=1e-5)
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{checkpoint_dir}/{experiment_name}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        # Combine batches from all datasets
        for train_loader in self.train_loaders:
            for batch in tqdm(train_loader, desc=f'Training Epoch {epoch}'):
                if batch is None:
                    continue
                    
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                self.optimizer.zero_grad()
                print(batch.keys())
                logits, loss = self.model(batch)
                
                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                # Update metrics
                batch_size = batch['answer'].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        return total_loss / total_samples if total_samples > 0 else float('inf')
    
    def validate(self) -> float:
        """Run validation"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for val_loader in self.val_loaders:
                for batch in tqdm(val_loader, desc='Validating'):
                    if batch is None:
                        continue
                        
                    # Move batch to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Forward pass
                    logits, loss = self.model(batch)
                    
                    # Update metrics
                    batch_size = batch['answer'].size(0)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
        
        return total_loss / total_samples if total_samples > 0 else float('inf')
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / f"{self.experiment_name}_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint if needed
        if is_best:
            best_path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['val_loss']
    
    def train(self, num_epochs: int, patience: int = 5) -> Dict[str, List[float]]:
        """
        Train the model
        Args:
            num_epochs: Maximum number of epochs to train
            patience: Early stopping patience
        Returns:
            Dictionary containing training history
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        best_val_loss = float('inf')
        early_stop_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(epoch)
            train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate()
            val_losses.append(val_loss)
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}"
            )
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Save checkpoint and check early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                self.save_checkpoint(epoch, val_loss, is_best=False)
            
            if early_stop_counter >= patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Load best model
        best_checkpoint_path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
        if best_checkpoint_path.exists():
            self.load_checkpoint(best_checkpoint_path)
            self.logger.info(f"Loaded best model from {best_checkpoint_path}")
        
        return {
            'train_loss': train_losses,
            'val_loss': val_losses
        }