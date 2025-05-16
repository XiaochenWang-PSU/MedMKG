import torch
from torch.utils.data import DataLoader, random_split
import random
import numpy as npset_random_seed
from open_clip import create_model_from_pretrained, get_tokenizer
from data import *
from model import *
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import logging
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def print_grad_status(model):
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")


def train_kg_model(model, train_loader, val_loader, num_epochs, device, patience=5):
    """Train the KG embedding model"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_loss = float('inf')
    
    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}: Loss = {total_loss:.4f}')
        
        # Save if loss improved
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), 'checkpoints/kg_best.pt')
    
    # Load best model
    model.load_state_dict(torch.load('checkpoints/kg_best.pt'))
    return model

def train_kgemt(model, train_loader, val_loader, num_epochs, device, patience=2):
    """Train the KGEMT model with early stopping"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    best_val_loss = float('inf')
    early_stop_counter = 0
    print_grad_status(model)
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            # Move batch data to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            optimizer.zero_grad()
            img_embed, txt_embed, img_local, txt_local = model(batch)
            loss = model.compute_loss(img_embed, txt_embed, img_local, txt_local)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                img_embed, txt_embed, img_local, txt_local = model(batch)
                loss = model.compute_loss(img_embed, txt_embed, img_local, txt_local)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'checkpoints/kgemt_best.pt')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered")
                break
    
    # Load best model
    model.load_state_dict(torch.load('checkpoints/kgemt_best.pt'))
    return model

class FashionKLIPTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        experiment_name: str = "fashionklip"
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=1e-5)#5)
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
        
    def train(
        self,
        num_epochs: int,
        patience: int = 3,
        grad_clip: float = 1.0
    ) -> Dict[str, Any]:
        """
        Train the FashionKLIP model with early stopping
        
        Args:
            num_epochs: Maximum number of training epochs
            patience: Early stopping patience
            grad_clip: Gradient clipping value
            
        Returns:
            Dictionary containing training history
        """
        best_val_loss = float('inf')
        early_stop_counter = 0
        history = {
            'train_total_loss': [],
            'train_itc_loss': [],
            'train_cva_loss': [],
            'val_total_loss': [],
            'val_itc_loss': [],
            'val_cva_loss': []
        }
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self._log_model_info()
        
        for epoch in tqdm(range(num_epochs)):
            # Training
            train_metrics = self._train_epoch(grad_clip)
            
            # Validation
            val_metrics = self._validate_epoch()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step(val_metrics['total_loss'])
            
            # Log metrics
            self._log_epoch_metrics(epoch, train_metrics, val_metrics)
            
            # Update history
            for k, v in train_metrics.items():
                history[f'train_{k}'].append(v)
            for k, v in val_metrics.items():
                history[f'val_{k}'].append(v)
            
            # Save checkpoint and check early stopping
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                self._save_checkpoint(epoch, val_metrics['total_loss'], is_best=True)
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                self._save_checkpoint(epoch, val_metrics['total_loss'], is_best=False)
                
            if early_stop_counter >= patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Load best model
        best_checkpoint_path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
        if best_checkpoint_path.exists():
            self._load_checkpoint(best_checkpoint_path)
            self.logger.info(f"Loaded best model from {best_checkpoint_path}")
        
        return history
    
    def _train_epoch(self, grad_clip: float) -> Dict[str, float]:
        """Run one training epoch"""
        self.model.train()
        total_loss = 0
        total_itc_loss = 0
        total_cva_loss = 0
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc="Training", leave=False) as pbar:
            for batch in pbar:
                # Move batch to device
                batch = self._prepare_batch(batch)
                
                # Forward pass
                self.optimizer.zero_grad()
                similarity_itc, loss_cva = self.model(batch)
                
                # Calculate loss
                loss = self.criterion(similarity_itc, loss_cva)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                total_itc_loss += similarity_itc.mean().item()
                total_cva_loss += loss_cva.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'itc': f'{similarity_itc.mean().item():.4f}',
                    'cva': f'{loss_cva.item():.4f}'
                })
        
        return {
            'total_loss': total_loss / num_batches,
            'itc_loss': total_itc_loss / num_batches,
            'cva_loss': total_cva_loss / num_batches
        }
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Run one validation epoch"""
        self.model.eval()
        total_loss = 0
        total_itc_loss = 0
        total_cva_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc="Validation", leave=False) as pbar:
                for batch in pbar:
                    batch = self._prepare_batch(batch)
                    similarity_itc, loss_cva = self.model(batch)
                    loss = self.criterion(similarity_itc, loss_cva)
                    
                    total_loss += loss.item()
                    total_itc_loss += similarity_itc.mean().item()
                    total_cva_loss += loss_cva.item()
                    
                    pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        return {
            'total_loss': total_loss / num_batches,
            'itc_loss': total_itc_loss / num_batches,
            'cva_loss': total_cva_loss / num_batches
        }
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch data to device and handle any necessary preprocessing"""
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
    
    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
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
    
    def _load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def _log_model_info(self):
        """Log model information"""
        self.logger.info(f"Model Architecture: {type(self.model).__name__}")
        self.logger.info(f"Total Parameters: {sum(p.numel() for p in self.model.parameters())}")
        self.logger.info(f"Device: {self.device}")
    
    def _log_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float], 
                          val_metrics: Dict[str, float]):
        """Log epoch metrics"""
        metric_str = f"Epoch {epoch + 1} - "
        metric_str += f"Train Loss: {train_metrics['total_loss']:.4f} "
        metric_str += f"(ITC: {train_metrics['itc_loss']:.4f}, CVA: {train_metrics['cva_loss']:.4f}) - "
        metric_str += f"Val Loss: {val_metrics['total_loss']:.4f} "
        metric_str += f"(ITC: {val_metrics['itc_loss']:.4f}, CVA: {val_metrics['cva_loss']:.4f})"
        self.logger.info(metric_str)
class EarlyStopping:
    """Early stopping handler"""
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
        
        return self.early_stop

def train_KCLIP(model, train_loader, val_loader, optimizer, device, num_epochs, patience=3):
    """Training loop with early stopping"""
    early_stopping = EarlyStopping(patience=patience)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1} Training'):
            batch = {k: v.to(device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            loss = model(batch, mode='finetune')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                batch = {k: v.to(device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                loss = model(batch, mode='finetune')
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Print epoch results
        print(f"Epoch {epoch + 1}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Early stopping check
        if early_stopping(avg_val_loss, model):
            print("Early stopping triggered")
            # Restore best model
            model.load_state_dict(early_stopping.best_state)
            break
    
    return model, early_stopping.best_loss

