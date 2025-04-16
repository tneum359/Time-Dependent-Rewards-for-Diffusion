import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import json
import os
import io
import sys
import argparse
from PIL import Image
from torch.utils.data import DataLoader, random_split
import glob

# Add parent directory to path to import dataloader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from source.timedep_dataloader import load_diffusion_dataloader

# Import model from train.py
from train import PEFTImageReward, TimestepEmbedding

class ModelEvaluator:
    def __init__(self, 
                 model_dir="./", 
                 checkpoint_pattern="checkpoint_epoch_*.pt", 
                 output_dir="./",
                 metrics_file="training_metrics.json",
                 image_size=512,
                 batch_size=4,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 val_ratio=0.2):
        """
        Evaluator to monitor model performance and generate visualizations
        
        Args:
            model_dir: Directory where model checkpoints are stored
            checkpoint_pattern: Pattern to match checkpoint files
            output_dir: Directory to save output visualizations
            metrics_file: File to save/load metrics history
            image_size: Size of images for evaluation
            batch_size: Batch size for evaluation
            device: Device to run evaluation on
            val_ratio: Portion of data to use for validation
        """
        self.model_dir = model_dir
        self.checkpoint_pattern = checkpoint_pattern
        self.output_dir = output_dir
        self.metrics_file = os.path.join(output_dir, metrics_file)
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = device
        self.val_ratio = val_ratio
        
        # Load or initialize metrics history
        self.metrics = self._load_metrics()
        
        # Create validation dataset
        self._setup_validation_data()
        
    def _setup_validation_data(self):
        """Setup validation dataset"""
        # Load the full dataset
        dataset = load_diffusion_dataloader(
            batch_size=1,  # We'll set batch size in DataLoader
            image_size=self.image_size
        ).dataset
        
        # Calculate split sizes
        val_size = int(len(dataset) * self.val_ratio)
        train_size = len(dataset) - val_size
        
        # Split dataset and get validation set
        _, val_dataset = random_split(dataset, [train_size, val_size], 
                                     generator=torch.Generator().manual_seed(42))  # Fixed seed for consistent splits
        
        # Create validation dataloader
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
        
    def _load_metrics(self):
        """Load metrics from file or initialize new metrics dict"""
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'epochs': [],
                'train_loss': [],
                'val_loss': [],
                'correlation': []
            }
    
    def _save_metrics(self):
        """Save metrics to file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def find_latest_checkpoint(self):
        """Find the latest checkpoint file"""
        checkpoint_files = glob.glob(os.path.join(self.model_dir, self.checkpoint_pattern))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found matching pattern {self.checkpoint_pattern} in {self.model_dir}")
        
        # Extract epoch numbers and find the latest
        latest_epoch = -1
        latest_file = None
        for file in checkpoint_files:
            try:
                epoch = int(file.split('_')[-1].split('.')[0])
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_file = file
            except:
                continue
        
        return latest_file, latest_epoch
    
    def load_model(self, checkpoint_path):
        """Load model from checkpoint"""
        model = PEFTImageReward().to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, checkpoint
    
    def evaluate_model(self, model):
        """Evaluate model on validation set"""
        total_loss = 0.0
        total_items = 0
        all_predicted = []
        all_target = []
        
        loss_fn = nn.MSELoss(reduction='sum')
        
        with torch.no_grad():
            for final_images, intermediate_images, timesteps in self.val_dataloader:
                # Move data to device
                final_images = final_images.to(self.device)
                intermediate_images = intermediate_images.to(self.device)
                timesteps = timesteps.to(self.device)
                
                # Get target reward from original model
                target_rewards = model.original_reward_model.score(final_images)
                
                # Get predicted reward from our model
                predicted_rewards = model(intermediate_images, timesteps)
                
                # Calculate loss
                loss = loss_fn(predicted_rewards, target_rewards)
                
                # Track statistics
                batch_size = final_images.size(0)
                total_loss += loss.item()
                total_items += batch_size
                
                # Store predictions and targets for correlation calculation
                all_predicted.append(predicted_rewards.cpu())
                all_target.append(target_rewards.cpu())
        
        # Calculate metrics
        avg_loss = total_loss / total_items
        
        # Calculate Pearson correlation
        all_predicted = torch.cat(all_predicted).numpy().flatten()
        all_target = torch.cat(all_target).numpy().flatten()
        
        # Correlation is only defined if we have variation in our predictions and targets
        if np.std(all_predicted) > 0 and np.std(all_target) > 0:
            correlation = np.corrcoef(all_predicted, all_target)[0, 1]
        else:
            correlation = 0.0
        
        return avg_loss, correlation
    
    def create_plots(self):
        """Create plots of training progress"""
        # Create a figure with two subplots (losses and correlation)
        plt.figure(figsize=(12, 6))
        
        # Plot losses
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(self.metrics['epochs'], self.metrics['train_loss'], label='Training Loss', marker='o')
        ax1.plot(self.metrics['epochs'], self.metrics['val_loss'], label='Validation Loss', marker='s')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot correlation
        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(self.metrics['epochs'], self.metrics['correlation'], label='Correlation', marker='d', color='green')
        ax2.set_title('Prediction Correlation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Pearson Correlation')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Convert plot to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        
        return img
    
    def display_terminal_image(self, img, width=80):
        """Display an image in the terminal using ASCII characters"""
        # Resize image to fit terminal width
        aspect_ratio = img.width / img.height
        height = int(width / aspect_ratio)
        img = img.resize((width, height))
        
        # Convert to grayscale for ASCII display
        img = img.convert('L')
        
        # ASCII characters from dark to light
        ascii_chars = ' .:-=+*#%@'
        
        # Convert pixels to ASCII
        pixels = np.array(img)
        ascii_img = []
        
        for row in pixels:
            ascii_row = ''
            for pixel in row:
                # Map pixel value (0-255) to ASCII character
                index = int(pixel * (len(ascii_chars) - 1) / 255)
                ascii_row += ascii_chars[index]
            ascii_img.append(ascii_row)
        
        # Print ASCII image
        print("\n")
        for row in ascii_img:
            print(row)
        print("\n")
    
    def update_metrics_from_checkpoint(self, checkpoint, epoch):
        """Update metrics from checkpoint data"""
        # Check if we already have this epoch in metrics
        if epoch in self.metrics['epochs']:
            idx = self.metrics['epochs'].index(epoch)
            self.metrics['epochs'][idx] = epoch
            self.metrics['train_loss'][idx] = checkpoint.get('train_loss', 0)
            self.metrics['val_loss'][idx] = checkpoint.get('val_loss', 0)
            self.metrics['correlation'][idx] = checkpoint.get('correlation', 0)
        else:
            self.metrics['epochs'].append(epoch)
            self.metrics['train_loss'].append(checkpoint.get('train_loss', 0))
            self.metrics['val_loss'].append(checkpoint.get('val_loss', 0))
            self.metrics['correlation'].append(checkpoint.get('correlation', 0))
            
        # Sort metrics by epoch
        idxs = sorted(range(len(self.metrics['epochs'])), key=lambda i: self.metrics['epochs'][i])
        self.metrics['epochs'] = [self.metrics['epochs'][i] for i in idxs]
        self.metrics['train_loss'] = [self.metrics['train_loss'][i] for i in idxs]
        self.metrics['val_loss'] = [self.metrics['val_loss'][i] for i in idxs]
        self.metrics['correlation'] = [self.metrics['correlation'][i] for i in idxs]
    
    def run_evaluation(self, checkpoint_path=None):
        """Run evaluation on the latest checkpoint or specified checkpoint"""
        # Find latest checkpoint if not specified
        if checkpoint_path is None:
            checkpoint_path, epoch = self.find_latest_checkpoint()
        else:
            try:
                epoch = int(checkpoint_path.split('_')[-1].split('.')[0])
            except:
                epoch = len(self.metrics['epochs']) + 1
        
        print(f"Evaluating checkpoint from epoch {epoch}: {checkpoint_path}")
        
        # Load model
        model, checkpoint = self.load_model(checkpoint_path)
        
        # Evaluate model (if not already evaluated)
        if ('val_loss' not in checkpoint or 'correlation' not in checkpoint):
            val_loss, correlation = self.evaluate_model(model)
            checkpoint['val_loss'] = val_loss
            checkpoint['correlation'] = correlation
            
            # Save updated checkpoint
            torch.save(checkpoint, checkpoint_path)
            print(f"Updated checkpoint with evaluation metrics")
        else:
            val_loss = checkpoint['val_loss']
            correlation = checkpoint['correlation']
        
        # Update metrics
        self.update_metrics_from_checkpoint(checkpoint, epoch)
        
        # Save metrics
        self._save_metrics()
        
        # Print evaluation results
        print(f"Evaluation Results (Epoch {epoch}):")
        print(f"  Train Loss: {checkpoint.get('train_loss', 'N/A'):.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Correlation: {correlation:.4f}")
        
        # Create and save plots
        plot_img = self.create_plots()
        plot_img.save(os.path.join(self.output_dir, "training_progress.png"))
        
        # Display plots in terminal
        self.display_terminal_image(plot_img)
        
        return val_loss, correlation

def eval_callback(epoch_num, model_dir="./"):
    """Callback function to evaluate after each epoch"""
    evaluator = ModelEvaluator(model_dir=model_dir)
    checkpoint_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch_num}.pt")
    return evaluator.run_evaluation(checkpoint_path)

def scan_all_checkpoints(model_dir="./"):
    """Scan and evaluate all available checkpoints"""
    evaluator = ModelEvaluator(model_dir=model_dir)
    checkpoints = glob.glob(os.path.join(model_dir, "checkpoint_epoch_*.pt"))
    
    for checkpoint in sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0])):
        evaluator.run_evaluation(checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ImageReward model checkpoints")
    parser.add_argument("--model_dir", type=str, default="./", help="Directory containing model checkpoints")
    parser.add_argument("--epoch", type=int, default=None, help="Specific epoch to evaluate (optional)")
    parser.add_argument("--all", action="store_true", help="Evaluate all checkpoints")
    args = parser.parse_args()
    
    if args.all:
        scan_all_checkpoints(args.model_dir)
    elif args.epoch is not None:
        eval_callback(args.epoch, args.model_dir)
    else:
        # Evaluate latest checkpoint
        evaluator = ModelEvaluator(model_dir=args.model_dir)
        evaluator.run_evaluation()

