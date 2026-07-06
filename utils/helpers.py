import os
import torch
import matplotlib.pyplot as plt

def save_checkpoint(model, history, config, num_classes, class_names):
    """Saves a unified model asset bundle containing weights and metadata."""
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    save_path = os.path.join(config['checkpoint_dir'], f"{config['model_save_name']}.pth")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'history': history,
        'num_classes': num_classes,
        'class_names': class_names,
        'model_architecture': config['model_save_name']
    }
    torch.save(checkpoint, save_path)
    print(f"\n Model package and metadata saved successfully to: {save_path}")

def plot_performance_curves(history, save_dir):
    """Generates clean performance curves using a premium dark aesthetic."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Apply a clean dark presentation style matching engineering environments
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#121212')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss Plotting
    ax1.set_facecolor('#1e1e1e')
    ax1.plot(epochs, history['train_loss'], color='#1db954', linewidth=2.5, label='Train Loss')
    ax1.plot(epochs, history['val_loss'], color='#b3b3b3', linewidth=2, linestyle='--', label='Val Loss')
    ax1.set_title('Cross Entropy Loss Curve', fontsize=12, fontweight='bold', color='#ffffff')
    ax1.set_xlabel('Epochs', color='#b3b3b3')
    ax1.set_ylabel('Loss', color='#b3b3b3')
    ax1.grid(True, color='#2c2c2c', linestyle=':')
    ax1.legend()
    
    # Accuracy Plotting
    ax2.set_facecolor('#1e1e1e')
    ax2.plot(epochs, [a * 100 for a in history['train_acc']], color='#1db954', linewidth=2.5, label='Train Acc')
    ax2.plot(epochs, [a * 100 for a in history['val_acc']], color='#b3b3b3', linewidth=2, linestyle='--', label='Val Acc')
    ax2.set_title('Classification Accuracy Curve', fontsize=12, fontweight='bold', color='#ffffff')
    ax2.set_xlabel('Epochs', color='#b3b3b3')
    ax2.set_ylabel('Accuracy (%)', color='#b3b3b3')
    ax2.grid(True, color='#2c2c2c', linestyle=':')
    ax2.legend()
    
    plt.tight_layout()
    chart_path = os.path.join(save_dir, 'training_metrics_chart.png')
    plt.savefig(chart_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
    plt.close()
    print(f"📊 Dark-mode training evaluation curves exported to: {chart_path}")