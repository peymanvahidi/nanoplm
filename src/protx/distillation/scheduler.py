import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
import numpy as np


class ProtXScheduler(_LRScheduler):
    """
    Scheduler that combines linear warmup with cosine annealing warm restarts and weight decay.
    
    Args:
        optimizer: Optimizer to adjust learning rate for
        warmup_steps: Number of steps for linear warmup
        initial_lr: Initial learning rate at the start of warmup
        max_lr: Maximum learning rate after warmup
        min_lr: Minimum learning rate during cosine annealing
        T_0: First cycle step size (iterations for first restart)
        T_mult: Multiplicative factor to increase T_i after restart
        gamma: Weight decay factor applied each cycle
        max_lr_threshold: Threshold for max_lr
        max_cycle_length: Maximum cycle length
        last_epoch: The index of last epoch
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 0,
        initial_lr: float = 1e-8,
        max_lr: float = 1e-3,
        min_lr: float = 1e-5,
        T_0: int = 5000,
        T_mult: int = 2,
        gamma: float = 1.0,
        max_lr_threshold: float = None,
        max_cycle_length: int = None,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.initial_lr = float(initial_lr)
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.T_0 = int(T_0)
        self.T_mult = int(T_mult)
        self.gamma = float(gamma)
        self.max_lr_threshold = max_lr_threshold if max_lr_threshold is not None else min_lr * 3
        self.max_cycle_length = max_cycle_length
        
        self.cycle = 0
        self.T_i = T_0  # Current cycle length
        self.T_cur = 0  # Steps since last restart
        self.cycle_start = warmup_steps  # Step where current cycle started
        
        super(ProtXScheduler, self).__init__(optimizer, last_epoch)
        
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr if self.warmup_steps > 0 else self.max_lr
            self.base_lrs.append(self.initial_lr)
    
    def get_lr(self):
        if self.last_epoch < 0:
            return self.base_lrs
        
        # Linear warmup phase
        if self.last_epoch < self.warmup_steps:
            factor = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [self.initial_lr + factor * (self.max_lr - self.initial_lr) for _ in self.base_lrs]
        
        # Cosine annealing phase with warm restarts
        current_max_lr = self.max_lr * (self.gamma ** self.cycle)
        # Apply threshold to current_max_lr
        current_max_lr = max(current_max_lr, self.max_lr_threshold)
        return [
            self.min_lr + 0.5 * (current_max_lr - self.min_lr) * 
            (1 + math.cos(math.pi * self.T_cur / self.T_i))
            for _ in self.base_lrs
        ]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        if self.last_epoch < 0:
            self.last_epoch = 0
            
        if self.last_epoch >= self.warmup_steps:
            # Calculate effective step in cosine schedule (steps since warmup ended)
            cosine_step = self.last_epoch - self.warmup_steps
            
            # Calculate steps since the start of current cycle
            steps_in_cycle = self.last_epoch - self.cycle_start
            
            # Handle restart logic
            if steps_in_cycle >= self.T_i:
                # We've completed a cycle, time to restart
                self.cycle += 1
                self.cycle_start = self.last_epoch  # Update cycle start step
                self.T_cur = 0  # Reset position within cycle
                # Update cycle length for next cycle
                self.T_i = self.T_0 * (self.T_mult ** self.cycle)
                # Apply max cycle length if set
                if self.max_cycle_length is not None:
                    self.T_i = min(self.T_i, self.max_cycle_length)
            else:
                # Update position within current cycle
                self.T_cur = steps_in_cycle
        
        # Apply new learning rates
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

def visualize_lr_schedule(
    train_examples: int,
    batch_size: int,
    num_epochs: int,
    warmup_steps: int = 1000,
    initial_lr: float = 1e-8,
    max_lr: float = 1e-3,
    min_lr: float = 1e-5,
    T_0: int = 5000,
    T_mult: int = 2,
    gamma: float = 0.95,
    max_lr_threshold: float = None,
    max_cycle_length: int = None,
    figsize: tuple = (12, 6),
    save_path: str = None
):
    """
    Visualize the learning rate schedule for a ProtXScheduler.
    
    Args:
        train_examples: Number of training examples
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        warmup_steps: Number of steps for linear warmup
        initial_lr: Initial learning rate at the start of warmup
        max_lr: Maximum learning rate after warmup
        min_lr: Minimum learning rate during cosine annealing
        T_0: First cycle step size (iterations for first restart)
        T_mult: Multiplicative factor to increase T_i after restart
        gamma: Weight decay factor applied each cycle
        max_lr_threshold: Threshold for max_lr during cosine annealing
        max_cycle_length: Maximum cycle length in steps (defaults to 2 epochs worth of steps)
        figsize: Figure size for the plot
        save_path: Path to save the figure (optional)
    
    Returns:
        Matplotlib figure
    """
    # Calculate total steps
    steps_per_epoch = math.ceil(train_examples / batch_size)
    total_steps = steps_per_epoch * num_epochs
    
    # Calculate max_cycle_length if not provided (2 epochs worth of steps)
    if max_cycle_length is None:
        max_cycle_length = 2 * steps_per_epoch
    
    # Create a dummy model and optimizer
    dummy_model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(dummy_model.parameters(), lr=initial_lr)
    
    # Initialize the scheduler
    scheduler = ProtXScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        initial_lr=initial_lr,
        max_lr=max_lr,
        min_lr=min_lr,
        T_0=T_0,
        T_mult=T_mult,
        gamma=gamma,
        max_lr_threshold=max_lr_threshold,
        max_cycle_length=max_cycle_length
    )
    
    # Simulate the scheduler
    lr_values = []
    for step in range(total_steps):
        lr_values.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    # Create x-axis values (steps and epochs)
    steps = np.arange(total_steps)
    epochs = steps / steps_per_epoch
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Plot learning rate vs steps
    plt.plot(steps, lr_values)
    
    # Set Y-axis to logarithmic scale
    plt.yscale('log')
    
    # Add epoch x-axis at the top
    ax1 = plt.gca()
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(np.arange(0, num_epochs + 1, max(1, num_epochs // 10)) * steps_per_epoch)
    ax2.set_xticklabels(np.arange(0, num_epochs + 1, max(1, num_epochs // 10)))
    
    # Add markers for warmup end and cycle restarts
    if warmup_steps > 0:
        plt.axvline(x=warmup_steps, color='r', linestyle='--', alpha=0.5, label=f'Warmup End')
    
    # Mark cycle restarts
    current_step = warmup_steps
    current_T = T_0
    cycle = 0
    while current_step < total_steps:
        # Apply max cycle length constraint
        if max_cycle_length is not None:
            current_T = min(current_T, max_cycle_length)
        
        current_step += current_T
        if current_step < total_steps:
            plt.axvline(x=current_step, color='g', linestyle='--', alpha=0.5, 
                        label=f'Cycle End' if cycle == 0 else None)
            cycle += 1
            next_T = current_T * T_mult
            # Apply max cycle length constraint to next cycle too
            if max_cycle_length is not None:
                next_T = min(next_T, max_cycle_length)
            current_T = next_T
    
    # Set plot details
    plt.xlabel('Steps')
    plt.ylabel('Learning Rate')
    plt.title('ProtXScheduler Learning Rate Schedule')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    # Add annotations
    plt.annotate(f'Warmup Steps: {warmup_steps}', xy=(0.02, 0.95), xycoords='axes fraction')
    plt.annotate(f'Initial LR: {initial_lr}', xy=(0.02, 0.90), xycoords='axes fraction')
    plt.annotate(f'Max LR: {max_lr}', xy=(0.02, 0.85), xycoords='axes fraction')
    plt.annotate(f'Min LR: {min_lr}', xy=(0.02, 0.80), xycoords='axes fraction')
    plt.annotate(f'T_0: {T_0}', xy=(0.02, 0.75), xycoords='axes fraction')
    plt.annotate(f'T_mult: {T_mult}', xy=(0.02, 0.70), xycoords='axes fraction')
    plt.annotate(f'Gamma: {gamma}', xy=(0.02, 0.65), xycoords='axes fraction')
    if max_lr_threshold is not None:
        plt.annotate(f'Max LR Threshold: {max_lr_threshold}', xy=(0.02, 0.60), xycoords='axes fraction')
    plt.annotate(f'Max Cycle Length: {max_cycle_length} steps ({max_cycle_length/steps_per_epoch:.1f} epochs)', 
                 xy=(0.02, 0.55), xycoords='axes fraction')
    
    # Add secondary axis for epochs
    ax2.set_xlabel('Epochs')
    
    # Save if requested
    if save_path:
        plt.savefig(save_path)
    
    return plt.gcf()

if __name__ == "__main__":
    visualize_lr_schedule(
        train_examples=5000 * 0.9,
        batch_size=64,
        num_epochs=15,
        warmup_steps=500,
        initial_lr=1e-8,
        max_lr=3e-3,
        min_lr=1e-5,
        T_0=300,
        T_mult=1.25,
        gamma=0.55,
        max_cycle_length=2000,
        save_path="lr_schedule14.png"
    )
