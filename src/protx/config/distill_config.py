from pathlib import Path

class DistillConfig:
    def __init__(self):
        self.base_dir = Path("output")
        self.trained_protx_dir = self.base_dir / "protx"
        self.checkpoint_dir = self.base_dir / "checkpoints"
        self.plots_dir = self.base_dir / "plots"

        self.num_epochs = 9
        self.batch_size = 32

        # Scheduler
        self.warmup_steps = 1000
        self.start_lr = 1e-8

        self.min_lr = 2e-6
        self.max_lr = 1e-3
        self.T_0 = 1000
        self.T_mult = 1.25
        self.gamma = 0.5

        self.student_embed_dim = 256
        self.student_num_layers = 6
        self.student_num_heads = 4
