from pathlib import Path

class DistillConfig:
    def __init__(self):
        self.base_dir = Path("output")
        self.trained_protx_dir = self.base_dir / "protx"
        self.checkpoint_dir = self.base_dir / "checkpoints"
        self.plots_dir = self.base_dir / "plots"

        self.batch_size = 32
        self.lr = 3e-5
        self.num_epochs = 10

        self.student_embed_dim = 512
        self.student_num_layers = 6
        self.student_num_heads = 8
