from pathlib import Path

class DistillConfig:
    def __init__(self):
        self.base_dir = Path("output")
        self.trained_protx_dir = self.base_dir / "protx"

        self.batch_size = 32
        self.lr = 0.001
        self.num_epochs = 5