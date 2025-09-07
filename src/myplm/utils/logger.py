import logging
import sys
from pathlib import Path

# Create logs directory if it doesn't exist
(Path("output") / "logs").mkdir(parents=True, exist_ok=True)

# Set up file handler for all logs in a unique file per run with timestamp
log_file = f'output/logs/pipeline.log'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Set up console handler with a standard formatter
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(console_formatter)

# Configure the logger
logger = logging.getLogger("myplm")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Prevent propagation to avoid duplicate logs
logger.propagate = False

# Add a helper method to log stage separators
def log_stage(stage_name):
    separator = "="*20
    logger.info(f"{separator}STAGE: {stage_name}{separator}")
