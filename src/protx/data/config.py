import yaml
from pathlib import Path

# ------------------------
# DO NOT MODIFY THIS FILE!
# ------------------------

BASE_DIR = Path("output")

RAW_DIR = BASE_DIR / "data/raw"
PROCESSED_DIR = BASE_DIR / "data/processed"

UNIREF50_URL = "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.fasta.gz"
UNIREF50_FASTA_GZ = RAW_DIR / "uniref50.fasta.gz"
UNIREF50_FASTA = RAW_DIR / "uniref50.fasta"

FILTERED_SEQS = PROCESSED_DIR / "uniref50_filtered.fasta"
TRAIN_FILE = PROCESSED_DIR / "train.fasta"
VAL_FILE = PROCESSED_DIR / "val.fasta"
INFO_FILE = PROCESSED_DIR / "dataset_info.txt"

yaml_path = Path(__file__).parent / "params.yaml"
with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

VAL_RATIO = data["val_ratio"]
MAX_SEQS_NUM = data["max_seqs_num"]
MIN_SEQ_LEN = data["min_seq_len"]
MAX_SEQ_LEN = data["max_seq_len"]
BATCH_SIZE = data["batch_size"]
