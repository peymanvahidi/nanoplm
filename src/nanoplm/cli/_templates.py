"""
YAML template strings emitted by `nanoplm data get-yaml`. Two templates
each for data.yaml and dvc.yaml — pretrain and distillation variants.
"""

PRETRAIN_DATA_YAML = """\
data_dirs:
  url: "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.fasta.gz"
  local_fasta: null   # optional local .fasta or .fasta.gz file (skips network download)
  # swissprot: "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.fasta.gz"
  # trembl:    "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_trembl.fasta.gz"
  # uniref50:  "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"
  # uniref90:  "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz"
  # uniref100: "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref100/uniref100.fasta.gz"

  compressed_fasta:   "output/data/raw/uniref50.fasta.gz"
  extracted_fasta:    "output/data/raw/uniref50.fasta"
  validation_dir:     "output/data/validation"
  shuffled_fasta:     "output/data/raw/uniref50_shuffled.fasta"
  filtered_fasta:     "output/data/filter/uniref50_filtered.fasta"
  splitted_fasta_dir: "output/data/split"
  output_dir:         "output/data/pretrain_data"

data_params:
  # FASTA pipeline
  seqs_num: 20000
  min_seq_len: 20
  max_seq_len: 512
  val_ratio: 0.1
  shuffle: true
  shuffle_seed: 24
  drop_invalid_sequences: false

  # Shard generation
  samples_per_shard: 2000
  max_workers: 2
  register_tokens: 0
"""

DISTILL_DATA_YAML = """\
data_dirs:
  url: "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.fasta.gz"
  local_fasta: null
  # swissprot: "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.fasta.gz"
  # trembl:    "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_trembl.fasta.gz"
  # uniref50:  "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"
  # uniref90:  "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz"
  # uniref100: "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref100/uniref100.fasta.gz"

  compressed_fasta:   "output/data/raw/uniref50.fasta.gz"
  extracted_fasta:    "output/data/raw/uniref50.fasta"
  validation_dir:     "output/data/validation"
  shuffled_fasta:     "output/data/raw/uniref50_shuffled.fasta"
  filtered_fasta:     "output/data/filter/uniref50_filtered.fasta"
  splitted_fasta_dir: "output/data/split"
  output_dir:         "output/data/distillation_data"

data_params:
  seqs_num: 20000
  min_seq_len: 20
  max_seq_len: 512
  val_ratio: 0.1
  shuffle: true
  shuffle_seed: 24
  drop_invalid_sequences: false
  device: "auto"

  teacher_model: "prott5"
  samples_per_shard: 2000
  embed_calc_batch_size: 4
"""
