"""
Canonical dataset URL mappings used by the CLI.

Users can supply a known dataset key (e.g., "uniref50") or override with
an explicit --url. Keeping mappings centralized here makes it easy to add
new sources later without changing CLI logic.
"""

from typing import Dict, Optional


# Minimal, curated defaults. Extend this mapping over time as needed.
DATASET_URLS: Dict[str, str] = {
    "swissprot": "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.fasta.gz",
    "trembl": "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_trembl.fasta.gz",
    # UniRef clusters
    "uniref50": "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz",
    "uniref90": "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz",
    "uniref100": "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref100/uniref100.fasta.gz",
}


def get_supported_datasets() -> Dict[str, str]:
    return dict(DATASET_URLS)
