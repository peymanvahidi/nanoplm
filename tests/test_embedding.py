import sys
import os
# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.protx.distillation.teacher_embed import TeacherEmbedder
from src.protx.data.pipeline import DataPipeline
from pathlib import Path
from tests.analyze_embedding import analyze_embedding
output_dir = Path("testing_dir")
custom_url = "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=accession%3AA0A0C5B5G6+OR+accession%3AA0A1B0GTW7+OR+accession%3AA1A519+OR+accession%3AA6NFY7+OR+accession%3AO60939+OR+accession%3AO75838+OR+accession%3AO77932+OR+accession%3AO95139+OR+accession%3AO95154+OR+accession%3AO95298"

# Create pipeline with custom directory and URL
data_pipeline = DataPipeline(pipeline_output_dir=output_dir, uniref50_url=custom_url)

data_pipeline.run_download()

data_pipeline.run_extract()

data_pipeline.run_filter()

data_pipeline.run_split()

data_pipeline.run_tokenize()

# Create TeacherEmbedder with correct parameter order
test_embedder = TeacherEmbedder(model_name="Rostlab/prot_t5_xl_uniref50", output_dir=output_dir)

test_embedder.process_all()

analyze_embedding()