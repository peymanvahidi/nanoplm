# ProtX - Protein Language Model with Knowledge Distillation

## MLOps Pipeline

This project uses DVC (Data Version Control) to manage the end-to-end MLOps pipeline for training protein language models.

### Pipeline Stages

1. **Data Download**: Downloads the UniRef50 dataset
2. **Data Extraction**: Extracts the compressed data 
3. **Filter & Split**: Filters sequences by length and splits into train/validation sets
4. **Dataset Creation**: Creates H5 datasets ready for model training
5. **Model Training**: Trains the student model using knowledge distillation

### Running the Pipeline

#### Complete Pipeline

To run the complete pipeline end-to-end (pre compute embeddings):

```bash
dvc repro
```

To run the completel pipeline with the calculation of the embeddings on-the-fly:

```bash
dvc repro --no-exec save_protx_dataset
```

- Also in params.yaml, set `distill_params.on_the_fly=true`

This will execute all stages in the pipeline in the correct order. DVC automatically skips stages where inputs haven't changed.

#### Individual Stages

You can also run individual stages:

```bash
# Run only the data download stage
dvc repro download

# Run up to the dataset creation stage
dvc repro save_protx_dataset

# Run only the model training stage
dvc repro train_model
```

### Pipeline Parameters

All pipeline parameters are defined in `params.yaml`:

- Data processing parameters (sequence length, validation split ratio, etc.)
- Model architecture parameters (embedding dimensions, number of layers, etc.)
- Training parameters (learning rate, batch size, epochs, etc.)

To modify pipeline parameters, edit the `params.yaml` file and then rerun the pipeline.

### Experiment Tracking

The pipeline integrates with Weights & Biases for experiment tracking. Model checkpoints and training metrics are saved automatically.

## Development

### Adding New Pipeline Stages

To add a new stage to the pipeline:

1. Create the necessary processing code in the `src/protx` directory
2. Add a new command-line argument in `main.py`
3. Define the new stage in `dvc.yaml` with appropriate dependencies and outputs
4. Update the training parameters in `params.yaml` if needed
