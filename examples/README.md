# nanoPLM quickstart

`nanoplm_quickstart.ipynb` loads an exported nanoPLM protein language model,
tokenizes sequences, computes mean-pooled embeddings, and visualizes a public
subcellular-localization dataset with UMAP.

## Run it

```
pip install "transformers>=5" torch umap-learn matplotlib datasets
```

Open the notebook and change `REPO` in the first code cell to your published
model id (or a local export directory), then run all cells.

**No `nanoplm` install is needed.** The export ships its own modeling code and
loads through `transformers` alone with `trust_remote_code=True`. It needs
`transformers>=5`; transformers 4.x fails to load it.

## Device support

Verified on CPU and CUDA (Hopper / H100). MPS (Apple Silicon) and other CUDA
architectures are expected to work -- the export runs a portable pure-PyTorch
forward path with no custom kernels -- but are not verified.
