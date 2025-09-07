nanoplm train student \
    --protx-train-prefix output/data/kd_dataset/train/train_kd_dataset.h5 \
    --protx-val-prefix output/data/kd_dataset/val/val_kd_dataset.h5 \
    --train-file output/data/split/train.fasta \
    --val-file output/data/split/val.fasta \
    --wandb-dir output/wandb_checkpoints \
    --project-name protx_distillation \
    --student-embed-dim 256 \
    --student-num-layers 4 \
    --student-num-heads 8 \
    --num-epochs 20 \
    --batch-size 16 \
    --max-lr 0.001 \
    --max-seq-len 512 \
    --max-seqs-num 100 \
    --max-grad-norm 100.0 \
    --val-ratio 0.1 \
    --num-workers 0 \
    --sharded
    # --on-the-fly
