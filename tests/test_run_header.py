from nanoplm.pretraining.pure_pipeline import (
    _RunHeader, _RunProgress, _format_run_header, _format_run_exit, _format_token_count,
    _node_count,
)

RUN_ROOT = "output/checkpoints/loop-imp-abl-baseline-l10/loop-imp-abl-baseline-l10-1684851-3"


def _reference_header(**over) -> _RunHeader:
    fields = dict(
        run_name="loop-imp-abl-baseline-l10-1684851-3",
        resume_step=None,
        job_id="1684851",
        git_sha="d3c5e79",
        wandb_run_id="3k2j4h5",
        run_root=RUN_ROOT,
        dataset_dir="output/data/pretrain_data",
        train_seqs=52479582,
        val_seqs=65681,
        max_seq_len=512,
        world_size=16,
        nodes=4,
        dist_mode="fsdp",
        global_batch_tokens=1048576,
        grad_accum=1,
        total_steps=190735,
        total_tokens=200000143360,
        warmup_steps=1000,
        logging_steps=25,
        eval_steps=1000,
        save_steps=2500,
        stop="none configured; the job's SLURM time limit will kill this run",
    )
    fields.update(over)
    return _RunHeader(**fields)


def test_token_counts():
    assert _format_token_count(4_500_000_000_000) == "4.50T"
    assert _format_token_count(200_000_143_360) == "200.0B"
    assert _format_token_count(55_898_238_976) == "55.9B"
    assert _format_token_count(3_500_000) == "3.5M"
    assert _format_token_count(1_048_576) == "1.0M"
    assert _format_token_count(999_999) == "999,999"


def test_header_renders_a_real_run():
    text = _format_run_header(_reference_header())
    lines = text.split("\n")

    assert len(lines) == 8
    assert lines[0] == "=" * 80
    assert lines[-1] == "=" * 80

    for line in lines[1:-1]:
        assert line[:8] == line[:8].rstrip().ljust(8)
        assert line[8] != " "
        assert line[:8].rstrip()

    assert max(len(line) for line in lines) <= 200

    assert "run     loop-imp-abl-baseline-l10-1684851-3   job=1684851 code=d3c5e79 wandb=3k2j4h5" in text
    assert "data    output/data/pretrain_data   train_seqs=52,479,582 val_seqs=65,681 max_len=512" in text
    assert "scale   ranks=16 nodes=4 mode=fsdp   global_batch=1,048,576 tok grad_accum=1" in text
    assert "budget  total_steps=190,735 = 200.0B tokens   warmup=1,000   log/eval/save=25/1000/2500" in text


def test_header_marks_a_resume():
    text = _format_run_header(_reference_header(resume_step=52500))
    assert "run     loop-imp-abl-baseline-l10-1684851-3 (resume from step 52,500)   job=1684851" in text


def test_exit_line_for_a_wallclock_stop():
    text = _format_run_exit(
        _RunProgress(
            total_steps=190735,
            run_root=RUN_ROOT,
            wandb_run_id="3k2j4h5",
            step=53307,
            tokens=55898238976,
            reason="wallclock",
            last_checkpoint="checkpoint-53300",
        ),
        wall_sec=10479.6,
    )
    lines = text.split("\n")

    assert len(lines) == 4
    assert lines[0] == "=" * 80
    assert lines[-1] == "=" * 80
    assert lines[1] == (
        "exit    reason=wallclock step=53,307/190,735 (27.9%) wall=2.91h tokens=55.9B"
    )
    assert lines[2].endswith("   last=checkpoint-53300   wandb=3k2j4h5")


def test_exit_line_without_a_checkpoint():
    text = _format_run_exit(
        _RunProgress(
            total_steps=190735,
            run_root=RUN_ROOT,
            wandb_run_id="3k2j4h5",
            step=190735,
            tokens=200000143360,
            reason="complete",
        ),
        wall_sec=36000.0,
    )
    assert "last=" not in text


def test_node_count_prefers_slurms_own_count(monkeypatch):
    monkeypatch.setenv("SLURM_NNODES", "4")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    assert _node_count(16) == 4


def test_node_count_falls_back_to_local_world_size(monkeypatch):
    monkeypatch.delenv("SLURM_NNODES", raising=False)
    monkeypatch.delenv("SLURM_JOB_NUM_NODES", raising=False)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    assert _node_count(16) == 4


def test_node_count_is_one_when_the_launcher_says_nothing(monkeypatch):
    for key in ("SLURM_NNODES", "SLURM_JOB_NUM_NODES", "LOCAL_WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)
    assert _node_count(16) == 1


def test_node_count_ignores_a_malformed_value(monkeypatch):
    monkeypatch.setenv("SLURM_NNODES", "four")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    assert _node_count(16) == 4
