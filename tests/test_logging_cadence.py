from nanoplm.pretraining.config import PretrainingConfig
from tools.schema_gen import templates


def test_logging_steps_default_is_25():
    assert PretrainingConfig(dataset_dir="/tmp").logging_steps == 25


def test_eval_steps_default_is_a_multiple_of_logging_steps_default():
    cfg = PretrainingConfig(dataset_dir="/tmp")
    assert cfg.eval_steps % cfg.logging_steps == 0, (
        "eval_steps must be a multiple of logging_steps or the vram=/peak= "
        "fields never get logged and the peak counter never resets"
    )


def test_template_logging_steps_matches_dataclass_default_and_divides_eval_steps():
    fields = templates.parse_fields(templates.load_template_text("pretrain"))
    by_key = {(f.section, f.key): f for f in fields}

    logging_steps = int(by_key[("pretraining", "logging_steps")].raw_default)
    eval_steps = int(by_key[("pretraining", "eval_steps")].raw_default)

    assert logging_steps == PretrainingConfig(dataset_dir="/tmp").logging_steps
    assert eval_steps % logging_steps == 0
