# src/nanoplm/data/preparation/checker.py
from dataclasses import dataclass
from pathlib import Path
from nanoplm.data.preparation._native_ops import (
    validate_fasta_native,
    NativeValidateResult,
)
from nanoplm.utils import logger, create_dirs


@dataclass(frozen=True)
class CheckResult:
    total_sequences: int
    invalid_count: int
    empty_count: int
    trailing_stop_count: int = 0


class Checker:
    """
    Validates a FASTA file. Always writes three sidecar files into output_dir:
      - validation.json   : machine-readable summary
      - invalid.fasta     : full records of problematic sequences
      - invalid_ids.txt   : one ID per line

    By default (drop_invalid=True), invalid_ids.txt is populated with the IDs
    of every invalid sequence so the downstream filter stage can exclude them.
    When drop_invalid=False, the file is left empty so downstream stages don't
    accidentally drop anything; invalid.fasta still contains the problem
    records for human inspection.
    """

    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        # Default True: the common case is "tell the next stage to drop these".
        drop_invalid: bool = True,
    ):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.drop_invalid = drop_invalid

    def check(self) -> CheckResult:
        create_dirs(self.output_dir)
        report_json = self.output_dir / "validation.json"
        invalid_fasta = self.output_dir / "invalid.fasta"
        invalid_ids = self.output_dir / "invalid_ids.txt"

        native_result = validate_fasta_native(
            input_path=self.input_path,
            invalid_fasta_path=invalid_fasta,
            invalid_ids_path=invalid_ids,
            report_json_path=report_json,
        )

        first_ids = _first_ids(invalid_ids) if native_result.invalid_count > 0 else []
        # If drop_invalid is False, blank out invalid_ids.txt so the filter
        # stage never excludes anything. The invalid.fasta sidecar still
        # contains the problem records for human inspection.
        if not self.drop_invalid and native_result.invalid_count > 0:
            invalid_ids.write_text("")

        result = CheckResult(
            total_sequences=native_result.total_sequences,
            invalid_count=native_result.invalid_count,
            empty_count=native_result.empty_count,
            trailing_stop_count=native_result.trailing_stop_count,
        )

        n_stop = result.trailing_stop_count
        if n_stop > 0:
            logger.warning(
                f"check: removing the trailing '*' (stop codon) from {n_stop:,} "
                f"record{'s' if n_stop != 1 else ''} "
                f"(first: {', '.join(native_result.first_trailing_stop_ids)})"
            )

        if result.invalid_count > 0 and self.drop_invalid:
            logger.warning(
                f"check: dropping {result.invalid_count:,} "
                f"record{'s' if result.invalid_count != 1 else ''} with characters outside "
                f"the amino-acid alphabet or empty (first: {', '.join(first_ids)}); "
                f"see {invalid_fasta}"
            )
        elif result.invalid_count > 0:
            logger.warning(
                f"check: found {result.invalid_count} invalid sequence(s); "
                f"see {invalid_fasta}"
            )
        else:
            logger.info(
                f"check: validated {result.total_sequences} sequences; no problems found"
            )
        return result


def _first_ids(ids_path: Path, n: int = 3) -> list:
    """The first n ids in invalid_ids.txt (one per line)."""
    out = []
    with open(ids_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                out.append(line.strip())
                if len(out) == n:
                    break
    return out
