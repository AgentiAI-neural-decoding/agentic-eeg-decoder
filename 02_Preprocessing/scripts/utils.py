"""Utilitas bersama untuk pipeline preprocessing COG-BCI & STEW."""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(LOG_FORMAT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


@dataclass
class ICReport:
    subject: str
    session: str
    task: str
    n_components: int
    rejected_indices: list[int] = field(default_factory=list)
    labels: dict[int, str] = field(default_factory=dict)
    probabilities: dict[int, float] = field(default_factory=dict)
    bad_channels: list[str] = field(default_factory=list)
    n_epochs_kept: int = -1
    n_epochs_total: int = -1
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "session": self.session,
            "task": self.task,
            "n_components": self.n_components,
            "rejected_indices": self.rejected_indices,
            "labels": self.labels,
            "probabilities": self.probabilities,
            "bad_channels": self.bad_channels,
            "n_epochs_kept": self.n_epochs_kept,
            "n_epochs_total": self.n_epochs_total,
            "notes": self.notes,
        }


def write_report_json(report: ICReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2))


def save_qc_psd(raw, png_path: Path, title: str, fmax: float = 50.0) -> None:
    """Simpan plot PSD untuk QC visual."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig = raw.compute_psd(fmax=fmax).plot(show=False)
    fig.suptitle(title)
    fig.savefig(png_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
