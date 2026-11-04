"""Derivasi label per skema decoding (config-driven).

Target PRIMER penelitian = transfer learning lintas-dataset untuk beban kognitif,
sehingga label harus sebanding antar COG-BCI & STEW. Skema:

- ``workload_binary``  : rendah(0) vs tinggi(1) — selaras STEW lo/hi (target transfer).
- ``nback_graded``     : 0/1/2-back (within COG-BCI).
- ``matb_graded``      : easy/med/diff (within COG-BCI).

Setiap fungsi mengembalikan label int untuk satu rekaman, atau ``None`` bila task
tidak termasuk dalam skema (rekaman dilewati untuk skema tsb).
"""
from __future__ import annotations


def label_for(scheme_cfg: dict, dataset: str, task: str) -> int | None:
    """Map (dataset, task) -> label int sesuai skema, atau None bila tak relevan.

    ``task`` untuk COG-BCI = nama task EEGLAB (mis. ``twoBACK``, ``MATBdiff``,
    ``RS_Beg_EC``). Untuk STEW = ``lo`` / ``hi``.
    """
    if dataset not in scheme_cfg.get("datasets", []):
        return None
    mapping = scheme_cfg.get(dataset)
    if mapping is None:
        return None

    # Bentuk 1: dict langsung task->int (mis. nback_graded, STEW lo/hi)
    if all(isinstance(v, int) for v in mapping.values()):
        return mapping.get(task)

    # Bentuk 2: dict kelas->list task (mis. workload_binary COG-BCI low/high)
    class_order = {"low": 0, "high": 1}
    for cls, tasks in mapping.items():
        if task in tasks:
            return class_order.get(cls, list(mapping).index(cls))
    return None
