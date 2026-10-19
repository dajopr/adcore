"""``admaint-run``: maintenance experiments from Hydra configs.

    admaint-run scenario=contamination
    admaint-run scenario=extension scenario.categories=[bottle] scenario.n_update=[1,4]
    admaint-run scenario=drift device=2
    admaint-run -m scenario=contamination,extension,stepwise,drift   # one after another

The config has a ``model`` (as for ``adcore-run``, but its module must be
`admaint.Maintainable`) and a ``scenario`` (dataset, kind, grid); ``admaint/conf/admaint.yaml``
documents the top-level keys. Each (scenario, model, variant) writes to
``runs/<scenario.name>/<model.name>[/<variant>]``, and rerunning resumes it. Runs go one
after another on one device.

Needs the ``run`` extra (``hydra-core``).
"""

from __future__ import annotations

import argparse
import hashlib
import json

import pandas as pd
from omegaconf import DictConfig, OmegaConf

# Also registers the `adcore.variant` / `adcore.subdir` resolvers used by admaint.yaml.
from adcore.runner import (
    ModuleFactory,
    _build,
    _check_resume,
    _patch_hydra_args_parser,
    _plain,
    _spec,
    _tracking,
)
from admaint.experiment import MaintenanceExperiment

# Scenario keys that may change between sessions of one output directory: the grid
# grows, and loader settings don't change results.
_UNHASHED = (
    "name", "categories", "target_defects", "shots", "n_contam", "n_update", "shifts",
    "seeds", "update_seeds", "min_eval", "batch_size", "num_workers",
)


def config_hash(cfg: DictConfig) -> str:
    """Hash of the settings that change results: the model, and the scenario without its grid.

    Shifts are grid entries, known by name: give a changed shift a new name.
    """
    model = OmegaConf.to_container(cfg.model, resolve=True)
    scenario = {
        k: v
        for k, v in OmegaConf.to_container(cfg.scenario, resolve=True).items()
        if k not in _UNHASHED
    }
    text = json.dumps({"model": model, "scenario": scenario}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def build_experiment(cfg: DictConfig, *, tracking: bool = True) -> MaintenanceExperiment:
    """The `MaintenanceExperiment` a resolved config describes; nothing is loaded until it runs."""
    model, scenario = cfg.model, cfg.scenario
    trainer = {**_plain(scenario.get("trainer")), **_plain(model.get("trainer"))}
    if cfg.get("device") is not None:
        trainer["devices"] = [cfg.device]

    def listed(key: str, default=None):
        value = scenario.get(key, default)
        return list(value) if value is not None else None

    shifts = scenario.get("shifts") or {}
    return MaintenanceExperiment(
        scenario.root,
        module_factory=ModuleFactory(model),
        scenario=scenario.kind,
        categories=listed("categories"),
        target_defects=listed("target_defects"),
        shots=listed("shots", [8]),
        n_contam=listed("n_contam", [0]),
        n_update=listed("n_update", [1]),
        steps=listed("steps", []),
        shifts={name: _build(node) for name, node in shifts.items()},
        retire=scenario.get("retire", False),
        seeds=listed("seeds", [0, 1, 2]),
        update_seeds=listed("update_seeds", [0]),
        cache_fits=cfg.get("cache_fits", True),
        min_eval=scenario.get("min_eval", 3),
        n_calibration=scenario.get("n_calibration", 20),
        calibration_fpr=scenario.get("calibration_fpr", 0.05),
        output_dir=cfg.output_dir,
        trainer_kwargs=trainer,
        image_size=tuple(model.image_size),
        transform=_build(model.get("transform")),
        train_transform=_build(model.get("train_transform")),
        batch_size=scenario.get("batch_size", 32),
        num_workers=scenario.get("num_workers", 4),
        spec=_spec(scenario.get("spec", "mvtec")),
        tracking=_tracking(cfg) if tracking else None,
        metadata={"config_hash": config_hash(cfg)},
    )


def run(cfg: DictConfig) -> pd.DataFrame:
    """Run the experiment ``cfg`` describes; call it from your own ``@hydra.main`` if you like."""
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    experiment = build_experiment(cfg)
    _check_resume(experiment, cfg.get("force", False))
    return experiment.run(progress=cfg.get("progress", True))


def main() -> None:
    import hydra

    if hasattr(argparse.ArgumentParser, "_check_help"):
        _patch_hydra_args_parser()
    hydra.main(config_path="conf", config_name="admaint", version_base="1.3")(run)()
