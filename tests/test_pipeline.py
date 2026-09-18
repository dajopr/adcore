"""Few-shot sampling, the Lightning evaluation, PatchCore, a gradient-trained module and
the experiment runner, end to end on a tiny synthetic MVTec tree: gray noise frames,
with bright squares as the defects."""

from __future__ import annotations

import json
import warnings

import lightning as L
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from adcore import (
    AnomalyModule,
    CachedDataset,
    FewShotExperiment,
    MVTecDataModule,
    MVTecDataset,
    PatchCore,
    PatchcoreModel,
    evaluate,
    few_shot_subset,
    summarize,
    table,
)
from adcore.patchcore import reshape_embedding

SIZE = 32
CPU = {"accelerator": "cpu", "devices": 1}


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    # Lightning writes default checkpoints relative to the working directory.
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _write(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("mvtec")
    rng = np.random.default_rng(0)

    def good():
        return rng.integers(90, 110, size=(SIZE, SIZE, 3), dtype=np.uint8)

    for category in ("alpha", "beta"):
        for i in range(6):
            _write(root / category / "train" / "good" / f"{i:03d}.png", good())
        for i in range(3):
            _write(root / category / "test" / "good" / f"{i:03d}.png", good())
        for defect, (y, x) in {"spot": (4, 4), "blob": (18, 12)}.items():
            for i in range(3):
                image, mask = good(), np.zeros((SIZE, SIZE), dtype=np.uint8)
                image[y + i : y + i + 8, x : x + 8] = 250
                mask[y + i : y + i + 8, x : x + 8] = 255
                _write(root / category / "test" / defect / f"{i:03d}.png", image)
                _write(
                    root / category / "ground_truth" / defect / f"{i:03d}_mask.png", mask
                )
    return root


def datamodule(root, category="alpha", **kwargs):
    return MVTecDataModule(
        root, category, image_size=(SIZE, SIZE), batch_size=4, num_workers=0, **kwargs
    )


class Brightness(AnomalyModule):
    """Zero-shot: scores each pixel by its brightness, perfect on the synthetic defects."""

    def forward(self, images):
        anomaly_map = images.mean(dim=1, keepdim=True)
        return {"anomaly_map": anomaly_map, "pred_score": anomaly_map.amax(dim=(1, 2, 3))}


class Segmenter(AnomalyModule):
    """Gradient-trained: a 1x1 conv fit to the masks of anomalous training frames."""

    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, images):
        anomaly_map = torch.sigmoid(self.head(images))
        return {"anomaly_map": anomaly_map, "pred_score": anomaly_map.amax(dim=(1, 2, 3))}

    def training_step(self, batch, batch_idx):
        logits = self.head(batch["image"])
        loss = F.binary_cross_entropy_with_logits(logits, batch["mask"])
        self.log("train/loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)


def pixel_extractor(images):
    """Raw colour at 1/4 resolution, a stand-in backbone."""
    return F.avg_pool2d(images, 4)


# --- data ------------------------------------------------------------------------


def test_few_shot_is_nested_deterministic_and_per_category(root):
    both = MVTecDataset(root, split="train", image_size=(SIZE, SIZE))
    alpha = MVTecDataset(root, "alpha", split="train", image_size=(SIZE, SIZE))

    def paths(subset):
        return [subset.dataset.records[i]["image_path"] for i in subset.indices]

    one, two = few_shot_subset(both, 1, seed=3), few_shot_subset(both, 2, seed=3)
    assert len(one) == 2 and len(two) == 4
    assert set(paths(one)) <= set(paths(two))
    assert paths(one) == paths(few_shot_subset(both, 1, seed=3))
    # A category's draw does not depend on which other categories are loaded.
    assert set(paths(few_shot_subset(alpha, 2, seed=3))) <= set(paths(two))
    assert len(few_shot_subset(both, None)) == 12

    draws = {tuple(paths(few_shot_subset(alpha, 2, seed=s))) for s in range(10)}
    assert len(draws) > 1

    with pytest.raises(ValueError, match="7-shot"):
        few_shot_subset(alpha, 7)


def test_datamodule_moves_anomalous_frames_out_of_test(root):
    test = MVTecDataset(root, "alpha", split="test", image_size=(SIZE, SIZE))
    cached = CachedDataset(test)
    assert len(cached) == 9 and cached.records == test.records

    dm = datamodule(root, shots=2, anomalous_fraction=1 / 3, test_dataset=cached)
    dm.setup()
    assert len(dm.reference) == 2
    train_labels = [item["label"] for item in dm.train]
    assert sorted(train_labels) == [0, 0, 1, 1]  # one of three frames per defect type
    test_paths = {item["image_path"] for item in dm.test}
    train_paths = {item["image_path"] for item in dm.train}
    assert len(test_paths) == 7 and not test_paths & train_paths


# --- evaluation ------------------------------------------------------------------


def test_evaluate_scores_per_category_and_defect(root):
    test = MVTecDataset(root, split="test", image_size=(SIZE, SIZE))
    result = evaluate(Brightness(), DataLoader(test, batch_size=4), **CPU)

    metrics = result.metrics.set_index(["category", "defect_type"])
    assert set(metrics.index) == {
        (c, d) for c in ("alpha", "beta") for d in ("all", "blob", "spot")
    }
    assert metrics.loc[("alpha", "all"), "n_images"] == 9
    assert metrics.loc[("alpha", "spot"), "n_images"] == 6
    assert metrics.loc[("alpha", "spot"), "n_anomalous"] == 3
    for metric in ("image_auroc", "pixel_auroc", "pixel_aupr", "aupro"):
        np.testing.assert_allclose(metrics[metric], 1.0, atol=1e-6)

    assert len(result.overall) == 2
    assert result.predictions.anomaly_maps.shape == (18, SIZE, SIZE)


def test_evaluate_resizes_maps_to_the_masks(root):
    class LowRes(Brightness):
        def forward(self, images):
            output = super().forward(images)
            output["anomaly_map"] = F.avg_pool2d(output["anomaly_map"], 2)[:, 0]
            return output

    result = evaluate(LowRes(), datamodule(root), **CPU)
    assert result.predictions.anomaly_maps.shape == (9, SIZE, SIZE)
    assert result.overall.loc[0, "pixel_auroc"] > 0.95


def test_test_metrics_are_logged_overall_and_per_category(root):
    trainer = L.Trainer(logger=False, enable_progress_bar=False, **CPU)
    dm = MVTecDataModule(root, None, image_size=(SIZE, SIZE), num_workers=0)
    (logged,) = trainer.test(Brightness(), datamodule=dm, verbose=False)
    assert logged["test/image_auroc"] == pytest.approx(1.0)
    assert logged["test/beta/aupro"] == pytest.approx(1.0)


# --- PatchCore -------------------------------------------------------------------


@pytest.mark.parametrize("sampling_ratio", [1.0, 0.5, 10])
def test_patchcore_fits_and_detects(root, sampling_ratio):
    module = PatchCore(pixel_extractor, sampling_ratio=sampling_ratio, blur_sigma=1.0)
    # Anomalous training frames must be kept out of the memory bank.
    dm = datamodule(root, shots=2, anomalous_fraction=1 / 3)
    trainer = L.Trainer(max_epochs=1, logger=False, enable_progress_bar=False, **CPU)
    trainer.fit(module, datamodule=dm)

    patches = 2 * (SIZE // 4) ** 2
    expected = {1.0: patches, 0.5: patches // 2, 10: 10}[sampling_ratio]
    assert module.model.memory_bank.shape == (expected, 3)

    trainer.test(module, datamodule=dm, verbose=False)
    assert module.test_result.overall.loc[0, "image_auroc"] == pytest.approx(1.0)
    assert module.test_result.overall.loc[0, "pixel_auroc"] > 0.9


def test_patchcore_memory_bank_round_trips_through_a_checkpoint(root, tmp_path):
    module = PatchCore(pixel_extractor, sampling_ratio=1.0)
    trainer = L.Trainer(max_epochs=1, logger=False, enable_progress_bar=False, **CPU)
    trainer.fit(module, datamodule=datamodule(root, shots=1))
    trainer.save_checkpoint(tmp_path / "patchcore.ckpt")

    restored = PatchCore(pixel_extractor, sampling_ratio=1.0)
    state = torch.load(tmp_path / "patchcore.ckpt", weights_only=False)["state_dict"]
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.model.memory_bank, module.model.memory_bank)


# --- gradient training -----------------------------------------------------------


def test_gradient_training_with_validation(root):
    module = Segmenter()
    dm = datamodule(root, shots=6, anomalous_fraction=2 / 3)
    trainer = L.Trainer(
        max_epochs=15,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        check_val_every_n_epoch=5,
        **CPU,
    )
    dm.setup()
    trainer.fit(module, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.test_dataloader())

    assert module.val_result is not None
    assert trainer.callback_metrics["val/pixel_auroc"] > 0.95
    trainer.test(module, datamodule=dm, verbose=False)
    assert module.test_result.overall.loc[0, "pixel_auroc"] > 0.95


class ProjectedPatchCore(AnomalyModule):
    """Gradient epochs on a projection, then a memory bank of projected good frames."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 4, kernel_size=1)
        self.patchcore = PatchcoreModel(num_neighbors=1)

    def training_step(self, batch, batch_idx):
        # Any objective; this one just keeps the projection trainable.
        return self.projection(batch["image"]).pow(2).mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    @torch.no_grad()
    def on_train_end(self):
        loader = self.trainer.datamodule.reference_dataloader()
        embedding = torch.cat(
            [
                reshape_embedding(self.projection(batch["image"].to(self.device)))
                for batch in loader
            ]
        )
        self.patchcore.subsample_embedding(embedding, 1.0)

    def forward(self, images):
        output = self.patchcore(self.projection(images))
        return {"anomaly_map": output["anomaly_map"], "pred_score": output["pred_score"]}


def test_memory_bank_after_gradient_training(root):
    module = ProjectedPatchCore()
    dm = datamodule(root, shots=3, anomalous_fraction=1 / 3)
    trainer = L.Trainer(max_epochs=2, logger=False, enable_progress_bar=False, **CPU)
    trainer.fit(module, datamodule=dm)
    assert module.patchcore.memory_bank.shape == (3 * SIZE * SIZE, 4)

    trainer.test(module, datamodule=dm, verbose=False)
    assert module.test_result.overall.loc[0, "image_auroc"] == pytest.approx(1.0)


# --- experiments -----------------------------------------------------------------


def test_experiment_runs_every_combination_and_resumes(root, tmp_path):
    built = []

    def factory(run):
        built.append(run)
        return Brightness()

    kwargs = dict(
        module_factory=factory,
        categories=["alpha", "beta"],
        shots=[1, 2, None],
        seeds=[0, 1],
        output_dir=tmp_path / "exp",
        trainer_kwargs=CPU,
        image_size=(SIZE, SIZE),
        batch_size=4,
        num_workers=0,
    )
    results = FewShotExperiment(root, **kwargs).run(progress=False)

    assert len(built) == 2 * 3 * 2
    overall = results[results["defect_type"] == "all"]
    assert len(overall) == 12 and len(results) == 12 * 3
    assert sorted(overall["n_train"].unique()) == [1, 2, 6]
    assert overall["shots"].isna().sum() == 4

    exp_dir = tmp_path / "exp"
    assert json.loads((exp_dir / "config.json").read_text())["shots"] == [1, 2, None]
    scores = pd.read_csv(exp_dir / "scores" / "alpha_fullshot_seed1.csv")
    assert len(scores) == 9 and set(scores["label"]) == {0, 1}

    # A rerun over a wider grid builds only the new runs.
    built.clear()
    results = FewShotExperiment(root, **{**kwargs, "seeds": [0, 1, 2]}).run(progress=False)
    assert {(r.category, r.shots, r.seed) for r in built} == {
        (c, s, 2) for c in ("alpha", "beta") for s in (1, 2, None)
    }
    assert len(results) == 18 * 3

    summary = summarize(results)
    assert list(summary.index.get_level_values("shots").unique()) == ["1", "2", "full"]
    assert list(summary.loc["1"].index) == ["alpha", "beta", "mean"]
    assert summary.loc[("full", "mean"), ("image_auroc", "mean")] == pytest.approx(1.0)
    assert summary.loc[("1", "alpha"), ("aupro", "std")] == pytest.approx(0.0)

    text = table(results, "image_auroc")
    assert list(text.columns) == ["1", "2", "full"]
    assert text.loc["mean", "full"] == "100.0 ± 0.0"


def test_experiment_trains_and_logs_each_run(root, tmp_path):
    results = FewShotExperiment(
        root,
        lambda run: Segmenter(),
        categories=["beta"],
        shots=[2],
        seeds=[0, 1],
        output_dir=tmp_path / "exp",
        trainer_kwargs=lambda run: {**CPU, "max_epochs": 10},
        anomalous_fraction=2 / 3,
        image_size=(SIZE, SIZE),
        batch_size=4,
        num_workers=0,
    ).run(progress=False)

    overall = results[results["defect_type"] == "all"]
    assert list(overall["n_anomalous_train"]) == [4, 4]
    assert (overall["n_images"] == 5).all()  # 3 good + 1 of 3 per defect type
    assert (overall["pixel_auroc"] > 0.95).all()
    metrics = pd.read_csv(
        next((tmp_path / "exp" / "logs" / "beta_2shot_seed0").rglob("metrics.csv"))
    )
    assert "train/loss_epoch" in metrics and "test/image_auroc" in metrics


def test_experiment_without_output_dir_keeps_results_in_memory(root):
    results = FewShotExperiment(
        root,
        lambda run: PatchCore(pixel_extractor, sampling_ratio=1.0, blur_sigma=1.0, per_defect=False),
        categories=["beta"],
        shots=[1],
        seeds=[0],
        trainer_kwargs=CPU,
        image_size=(SIZE, SIZE),
        num_workers=0,
    ).run(progress=False)
    assert list(results["defect_type"]) == ["all"]
    assert results.loc[0, "image_auroc"] == pytest.approx(1.0)
