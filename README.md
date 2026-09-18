# adcore

A utility library for anomaly detection.

Anomaly detection on Lightning: PatchCore, MVTec AD loading, metrics, evaluation and
few-shot experiments.

| module | what |
| --- | --- |
| `adcore.module` | `AnomalyModule`: a `LightningModule` whose val/test steps collect predictions and score them per category and defect type |
| `adcore.detectors` | `PatchCore`: extractor + memory bank, fitted in one optimizer-free epoch |
| `adcore.patchcore` | `PatchcoreModel`: memory bank + nearest-neighbour scoring on an already extracted `(B, C, H, W)` embedding (adapted from [anomalib](https://github.com/open-edge-platform/anomalib)) |
| `adcore.coreset` | k-center greedy coreset subsampling (adapted from [anomalib](https://github.com/open-edge-platform/anomalib)) |
| `adcore.extractors` | `TimmExtractor`: frozen timm backbone → patch embedding |
| `adcore.datamodule` | `MVTecDataModule` (few-shot, optional anomalous training frames), `CachedDataset` |
| `adcore.mvtec` | `MVTecDataset`, `few_shot_subset`, `split_by_defect_type` |
| `adcore.metrics` | image AUROC, pixel AUROC / AUPR / AUPRO from per-image histograms |
| `adcore.evaluation` | `Predictions`, `score`, and `evaluate(module, loader)` for a one-call `Trainer.test` |
| `adcore.experiment` | `FewShotExperiment`: categories × shots × seeds, resumable, plus `summarize` / `table` |

## Writing a module

Subclass `AnomalyModule` and implement `forward(images) -> {"anomaly_map", "pred_score"}`.
Train it the usual Lightning way; `validation_step` / `test_step` are already there.

```python
import lightning as L
from adcore import AnomalyModule, MVTecDataModule

class MyDetector(AnomalyModule):
    def forward(self, images):          # (B, 3, H, W)
        ...
        return {"anomaly_map": maps,    # (B, 1, h, w), resized to the masks if needed
                "pred_score": scores}   # (B,)

    def training_step(self, batch, batch_idx):   # omit for zero-shot modules
        ...
    def configure_optimizers(self):
        ...

dm = MVTecDataModule(root, "bottle", shots=4, seed=0)
trainer = L.Trainer(max_epochs=10, devices=1)
trainer.fit(module, datamodule=dm)
trainer.test(module, datamodule=dm)
module.test_result.metrics      # per (category, defect_type); "all" = whole test split
module.test_result.predictions  # labels, image scores, masks, anomaly maps, paths
```

Batches are dicts: `image`, `mask` (1HW, 0/1), `label` (0 good / 1 anomalous),
`category`, `defect_type`, `image_path`.

Test / val metrics are logged as `test/image_auroc` etc. (mean over categories) and
`test/<category>/image_auroc`, so a `ModelCheckpoint(monitor="val/image_auroc")` works
when you pass a val loader. Scoring runs on one device.

### Supervised / prompt-guided training

`MVTecDataModule(..., anomalous_fraction=0.33)` moves that fraction of every defect
type's test frames into `train_dataloader()` and out of the test set.
`reference_dataloader()` yields only the defect-free shots, so a module that trains
with gradients and then needs a memory bank builds it at the end of fit:

```python
class ProjectedPatchCore(AnomalyModule):
    def __init__(self, extractor):
        super().__init__()
        self.extractor = extractor
        self.projection = nn.Conv2d(768, 768, 1)
        self.patchcore = PatchcoreModel(num_neighbors=1)

    def training_step(self, batch, batch_idx):
        ...  # loss on self.projection(self.extractor(batch["image"])), batch["mask"], ...

    @torch.no_grad()
    def on_train_end(self):
        loader = self.trainer.datamodule.reference_dataloader()
        embedding = torch.cat([
            reshape_embedding(self.projection(self.extractor(b["image"].to(self.device))))
            for b in loader
        ])
        self.patchcore.subsample_embedding(embedding, sampling_ratio=1.0)

    def forward(self, images):
        out = self.patchcore(self.projection(self.extractor(images)))
        return {"anomaly_map": upsample_and_blur(out["anomaly_map"], images.shape[-2:], 4.0),
                "pred_score": out["pred_score"]}
```

`PatchCore` itself skips anomalous frames in its training batches, so it can be fitted
on the same datamodule.

## Few-shot experiments

```python
from adcore import FewShotExperiment, PatchCore, TimmExtractor, summarize, table

extractor = TimmExtractor("wide_resnet50_2", out_indices=(2, 3))  # shared by every run

experiment = FewShotExperiment(
    root,
    module_factory=lambda run: PatchCore(extractor, sampling_ratio=1.0),
    shots=(1, 2, 4, 8),          # None = all training frames
    seeds=(0, 1, 2),
    output_dir="runs/patchcore-wrn50",
    trainer_kwargs={"devices": [3]},   # or a function of the run
)
results = experiment.run()       # DataFrame: one row per (category, shots, seed, defect_type)
table(results, "image_auroc")    # categories × shots, "mean ± std" in %
summarize(results)               # mean/std per metric, plus the category mean
```

- Each run builds a fresh module via `module_factory(run)`. The `Run` has `category`, `shots` and `seed`, so a module can depend on its category.
- Each run calls `L.seed_everything(seed)`, then `Trainer.fit` and `Trainer.test`. Modules without a `training_step` (zero-shot) skip the fit.
- Trainer defaults: one device, `max_epochs=1`, no checkpointing, no progress bar. Override them with `trainer_kwargs`.
- The seed also picks the shots. For a given seed the draws are nested: the 2-shot set contains the 1-shot set.
- `anomalous_fraction` is passed through to the datamodule, for supervised or prompt-guided modules.
- `output_dir` gets:
  - `config.json`
  - `results.jsonl`, appended after every run
  - `scores/<run>.csv` with per-image scores
  - `logs/<run>/`, Lightning's CSV logs of training losses and test metrics
- Rerunning skips runs already in `results.jsonl`, so an interrupted experiment resumes and a wider grid only runs what is new.
- Each category's test split is decoded once and held in memory for all its runs.
- Metrics are computed at `image_size` resolution (default 224×224).
- In `summarize`, the `"mean"` category averages categories within each seed first, so its std is the seed variance of the benchmark mean.

## Acknowledgements

`adcore.patchcore` and `adcore.coreset` contain code adapted from
[anomalib](https://github.com/open-edge-platform/anomalib) 2.2.0
(`PatchcoreModel`, `KCenterGreedy` and `SparseRandomProjection`),
Copyright (C) 2022-2025 Intel Corporation, licensed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
The vendored sections are marked in the source; see [NOTICE](NOTICE).

## License

adcore is licensed under the [Apache License 2.0](LICENSE).
