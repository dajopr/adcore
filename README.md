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
| `adcore.stats` | `OutputMean` and the `Metric` convention for a module's own val/test statistics |
| `adcore.extractors` | `TimmExtractor`: frozen timm backbone → patch embedding |
| `adcore.datamodule` | `MVTecDataModule` (few-shot, optional anomalous training frames), `CachedDataset` |
| `adcore.mvtec` | `MVTecDataset`, `few_shot_subset`, `split_by_defect_type` |
| `adcore.metrics` | image AUROC, pixel AUROC / AUPR / AUPRO from per-image histograms |
| `adcore.evaluation` | `Predictions`, `score`, and `evaluate(module, loader)` for a one-call `Trainer.test` |
| `adcore.experiment` | `FewShotExperiment`: categories × shots × seeds, resumable, plus `summarize` / `table` |
| `adcore.tracking` | `MLflowTracking`: record sweeps in a local MLflow store; `upload_sweep`, `load_runs` |
| `adcore.dashboard` | `adcore-dashboard`: Streamlit app comparing sweeps by architecture |
| `adcore.runner` | `adcore-run`: experiments from Hydra configs, one worker per GPU |

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

### Tracking model statistics

A module can record its own quantities over the val / test set, such as the terms of an
NLL or how much a subspace over-fits, next to the detection metrics. Pass a torchmetrics
`MetricCollection` (or a dict of metrics) as `stats`. Each metric:

- keeps its state with `add_state(..., dist_reduce_fx=...)`;
- takes `update(batch, output)`, where `output` is everything `forward` returned, so return
  the tensors your metrics need next to `anomaly_map` and `pred_score`;
- returns one scalar from `compute()`. To get several values from one state, write one
  metric per value and let `compute_groups` share the state.

```python
from torchmetrics import Metric, MetricCollection, PearsonCorrCoef
from adcore import AnomalyModule, OutputMean

class EnergyFraction(Metric):
    """In-subspace energy fraction on normal vs anomalous patches, per component."""
    full_state_update = False

    def __init__(self, n_components):
        super().__init__()
        self.add_state("sums", torch.zeros(2, n_components, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("counts", torch.zeros(2, dtype=torch.float64), dist_reduce_fx="sum")

    def update(self, batch, output):
        ef = output["energy_fraction"]                                   # (B, C, h, w)
        anomalous = F.adaptive_max_pool2d(batch["mask"].float(), ef.shape[-2:]).flatten() > 0
        ef = ef.permute(0, 2, 3, 1).flatten(0, 2).double()               # (B*h*w, C)
        self.sums += torch.stack([ef[~anomalous].sum(0), ef[anomalous].sum(0)])
        self.counts += torch.stack([(~anomalous).sum(), anomalous.sum()])

    def ratio(self):                                                     # (C,) normal / anomalous
        return (self.sums[0] / self.counts[0]) / (self.sums[1] / self.counts[1])

class EnergyFractionRatioMean(EnergyFraction):
    def compute(self): return self.ratio().mean()

class EnergyFractionRatioMax(EnergyFraction):
    def compute(self): return self.ratio().max()

class ScoreEnergyCorrelation(PearsonCorrCoef):
    def update(self, batch, output):
        super().update(output["patch_score"].flatten(), output["energy_fraction"].mean(1).flatten())

class MyFlow(AnomalyModule):
    def __init__(self, n_components=768):
        super().__init__(stats=MetricCollection(
            {"ef_ratio_mean": EnergyFractionRatioMean(n_components),
             "ef_ratio_max": EnergyFractionRatioMax(n_components),
             "score_ef_pearson": ScoreEnergyCorrelation(),
             "nll_logdet": OutputMean("nll_logdet")},       # mean of a (B,) or (B, ...) output
            compute_groups=[["ef_ratio_mean", "ef_ratio_max"], ["score_ef_pearson"], ["nll_logdet"]],
        ))
```

- The collection is updated on every val and test batch, then computed over the whole set
  and reset at epoch end. Validation and testing each get their own copy. Use `val_stats=`
  or `test_stats=` to give one stage different metrics, e.g. expensive ones only at test.
- The values appear in four places:
  - logged as `val/stat/<name>` / `test/stat/<name>`, so a checkpoint callback can monitor them;
  - kept on `module.val_result.stats` / `module.test_result.stats`, and returned by `evaluate`;
  - in a few-shot experiment, as `stat/<name>` columns on each run's `defect_type == "all"`
    row of `results.jsonl`. The stats cover the whole test split, not each defect type;
  - with tracking, as final metrics of the MLflow child run and in the dashboard's metric
    picker. `table(results, "stat/<name>")` prints them as they are, not in %.
- Keep the state small, e.g. running sums. A metric that stores every value, like
  `SpearmanCorrCoef`, holds the whole test set in memory.
- A metric's state is not saved in checkpoints.

In a model config, pass the collection through to `AnomalyModule` (`PatchCore` takes it as is):

```yaml
module:
  _target_: adcore.PatchCore
  stats:
    _target_: torchmetrics.MetricCollection
    metrics:
      pred_score_mean: {_target_: adcore.OutputMean, key: pred_score}
```

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

## Tracking and comparing sweeps

Install the extra with `uv sync --extra track` (or `pip install adcore[track]`). It adds
MLflow, Streamlit and Plotly. By default everything stays local in a SQLite file, so no
server or licence is needed.

```python
from adcore import FewShotExperiment, MLflowTracking

experiment = FewShotExperiment(
    root,
    module_factory=...,
    output_dir="runs/patchcore-wrn50",
    tracking=MLflowTracking(
        experiment="mvtec-fewshot",     # one per benchmark
        arch="patchcore-wrn50",         # what the dashboard compares by
        params={"backbone": "wide_resnet50_2", "layers": "2,3"},
    ),
)
experiment.run()
```

- Each sweep is a parent run named after `output_dir`, unless you pass `sweep=`.
- Each (category, shots, seed) is a nested child run. Its params are `category`, `shots` (`"full"` for all frames), `seed` and `arch`. Its metrics are the whole-test-set row (`image_auroc`, …) plus `defect/<type>/<metric>` for each defect type, and the module's `stat/<name>` values (see [Tracking model statistics](#tracking-model-statistics)).
- Trainable modules also log their training losses to the child run, through Lightning's `MLFlowLogger`.
- The store is `$MLFLOW_TRACKING_URI` when it is set, e.g. `http://mlflow-host:5000`; otherwise it is `mlflow.db` in the working directory. Pass `tracking_uri=...` to override either. MLflow reads its credentials (`MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `MLFLOW_TRACKING_TOKEN`) from the environment itself.
- `results.jsonl` stays the source of truth for resuming. When a sweep starts tracking, its runs already in `results.jsonl` are copied into MLflow first.
- To add a sweep that ran before tracking existed, call `upload_sweep("runs/old-sweep", MLflowTracking(experiment="mvtec-fewshot", arch="..."))`. Calling it again adds nothing.
- `load_runs(uri, experiments=[...])` returns the rows of `results.jsonl` with `arch`, `sweep` and `experiment` columns added. `summarize` and `table` work on any subset of them.

Start the dashboard with `adcore-dashboard [--tracking-uri URI] [--port 8501]`. The URI defaults to the same store. It compares by `arch` or by sweep, and has these tabs:

- **Overview**: the metric against shots, with the std over seeds as a band.
- **Per category**: one small plot per category.
- **Heatmap**: category × architecture at a chosen number of shots.
- **Tables**: the `table` view for each group.
- **Defect types**: the metric for each defect type in one category.
- **Cost**: fit and test time per run.

To browse the raw runs, use `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Running from configs

Install the extra with `uv sync --extra run` (or `pip install adcore[run]`). It adds Hydra
on top of `track`. `adcore-run` builds a `FewShotExperiment` from three config groups:

- `model`: architecture, backbone, transforms and Trainer settings.
- `experiment`: dataset, categories, shots, seeds.
- `launcher`: `single`, or `gpus` for one worker per GPU.

```bash
export MVTEC_ROOT=/data/shared-data/public_datasets/raw/MVTec   # VISA_ROOT for visa_fewshot
export MLFLOW_TRACKING_URI=http://mlflow-host:5000              # optional
adcore-run model=patchcore_wrn50 experiment=mvtec_fewshot
adcore-run model=patchcore_wrn50 experiment=visa_fewshot launcher=gpus launcher.devices=[0,1,2,3]
adcore-run model=patchcore_wrn50 experiment.categories=[bottle] experiment.shots=[1] model.module.num_neighbors=1
adcore-run -m model=patchcore_wrn50,my_model experiment=mvtec_fewshot   # one model after the other
adcore-run --cfg job                                                    # print the composed config
```

The shipped configs live in [src/adcore/conf](src/adcore/conf). `adcore.yaml` documents every top-level key.

### Models from another repository

A downstream repo depends on adcore and keeps its own config directory. The only
requirement on a model is that it is an `AnomalyModule`:

```
my-repo/
  my_models/__init__.py           # class MyDetector(AnomalyModule): ...
  configs/model/my_detector.yaml
```

```yaml
# configs/model/my_detector.yaml
name: my-detector             # MLflow `arch` and the output directory
shared:                       # built once per process, passed by name to every module
  extractor: {_target_: adcore.TimmExtractor, model_name: vit_base_patch14_dinov2, out_indices: [8]}
module:                       # built fresh for every run
  _target_: my_models.MyDetector
  lr: 1.0e-4
image_size: [448, 448]
transform: null               # or a torchvision v2 transform, e.g. {_target_: torchvision.transforms.v2.Compose, transforms: [...]}
train_transform: null
support_aug: null             # or {_target_: adcore.AugmentSpec, n_views: 30}
trainer: {max_epochs: 5}
```

```bash
cd my-repo && adcore-run --config-dir configs model=my_detector experiment=mvtec_fewshot
```

- If the module's constructor has a `run` parameter, it gets the `adcore.Run` (category, shots, seed), e.g. for per-category text prompts.
- `--config-dir` can also add `experiment/` or `launcher/` configs. A repo that prefers its own entrypoint can call `adcore.runner.run(cfg)` from its `@hydra.main`, putting `hydra: {searchpath: [pkg://adcore.conf]}` and `- adcore` in its primary config's defaults.
- `model.trainer` overrides `experiment.trainer`. Both override the defaults of `FewShotExperiment`.

### Output, resuming, tracking

- Each (experiment, model) writes to `runs/<experiment.name>/<model.name>/`; set `output_root` or `output_dir` to move it. Hydra's own `.hydra/` and `adcore.log` go there as well.
- Rerunning the same command resumes, and a wider grid only runs what is new.
- `config.json` records a hash of the model config and of the experiment settings, leaving out the grid and loader settings. If you change those settings and run into the same directory, the run stops with an error instead of mixing results. Pass `force=true` to add to it anyway.
- Tracking is on by default: the MLflow experiment is `experiment.name` and `arch` is `model.name`. Every key of the model config is logged as a `model.*` param, e.g. `model.shared.extractor.model_name`. Turn it off with `tracking.enabled=false`.

### Several GPUs

`launcher=gpus` uses every visible GPU. `launcher.devices=[0,2]` picks some by index into
`CUDA_VISIBLE_DEVICES`, and `launcher.workers_per_device=2` puts two workers on each.

- Each worker is a spawned process that sees only its GPU.
- Workers take whole categories off a queue, so each category's test split is still decoded once. When there are fewer categories than workers, the queue holds (category, shots) pairs instead.
- Runs are split across GPUs, never one run across several. A run's results don't depend on which worker ran it.
- The main process alone writes `results.jsonl`, so resuming works as before. A unit that fails is reported at the end, and the other units still finish.
- Against a tracking server (`http(s)://`, or a database such as PostgreSQL), workers log their runs to MLflow themselves, training curves included. A SQLite store can't take several writers. With one, the main process records each run's final metrics and there are no training curves in MLflow, though they stay in `logs/`.
- Every worker loads the backbone once, which takes a few seconds. For a handful of short runs, one GPU is faster.

## Acknowledgements

`adcore.patchcore` and `adcore.coreset` contain code adapted from
[anomalib](https://github.com/open-edge-platform/anomalib) 2.2.0
(`PatchcoreModel`, `KCenterGreedy` and `SparseRandomProjection`),
Copyright (C) 2022-2025 Intel Corporation, licensed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
The vendored sections are marked in the source; see [NOTICE](NOTICE).

## License

adcore is licensed under the [Apache License 2.0](LICENSE).
