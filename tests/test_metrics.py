"""Contract for the pixel metrics computed from per-image score histograms.

Two things are being pinned. The algebra: every pixel metric is a functional of counts
at or above a threshold, so a query has to be a sum of per-image rows and nothing else.
And the definition of AUPRO: it averages *per region*, so a large region and a tiny one
count the same — the failure mode of a fast reimplementation is silently collapsing it
into pixel recall.

The exact reference at the bottom is deliberately independent of the implementation and
of the kornia-based predecessor, so it stays a net after both are gone.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from adcore.metrics import (
    DEFAULT_FPR_LIMIT,
    REGION_STRUCTURE,
    ScoreBins,
    build_pixel_histograms,
    compute_metrics,
    image_auroc,
    pixel_metrics,
)


def metrics_of(masks, maps, *, n_bins=4096, strategy="quantile", structure=REGION_STRUCTURE):
    histograms = build_pixel_histograms(
        np.asarray(masks), np.asarray(maps), n_bins=n_bins, strategy=strategy, structure=structure
    )
    return pixel_metrics(histograms)


# --- the definition of AUPRO -------------------------------------------------------


def test_regions_are_averaged_not_pixels():
    """A 100-px region ranked first and a 1-px region ranked last gives 0.5, not 100/101.

    This is the test that catches collapsing PRO into pixel recall: under pixel recall
    the big region dominates and the answer is ~1.0.
    """
    mask = np.zeros((1, 20, 20), dtype=np.uint8)
    mask[0, 0:10, 0:10] = 1  # 100 px
    mask[0, 19, 19] = 1  # 1 px

    scores = np.zeros((1, 20, 20), dtype=np.float32)
    scores[0, 0:10, 0:10] = 1.0  # the big region scores above all background
    scores[0, 19, 19] = -1.0  # the small one below all of it

    assert metrics_of(mask, scores).aupro == pytest.approx(0.5, abs=1e-6)


def test_perfect_and_inverted_separation():
    mask = np.zeros((2, 8, 8), dtype=np.uint8)
    mask[0, 1:3, 1:3] = 1
    mask[1, 5:7, 5:7] = 1
    scores = mask.astype(np.float32)

    assert metrics_of(mask, scores).aupro == pytest.approx(1.0, abs=1e-6)
    assert metrics_of(mask, -scores).aupro == pytest.approx(0.0, abs=1e-6)


def test_two_regions_split_the_curve():
    # One region always ranked first, one always last: PRO averages to 0.5 everywhere.
    mask = np.zeros((1, 10, 10), dtype=np.uint8)
    mask[0, 0, 0] = 1
    mask[0, 9, 9] = 1

    scores = np.zeros((1, 10, 10), dtype=np.float32)
    scores[0, 0, 0] = 1.0
    scores[0, 9, 9] = -1.0

    assert metrics_of(mask, scores).aupro == pytest.approx(0.5, abs=1e-6)


def test_connectivity_is_an_explicit_convention():
    """Diagonally touching pixels are two regions under the MVTec convention, one under 8.

    The kornia predecessor was a 3x3 max-pool, i.e. 8-connected, and did not converge at
    `num_iterations=1000`, so it matched neither convention.
    """
    mask = np.zeros((1, 8, 8), dtype=np.uint8)
    mask[0, 2, 2] = 1
    mask[0, 3, 3] = 1
    maps = np.zeros((1, 8, 8), dtype=np.float32)

    four = build_pixel_histograms(mask, maps, structure=REGION_STRUCTURE)
    eight = build_pixel_histograms(mask, maps, structure=np.ones((3, 3), dtype=int))

    assert int(four.n_regions.sum()) == 2
    assert int(eight.n_regions.sum()) == 1


# --- the algebra -------------------------------------------------------------------


def test_binning_is_exact_when_distinct_scores_fit_in_the_bins():
    rng = np.random.default_rng(0)
    masks = (rng.random((3, 10, 10)) < 0.2).astype(np.uint8)
    maps = rng.integers(0, 20, size=(3, 10, 10)).astype(np.float32)

    result = metrics_of(masks, maps, n_bins=64)

    flat_mask, flat_map = masks.ravel(), maps.ravel()
    assert result.auroc == pytest.approx(roc_auc_score(flat_mask, flat_map), abs=1e-12)
    assert result.aupr == pytest.approx(
        average_precision_score(flat_mask, flat_map), abs=1e-12
    )


def test_binned_metrics_track_sklearn_on_continuous_scores():
    rng = np.random.default_rng(1)
    masks = (rng.random((4, 32, 32)) < 0.1).astype(np.uint8)
    maps = (rng.normal(size=(4, 32, 32)) + masks * 1.5).astype(np.float32)

    result = metrics_of(masks, maps)

    flat_mask, flat_map = masks.ravel(), maps.ravel()
    assert result.auroc == pytest.approx(roc_auc_score(flat_mask, flat_map), abs=1e-3)
    assert result.aupr == pytest.approx(
        average_precision_score(flat_mask, flat_map), abs=1e-3
    )


def test_a_query_is_a_sum_of_rows_on_the_full_set_grid():
    """The property the whole design rests on.

    Bins come from every pixel of the set; a query re-aggregates the rows it selects
    without rebinning. That has to agree with scoring those images on their own.
    """
    rng = np.random.default_rng(2)
    masks = (rng.random((6, 24, 24)) < 0.15).astype(np.uint8)
    maps = (rng.normal(size=(6, 24, 24)) + masks).astype(np.float32)
    rows = np.array([0, 2, 5])

    histograms = build_pixel_histograms(masks, maps)
    subset = pixel_metrics(histograms, rows)

    flat_mask, flat_map = masks[rows].ravel(), maps[rows].ravel()
    assert subset.auroc == pytest.approx(roc_auc_score(flat_mask, flat_map), abs=1e-3)
    assert subset.aupr == pytest.approx(
        average_precision_score(flat_mask, flat_map), abs=1e-3
    )
    assert subset.aupro == pytest.approx(
        exact_aupro(masks[rows], maps[rows]), abs=1e-3
    )


def test_region_counts_and_weights_are_additive_over_rows():
    rng = np.random.default_rng(3)
    masks = (rng.random((5, 16, 16)) < 0.2).astype(np.uint8)
    maps = rng.normal(size=(5, 16, 16)).astype(np.float32)

    histograms = build_pixel_histograms(masks, maps)
    rows = np.array([1, 3])

    alone = build_pixel_histograms(masks[rows], maps[rows], strategy="uniform")
    assert int(histograms.n_regions[rows].sum()) == int(alone.n_regions.sum())


def test_both_mask_layouts_agree():
    # `collate_fn` stacks masks that `ImageDataset` already unsqueezed, giving (N, 1, H, W).
    rng = np.random.default_rng(4)
    masks = (rng.random((3, 12, 12)) < 0.2).astype(np.uint8)
    maps = rng.normal(size=(3, 12, 12)).astype(np.float32)

    flat = metrics_of(masks, maps)
    channelled = metrics_of(masks[:, None], maps)

    assert flat.aupro == pytest.approx(channelled.aupro)
    assert flat.auroc == pytest.approx(channelled.auroc)


def test_positive_pixel_rate_is_the_defect_pixel_fraction():
    masks = np.zeros((2, 10, 10), dtype=np.uint8)
    masks[0, :5, :10] = 1  # 50 of 100
    maps = np.zeros((2, 10, 10), dtype=np.float32)

    assert metrics_of(masks, maps).positive_pixel_rate == pytest.approx(0.25)


# --- robustness --------------------------------------------------------------------


def test_quantile_bins_survive_an_outlier_that_uniform_bins_do_not():
    """Why quantile is the default: uniform bin width degrades with dynamic range.

    PatchCore scores are unbounded distances, so a single hot pixel is the normal case,
    not a pathological one.
    """
    rng = np.random.default_rng(5)
    masks = (rng.random((4, 32, 32)) < 0.1).astype(np.uint8)
    maps = (rng.normal(size=(4, 32, 32)) + masks * 1.5).astype(np.float32)
    maps[0, 0, 0] = float(maps.max() * 50)

    exact = average_precision_score(masks.ravel(), maps.ravel())
    quantile = metrics_of(masks, maps, n_bins=1024, strategy="quantile").aupr
    uniform = metrics_of(masks, maps, n_bins=1024, strategy="uniform").aupr

    assert abs(quantile - exact) < abs(uniform - exact)
    assert quantile == pytest.approx(exact, abs=1e-3)


def test_degenerate_selections_return_nan_rather_than_raising():
    maps = np.random.default_rng(6).normal(size=(2, 8, 8)).astype(np.float32)

    no_defects = metrics_of(np.zeros((2, 8, 8), dtype=np.uint8), maps)
    assert np.isnan(no_defects.auroc)
    assert np.isnan(no_defects.aupro)

    all_defects = metrics_of(np.ones((2, 8, 8), dtype=np.uint8), maps)
    assert np.isnan(all_defects.auroc)


def test_constant_scores_are_a_coin_flip_not_a_nan():
    mask = np.zeros((1, 10, 10), dtype=np.uint8)
    mask[0, 2:4, 2:4] = 1
    maps = np.full((1, 10, 10), 3.5, dtype=np.float32)

    result = metrics_of(mask, maps)

    assert result.auroc == pytest.approx(0.5, abs=1e-6)
    assert result.aupro == pytest.approx(DEFAULT_FPR_LIMIT / 2, abs=1e-6)


def test_score_bins_handle_a_degenerate_range():
    bins = ScoreBins.from_scores(np.full(100, 2.0, dtype=np.float32), n_bins=256)

    assert np.all(np.isfinite(bins.edges))
    assert np.all(bins.index(np.full(10, 2.0, dtype=np.float32)) < bins.n_bins)


# --- an exact reference, independent of the implementation -------------------------


def exact_aupro(masks: np.ndarray, maps: np.ndarray, fpr_limit: float = DEFAULT_FPR_LIMIT) -> float:
    """AUPRO by global sort, with no binning and no resampling.

    PRO(t) = mean over regions of the fraction of the region at or above t, which is a
    sum over defect pixels weighted by 1 / (n_regions * region_size). Both curves are
    stepped at every distinct score, so this is the definition with nothing in between.
    """
    from scipy import ndimage

    masks = masks[:, 0] if masks.ndim == 4 else masks
    weights = np.zeros(masks.shape, dtype=np.float64)
    n_regions = 0
    for i, mask in enumerate(masks):
        labels, count = ndimage.label(mask > 0, structure=REGION_STRUCTURE)
        sizes = np.bincount(labels.ravel())
        hit = labels > 0
        weights[i][hit] = 1.0 / sizes[labels[hit]]
        n_regions += count
    if n_regions == 0:
        return float("nan")

    scores = maps.ravel()
    order = np.argsort(-scores, kind="stable")
    background = (masks.ravel() == 0)[order]
    pro = np.cumsum(weights.ravel()[order]) / n_regions
    fpr = np.cumsum(background) / background.sum()

    # Step only at the last member of each run of equal scores, so ties resolve together.
    ranked = scores[order]
    last = np.flatnonzero(np.r_[np.diff(ranked) != 0, True])
    pro, fpr = np.r_[0.0, pro[last]], np.r_[0.0, fpr[last]]

    keep = fpr <= fpr_limit
    fpr_cut, pro_cut = fpr[keep], pro[keep]
    if fpr_cut[-1] < fpr_limit and keep.sum() < len(fpr):
        nxt = keep.sum()
        span = fpr[nxt] - fpr_cut[-1]
        frac = (fpr_limit - fpr_cut[-1]) / span if span > 0 else 0.0
        fpr_cut = np.r_[fpr_cut, fpr_limit]
        pro_cut = np.r_[pro_cut, pro_cut[-1] + frac * (pro[nxt] - pro_cut[-1])]

    return float(np.trapezoid(pro_cut, fpr_cut) / fpr_limit)


@pytest.mark.parametrize("seed", [10, 11, 12, 13, 14])
def test_aupro_matches_the_exact_reference(seed):
    rng = np.random.default_rng(seed)
    masks = np.zeros((3, 40, 40), dtype=np.uint8)
    for i in range(3):
        for _ in range(rng.integers(1, 4)):
            r, c = rng.integers(0, 33, size=2)
            h, w = rng.integers(2, 7, size=2)
            masks[i, r : r + h, c : c + w] = 1
    maps = (rng.normal(size=(3, 40, 40)) + masks * 1.2).astype(np.float32)

    assert metrics_of(masks, maps).aupro == pytest.approx(
        exact_aupro(masks, maps), abs=1e-3
    )




# --- image level and the one-call wrapper --------------------------------------------


def test_image_auroc_matches_sklearn_and_is_nan_on_one_class():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=50)
    scores = rng.normal(size=50) + labels
    assert image_auroc(labels, scores) == pytest.approx(roc_auc_score(labels, scores))
    assert np.isnan(image_auroc(np.zeros(5), np.arange(5)))


def test_compute_metrics_combines_image_and_pixel_scores():
    rng = np.random.default_rng(1)
    masks = np.zeros((4, 1, 32, 32), dtype=np.uint8)
    masks[2:, :, 8:16, 8:16] = 1
    maps = rng.normal(size=masks.shape).astype(np.float32) + 3 * masks
    labels = np.array([0, 0, 1, 1])
    result = compute_metrics(labels, maps.reshape(4, -1).max(1), masks, maps)

    pixel = metrics_of(masks, maps, n_bins=16384)
    assert result.image_auroc == pytest.approx(1.0)
    assert (result.pixel_auroc, result.pixel_aupr, result.aupro) == pytest.approx(
        (pixel.auroc, pixel.aupr, pixel.aupro)
    )
