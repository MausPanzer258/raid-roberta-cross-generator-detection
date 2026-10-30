# Published experiment results

These artifacts document the original RoBERTa-base IID and four leave-one-generator-out (LOGO) runs on a selected RAID benchmark subset. Values are recorded from the original experiment; results have **not** been recomputed or retrained for this upload.

## Results (AI-class F1)

| Generator | IID F1 (%) | LOGO F1 (%) | Matched test rows |
|---|---:|---:|---:|
| chatgpt | 98.68 | 98.90 | 448 |
| gpt4 | 98.98 | 99.49 | 390 |
| llama-chat | 99.24 | 99.74 | 390 |
| mistral-chat | 98.81 | 99.28 | 414 |

Overall IID test: 1,610 rows (789 human, 821 AI); accuracy 99.13%, AI F1 99.15%. Per-generator comparisons use the fixed manifest in `iid/per_generator_test_manifest.csv` with equal numbers of human and target-generator AI texts. Some human examples occur in more than one generator-specific comparison, so the four comparisons do **not** constitute 1,642 independent records.

## Files

- `iid/iid_test_metrics.json`: one evaluation of the full original IID test split.
- `iid/iid_per_generator_metrics.csv`: IID scores on generator-matched test groups.
- `iid/per_generator_test_manifest.csv`: row IDs, source IDs, domain and label to reconstruct matched comparisons; **no raw text** is included.
- `iid/data_preflight_report.json`: sizes and leakage checks of the original fixed split.
- `iid/run_config.json` and `iid/training_history.csv`: training settings and validation history.
- `logo/<generator>/logo_test_metrics.json`: evaluation on the held-out generator plus matched human test samples.
- `logo/<generator>/run_config.json` and `training_history.csv`: fold-level details.
- `iid_vs_logo_comparison.csv`: the values used in the poster's comparison chart.

## Provenance and paths

The output files were originally stored in local directories named `iid_roberta_results_jupyter/` and `logo_roberta_results_jupyter/`. They were **copied** into this repository's `results/` directory for publication. All 19 experiment files preserve their original bytes, **except** `iid/run_config.json`: its two absolute Windows paths (`data_dir` and `output_dir`) were replaced with relative directory names to avoid disclosing local user-directory details. Model parameters, hash records and scores are unchanged. The original `run_config.json` byte sequence is not retained here.

For reproducing the experiments, obtain RAID from the original source and run the provided sampling and splitting scripts. The notebooks use output directories under a working project root; the published `results/` folder is a read-only record of the historical runs, not necessarily their live output directory. Original split CSVs are not redistributed here. The file digests in the run configuration provide a way to verify whether independently reconstructed splits match the original experiment inputs.

## Interpretation constraints

LOGO removes the held-out generator from **train and validation**, retrains a new classifier, and tests on the same fixed samples used for that generator in IID. The LOGO training sets contain fewer AI examples and different class proportions than IID. Thus small IID-vs-LOGO changes cannot be attributed solely to generator visibility. Scores pertain to this selected RAID subset, generators, domains, and preprocessing.

The repository does not distribute original RAID text or large trained checkpoint weights. Results are provided for transparency, not as evidence of universal real-world detection performance.
