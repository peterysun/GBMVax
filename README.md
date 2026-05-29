# GBMVax

**Personalized cancer vaccine neoantigen prediction pipeline for glioblastoma.**

GBMVax predicts which mutated peptides arising in a GBM patient are the strongest candidates for a personalized peptide vaccine. The pipeline scores every candidate on five biologically meaningful axes (HLA binding, antigen processing, clonal architecture, immunogenicity, autoimmune cross-reactivity) and returns a ranked top-N list.

The five innovations relative to general-purpose neoantigen tools:

1. **GBM-specific training** — the HLA binding transformer is trained on IEDB and fine-tuned on GBM-relevant immunopeptidomics, not on melanoma like most published predictors.
2. **Immunopeptidomics integration** — multi-task model with separate heads for binding affinity (IC50) and presentation (mass-spec elution), so the model learns the joint event "actually presented," not just "binds in vitro."
3. **Cross-reactivity filtering** — BLOSUM62-based scan against the human proteome to flag candidates likely to cause autoimmunity.
4. **Clonal architecture weighting** — VAF-derived clonality score prioritizes truncal mutations (with a PyClone-VI hook for proper CCF in v2).
5. **Composite vaccine score** — weighted combination of all five signals into a single ranking.

---

## Repository layout

```
GBMVax/
├── configs/
│   └── config.yaml              # All hyperparameters and data paths
├── gbmvax/
│   ├── utils/                   # Config loading, device, sequence + HLA helpers
│   ├── data/                    # IEDB, proteome, MAF, validation supplement loaders + peptide windowing
│   ├── models/                  # HLA binding transformer, processing reranker, NetMHCpan wrapper, trainer
│   └── pipeline/                # Cross-reactivity, clonal, composite, orchestrator
├── scripts/
│   ├── train.py                 # Train the multi-task HLA binding model on IEDB
│   ├── predict.py               # Run the full pipeline on a single patient
│   ├── evaluate.py              # Secondary metric: Spearman on IEDB holdout + NetMHCpan baseline
│   └── validate.py              # Primary metric: AUC on Keskin/Hilf T-cell response data
├── notebooks/
│   └── train_colab.ipynb        # GPU-based training notebook
├── tests/                       # Unit tests (36 currently)
└── requirements.txt
```

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify tests pass
PYTHONPATH=. pytest tests/ -q

# 3. Train the HLA binding model (use --debug first to smoke-test in ~10 min)
PYTHONPATH=. python scripts/train.py --config configs/config.yaml --debug
PYTHONPATH=. python scripts/train.py --config configs/config.yaml

# 4. Evaluate on the IEDB holdout + NetMHCpan baseline
PYTHONPATH=. python scripts/evaluate.py --config configs/config.yaml

# 5. PRIMARY VALIDATION: predict Keskin/Hilf T-cell responses
PYTHONPATH=. python scripts/validate.py --config configs/config.yaml

# 6. Run the full pipeline on a patient
PYTHONPATH=. python scripts/predict.py \
    --config configs/config.yaml \
    --mutations data/your_patient/mutations.tsv \
    --hla HLA-A*02:01 HLA-A*03:01 HLA-B*07:02 HLA-B*44:02 HLA-C*05:01 HLA-C*07:02 \
    --output results/patient_X_neoantigens.tsv
```

---

## Patient mutation input format

A tab-separated file with header row:

| patient_id | gene  | protein_pos | wt_aa | mt_aa | vaf  |
|------------|-------|-------------|-------|-------|------|
| PT001      | TP53  | 175         | R     | H     | 0.42 |
| PT001      | EGFR  | 289         | A     | V     | 0.31 |
| PT001      | IDH1  | 132         | R     | H     | 0.48 |

This is the schema produced by `gbmvax.data.mutations.load_maf`, so you can also use a MAF file directly via the loader.

---

## Architecture

```
mutations + HLA type
        │
        ▼
peptide window generation        ← gbmvax/data/peptides.py
        │
        ▼
HLA binding prediction           ← gbmvax/models/hla_binding.py (multi-task transformer)
        │
        ▼
antigen processing scoring       ← gbmvax/models/processing.py (NetChop + TAP + reranker)
        │
        ▼
cross-reactivity filtering       ← gbmvax/pipeline/cross_reactivity.py (BLOSUM62 vs human proteome)
        │
        ▼
clonal weighting                 ← gbmvax/pipeline/clonal.py (VAF → score, PyClone hook for v2)
        │
        ▼
composite scoring + ranking      ← gbmvax/pipeline/composite.py
        │
        ▼
top N ranked candidates
```

The orchestrator (`gbmvax/pipeline/orchestrator.py`) wires every stage together with a `GBMVaxPipeline` class.

---

## Validation

**Primary** (Nature Methods headline):
- Predict which neoantigens from Keskin 2019 (Nature) and Hilf 2019 (Nature) produced T-cell responses in actual patients.
- Metric: AUC on binary response label; secondary: precision@top-K.

**Secondary**:
- Spearman ρ on IEDB binding-affinity holdout.
- Head-to-head against NetMHCpan 4.2c on the same test set.

---

## Hardware

The pipeline auto-detects CUDA → MPS → CPU via `accelerate`. Training the full transformer:

| Hardware             | Time per epoch  | Total (50 epochs) |
|----------------------|-----------------|-------------------|
| Mac M1/M2/M3 (MPS)   | ~25 min         | ~20 hours         |
| Colab T4 (free)      | ~5 min          | ~4 hours          |
| Colab A100 (paid)    | ~1 min          | ~1 hour           |

For dev iteration on Mac use `--debug` (100k IEDB rows → ~10 min total training).

---

## Status

- v1.0: all modules functional, training and inference work end-to-end, 36 unit tests passing.
- v1.1 backlog: full PyClone-VI CCF integration, learned immunogenicity model on `tcell_full_v3`, fine-tuning step on Keskin/Hilf immunopeptidomics.

---

## Paper target

Nature Methods. Primary headline figure: GBMVax composite-score AUC on Keskin+Hilf neoantigen response data vs NetMHCpan baseline + ablation showing each of the five innovations contributes positive ΔAUC.
