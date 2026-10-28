# GlucoSL-KAN

Public research code for predicting postprandial glucose response from routine clinical information, meal nutrients and meal time, without CGM predictor inputs.

Repository: https://github.com/WangDongWei515/glucosl-kan

The numerical reproduction covers the main model comparisons, ablations, cross-cohort evaluation and the two nutritional sensitivity analyses. The workflow is data preparation, model fitting and evaluation. Figure rendering is not part of this entry point.

## 1 Install

Python 3.12 matches the study environment. Create an environment, activate it and install the numerical dependencies:

    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

For the complete comparator set, including the explicitly selected TabPFN v2 weights:

    python -m pip install -r requirements-tabpfn.txt
    python scripts/reproduce.py check --models all

The model code requests ModelVersion.V2 explicitly; installing a newer default TabPFN model does not replace the study model. TabPFN downloads its own weights when used. We do not redistribute those checkpoints.

requirements.lock.txt preserves the original macOS environment record. Use requirements.txt rather than installing that platform-specific record on another system.

## 2 Prepare data

Download the pinned ShanghaiT2DM v3, CGMacros v1.0.0 and USDA SR Legacy archives from their official sources, then build the meal tables and temporal split:

    python scripts/reproduce.py prepare --download

The food-composition step queries the Chinese food-composition platform and caches responses locally. It respects a delay between requests and stops on retrieval errors. Source nutrient database values are not included in this repository. See DATA_SOURCES.md for sources, attribution and access details.

If source datasets and a food-composition cache are already available:

    python scripts/reproduce.py prepare --raw-dir /path/to/data/raw --cn-cache-dir /path/to/cncdc_query_cache --offline

The required raw-directory layout is:

    cgmacros/1.0.0/extracted/CGMacros/
    shanghai_t2dm/v3/extracted/
    usda_fdc/sr_legacy_2018-04/extracted/FoodData_Central_sr_legacy_food_csv_2018-04/

The prepared-data checks require:

| Sample | Participants | Meals |
| --- | ---: | ---: |
| Shanghai training A | 67 | 1,952 |
| Shanghai validation B | 33 | 670 |
| CGMacros, all eligible participants | 44 | 1,147 |
| CGMacros, HbA1c at least 6.5% | 14 | 353 |

The more specifically mapped Shanghai validation subset must contain 347 meals. No patient may occur in both A and B.

## 3 Fit and evaluate

Complete comparator set, using 2,000 patient-level bootstrap resamples:

    python scripts/reproduce.py fit --models all --jobs 4 --run-dir reproduction/full
    python scripts/reproduce.py evaluate --models all --jobs 4 --bootstrap-draws 2000 --run-dir reproduction/full
    python scripts/reproduce.py verify --run-dir reproduction/full

A single command performs preparation, fitting, evaluation and verification:

    python scripts/reproduce.py all --download --models all --jobs 4 --run-dir reproduction/full

The complete set includes GlucoSL-KAN, GlucoSL-MLP, single-branch KAN, unconstrained GlucoSL-KAN, constant, median/logistic regression and XGBoost, plus the US-trained and cooked-weight variants. MLP and comparator selection repeat the original patient-grouped cross-validation within A.

A faster core check uses GlucoSL-KAN, XGBoost and the constant model, retaining US-trained and cooked-weight variants:

    python scripts/reproduce.py fit --models core --jobs 4 --run-dir reproduction/core
    python scripts/reproduce.py evaluate --models core --bootstrap-draws 0 --run-dir reproduction/core
    python scripts/reproduce.py verify --run-dir reproduction/core

Zero bootstrap draws evaluate point estimates only; they do not reproduce confidence intervals. Use 2,000 to obtain the study uncertainty procedure.

All newly fitted models, predictions and metrics are written below the chosen reproduction directory. The supplied study reference CSVs in results/glucosl are never overwritten by the entry point. Each reproduction uses a separate output directory and does not need unpublished study-development files.

## 4 Additional numerical analyses

After fitting and evaluating all models, run the descriptive post hoc analyses:

    python scripts/reproduce.py posthoc --models all --run-dir reproduction/full

Expanded-search XGBoost and shape-constrained GAM comparators are implemented separately:

    GLUCOSL_RUN_DIR="$PWD/reproduction/full" python experiments/glucosl/scripts/posthoc_comparators.py --part xgboost
    GLUCOSL_RUN_DIR="$PWD/reproduction/full" python experiments/glucosl/scripts/posthoc_comparators.py --part gam
    GLUCOSL_RUN_DIR="$PWD/reproduction/full" python experiments/glucosl/scripts/posthoc_comparators.py --part evaluate

For the GAM step, create a separate Python 3.12 environment and install requirements-gam.txt. That file supplies a compatible environment rather than an exact record of the original GAM environment, whose versions were not retained. Run --part gam with that interpreter, and use the main environment for --part evaluate. Its smoothing search, constraints and transformations are specified in the script. These are additional post hoc analyses rather than part of the primary model comparison.

## 5 Check the implementation and view supplied results

    python -m unittest discover -s tests -v
    python scripts/summarize_results.py

The tests check positive incremental integration, baseline crossing, patient averaging, patient-cluster resampling and training-only preprocessing.

See REPRODUCTION_CHECK.md for what was executed to check this release, including the distinction between point-estimate checks and full bootstrap reproduction.

## Data and model interpretation

CGM is used for outcomes, not predictors. Regression predicts positive 2-h iAUC in mg/dL*h; classification predicts whether any scheduled reading reaches 180 mg/dL within 0--120 minutes, including meal onset.

The ensemble averages member-level sensitivity--load products. It does not multiply averaged branches. Personal sensitivity is an internal response coefficient, not measured insulin sensitivity. Response curves describe model behavior; monotonicity constraints are modeling assumptions. An interval containing zero is an uncertain difference, not evidence of equivalence.

## Availability

Raw datasets, source food-composition values, participant-level derived tables, fitted models and third-party pretrained weights are not committed. The scripts obtain or generate required inputs locally. Source datasets and weights retain their original terms; no blanket code-reuse license is assigned by this repository.

No manuscript draft, author information document, private conversation or development-history file is included.

## Funding and declarations

This study was supported by the Special Fund of the Capacity Building and Continuing Education Center of the National Health Commission (Party School of the National Health Commission of China) (Grant No. JJDJ20260002-6). The author confirmed no other funding and no competing interests.
