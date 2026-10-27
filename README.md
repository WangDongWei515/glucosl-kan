# GlucoSL-KAN
Research code and aggregate results for interpretable postprandial glucose-response prediction without CGM predictor inputs.

## Status
Private research code and aggregate-results repository for the GlucoSL-KAN study. The author approved the initial 58-file scope on 28 October 2026. Repository: https://github.com/WangDongWei515/glucosl-kan .

## Included
Core KAN and sensitivity--load models, preprocessing and evaluation implementations; public-data download/build scripts and reusable food-mapping rules; final model configuration and the original environment record; aggregate evaluation, post hoc and interpretability outputs.

## Inspect the primary result without participant-level data
From the repository root, run:
    python3 scripts/summarize_results.py
This reads supplied aggregate CSVs without retraining or reevaluating models.

## Data and execution
The primary outcome is positive 2-h iAUC, in mg/dL*h; classification concerns any scheduled glucose reading reaching 180 mg/dL within 0--120 minutes. CGM contributes labels, not predictor inputs.

Obtain source data separately:
- ShanghaiT2DM v3: https://doi.org/10.6084/m9.figshare.20444397.v3 (CC BY 4.0).
- CGMacros v1.0.0: https://physionet.org/content/cgmacros/1.0.0/ (CC BY-NC-SA 4.0).
- USDA SR Legacy: see experiments/shanghai_t2dm/scripts/download_usda.py.
- Chinese food-composition lookup: https://nlc.chinanutri.cn/fq/ .

This candidate omits raw datasets, participant- and meal-level tables, record-specific portion corrections, downloaded food-composition values, fitted models and private development records. It is NOT a self-contained, one-command reproduction of the complete study. Analysis scripts are preserved implementations requiring local inputs, including:
- data/processed/shanghai_t2dm/external_cohort.csv
- data/processed/glucosl/split.csv
- data/processed/cgmacros/primary_cohort.csv
- data/processed/cgmacros/harmonized_outcomes_15min.csv
- data/metadata/shanghai_t2dm_food_mapping.csv
- data/interim/shanghai_t2dm/intake_line_nutrients.csv
- experiments/shanghai_t2dm/scripts/food_curation/amount_overrides.csv

The original evaluation script checks an integrity manifest and refuses repeated execution. The manifest and underlying development files are not included. That script cannot execute the validation directly from this candidate package. A complete data-and-workflow reproduction release requires separate review.

requirements.lock.txt records the original macOS environment and includes platform-specific packages; it is not a universal installation recipe. pyGAM post hoc analyses used a separate environment, as explained in their script.

## Interpretation
The ensemble averages member-level sensitivity--load products, rather than multiplying averaged branches. Curves describe model behavior, not independently validated causal effects or physiology. Intervals containing zero indicate uncertain differences, not equivalence.

## Availability and license
No new license is assigned by this candidate package. Source data and pretrained weights retain their respective terms. Code licensing and public availability will be decided before public release.

## Funding and declarations
Supported by the West China Hospital High-level Talents Support Program (No. ZYGD24008). The author confirmed no other funding and no competing interests.
