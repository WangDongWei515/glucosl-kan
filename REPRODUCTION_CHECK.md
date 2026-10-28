# Release verification

Checked on 28 October 2026 using Python 3.12.13 on macOS with the study package versions.

## Executed

1. Rebuilt the CGMacros and ShanghaiT2DM preparation pipeline from the original official raw files. A local source food-composition cache was used for this offline reconstruction; that cache is not redistributed.
2. Confirmed the exact study cohort sizes: Shanghai A, 67 patients and 1,952 meals; Shanghai B, 33 and 670; CGMacros, 44 and 1,147; diabetes-range HbA1c subset, 14 and 353; more specifically mapped B subset, 347 meals. No training--validation patient overlap was found.
3. Refit the core model group from the reconstructed data: GlucoSL-KAN, XGBoost, the constant model, US-trained GlucoSL-KAN and cooked-weight sensitivity variants, for the applicable regression and classification tasks.
4. Recomputed core validation metrics. All 206 point estimates matched the supplied study reference values exactly; maximum absolute difference was 0.0.
5. Exercised patient-cluster bootstrapping with 20 draws. All 42 applicable headline intervals were defined, and the point estimates still matched the reference values.
6. Recomputed the complete model group's validation point estimates using copies of the existing study fits in a separate local output directory. All 416 point estimates matched exactly; maximum absolute difference was 0.0. This check tested full-model evaluation, not a fresh fit of the complete model group.
7. Ran the descriptive post hoc workflow with the complete evaluation outputs, including patient ranking, clinical associations, insulin subgroups, meal-period adjustment and monotonicity checks.
8. Passed seven implementation checks: positive-area units, baseline crossing, exclusion of negative area, patient averaging, repeated bootstrap clusters, training-only preprocessing and matching parsed source food names to the escaped curation rules.
9. Queried the live Chinese food-composition endpoint successfully. The first page returned 8 rows and the service reported 170 pages.

The original study model implementations, final configuration and saved scientific results were not modified by these checks. Fitted models and all individual data outputs remained local and are excluded from Git.

English-language documentation and annotations were checked across all tracked files. Source-language matching keys are stored with Unicode escapes. Decoding preserved all 7,252 non-annotation curation cells, and rebuilding the preparation pipeline produced seven processed CSV files byte-for-byte identical to those before this presentation change.

## Not claimed by this check

- The full comparator set was not retrained during this release check. Its original fitting and tuning implementations are available through --models all.
- The full 2,000-draw bootstrap was not rerun during this release check. Use --bootstrap-draws 2000 for that procedure; the 20-draw check is not a replacement for the paper's confidence intervals.
- All raw archives were not downloaded again. The reconstruction used the original locally available official files, and download scripts retain their pinned source checksums.
- The original GAM environment versions were not retained. requirements-gam.txt specifies a compatible separate environment; it is not an exact historical lock file. The optional expanded-search/GAM post hoc comparators were not refitted during this release check.
- Cross-platform bitwise equality is not guaranteed. Compare numerical parameters, predictions and metric values rather than serialized joblib bytes.
- The food-composition service is an external, updated source. Changes to its values can change derived meal nutrients; verify reports discrepancies instead of replacing study reference results.

The repository supplies a numerical reproduction workflow. It does not redistribute raw participant data, the source nutrient database or third-party pretrained weights.
