# Data sources and attribution

## ShanghaiT2DM

Use version 3 of the public dataset described by Zhao et al., Chinese diabetes datasets for data-driven machine learning, Scientific Data (2023).

- Dataset: https://doi.org/10.6084/m9.figshare.20444397.v3
- Article: https://doi.org/10.1038/s41597-023-01940-7
- Source license: CC BY 4.0.
- The downloader verifies the pinned archive MD5 and SHA-256 values.
- The portion-correction table in food_curation/amount_overrides.csv describes 70 corrections to public diet records. These annotations preserve source record identifiers so the published preparation rules can be applied. Attribute the source dataset when reusing those source-derived annotations.
- No CGM records, laboratory tables or full individual diet tables are redistributed here.

## CGMacros

Use CGMacros version 1.0.0, described by Das et al., CGMacros: a pilot scientific dataset for personalized nutrition and diet monitoring, Scientific Data (2025).

- Dataset: https://physionet.org/content/cgmacros/1.0.0/
- Source license: CC BY-NC-SA 4.0.
- The downloader checks the published study archive and associated source dictionaries against the pinned checksums.
- The official LICENSE.txt is empty; use the license stated on the PhysioNet dataset page.
- Files generated locally from these data remain subject to their source terms; the repository does not redistribute participant-level CGMacros tables.

## Chinese food-composition platform

- Platform: https://nlc.chinanutri.cn/fq/
- Provider: National Institute for Nutrition and Health, China CDC, together with the Chinese Nutrition Society.
- Provider information: https://www.chinanutri.cn/yyzyxxpt/sjzx/swcfsjk/201911/t20191128_207183.html
- The provider states that the data copyright and interpretation rights belong to the institute.
- The repository distributes food-name matches, source entry identifiers and recipe rules, not a copied nutrient database.
- Documentation and curation annotations are in English. Source-language food names and parser literals are stored as Unicode escapes to preserve exact matching without displaying source-language text in the repository. The mapping loader decodes escaped CSV values before applying the rules; identifiers, portions and recipe coefficients are unchanged.
- Preparation retrieves source values into a local cache. Users must obtain access under the source terms. The cache is excluded from Git.
- This platform is updated over time and has no versioned dataset archive linked by this code. Changes in source values can alter nutrient estimates. The verification command reports differences from the study's reference estimates rather than silently updating those references.
- If network access is unavailable, provide a locally obtained cache through --cn-cache-dir and use --offline.

## USDA SR Legacy

- Provider: USDA FoodData Central.
- Archive: https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_csv_2018-04.zip
- Database: SR Legacy, April 2018.
- USDA FoodData Central data are public-domain data. Cite the source when using the food-composition values.
- The downloader verifies the archive SHA-256 before extraction.

## TabPFN v2

- Implementation and model-version instructions: https://github.com/PriorLabs/TabPFN
- Classifier: https://huggingface.co/Prior-Labs/TabPFN-v2-clf
- Regressor: https://huggingface.co/Prior-Labs/TabPFN-v2-reg
- The v2 weights have a separate Prior Labs license and attribution requirements. Obtain weights through the library or the official source. No weights or account credentials are included in this repository.

## Local outputs

Raw data, food-composition responses, processed clinical/meal tables, fitted models and individual predictions are generated or retrieved locally. They are excluded by .gitignore. Users should not infer that the code repository's visibility changes the terms of any source dataset.
