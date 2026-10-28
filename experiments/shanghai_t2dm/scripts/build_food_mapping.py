from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parent
STAGE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPTS))
from query_cncdc import CACHE, cached_rows

INTERIM = STAGE / 'data/interim/shanghai_t2dm'
METADATA = STAGE / 'data/metadata'
CURATION = SCRIPTS / 'food_curation'
USDA = STAGE / 'data/raw/usda_fdc/sr_legacy_2018-04/extracted/FoodData_Central_sr_legacy_food_csv_2018-04'
NUTRIENTS = ['energy_kcal', 'protein_g', 'fat_g', 'carbohydrate_g', 'fiber_g']
USDA_NUTRIENTS = {1008: 'energy_kcal', 1003: 'protein_g', 1004: 'fat_g', 1005: 'carbohydrate_g', 1079: 'fiber_g', 1051: 'water_g'}
USDA_COLUMNS = [f'{key}_usda' for key in NUTRIENTS]
SHARE_COLUMNS = ['fiber_reported_share', 'fiber_usda_share', 'fiber_imputed_share', 'added_oil_g_per_100g']
NUTRIENT_COLUMNS = NUTRIENTS + SHARE_COLUMNS + USDA_COLUMNS + ['usda_matched_share']
OIL, WATER, OIL_SHARE = '1516', 'W0', 0.05
SPECIFIC_TOP, REVIEW_TOP = 300, 200
KJ_PER_KCAL, ETHANOL_KJ, CHO_KJ = 4.184, 29.0, 17.0
IN_ENTRY_FRIED = {'fried_wheat', 'fried_rice_noodles'}
FRIED_COOKING = ('fried_named', 'fried_dish_type', 'fried_default')
OIL_BASIS = {'fried_named': 'named', 'fried_dish_type': 'dish_type', 'fried_in_entry': 'in_entry', 'oil_in_recipe': 'in_recipe'}
# Oil rules O1 (unspecified single vegetables) and O2 (dishes fried by default in Shanghai), from the mapping audit.
O1_GROUPS = {'leafy_veg', 'fruit_veg', 'root_stem_veg', 'fungi_algae', 'fresh_legume', 'mixed_veg'}
O1_EXCLUDE = '\u9ec4\u74dc|\u9752\u74dc|\u756a\u8304|\u897f\u7ea2\u67ff|\u5723\u5973\u679c|\u5357\u74dc|^\u59dc|^\u8471$|\u59dc\u7247|\u849c\u84c9|\u7d2b\u83dc|\u94f6\u8033|\u767d\u6728\u8033|\u9b54\u828b|\u7ea2\u8c46|\u8d64\u8c46|\u7eff\u8c46(?!\u82bd)|\u9ec4\u8c46(?!\u82bd)|\u9e70\u5634\u8c46|\u8131\u6c34|\u6c34\u679c\u841d\u535c'
O2_ITEMS = '\u5e26\u9c7c|\u5927\u6392|\u5c0f\u9ec4\u9c7c|\u725b\u6392'
# Line-level weight states: raw staples in self-reported records, dry or soup noodles, wonton broth, prepared cereal and corn on the cob.
RICE_KEYS = {'rice'}
COARSE_KEYS = {'coarse grain', 'coarase grain', 'coarce grain', 'corase grain', 'coarse grain rice', 'coarse grain 13'}
STATE_RECIPES = {'raw_rice': '280:1', 'raw_grain': '280:0.48;301:0.115;1286:0.115;309:0.115;351:0.0875;286:0.0875', 'dry_noodle': '261:1',
                 'noodle_soup': 'W0:0.58;264:0.27;452:0.15', 'wonton_broth': '1311:0.5;W0:0.5', 'prepared_cereal': 'W0:0.86;1286:0.14'}
STATE_RECIPES_BY_STATE = {'raw_staple': lambda row: 'raw_rice' if row.key in RICE_KEYS else 'raw_grain', 'dry_noodle': lambda row: 'dry_noodle',
                          'noodle_soup': lambda row: 'noodle_soup', 'wonton_broth': lambda row: 'wonton_broth', 'prepared_cereal': lambda row: 'prepared_cereal'}
CORN_EDIBLE, CORN_MIN, MAX_LINE_G, RAW_STAPLE_MAX = 0.46, 150, 800, 150
VARIANTS = ['primary', 'no_oil', 'group_only', 'usda', 'staples_cooked']
NOT_A_DISH = {'rice', 'grain_rice', 'porridge', 'steamed_bread', 'bread_flatbread', 'glutinous_ricecake', 'dry_cereal_powder', 'milk', 'yogurt', 'milk_mixed',
              'soy_milk', 'fruit_temperate', 'fruit_citrus', 'fruit_melon', 'fruit_tropical', 'dried_fruit', 'nuts_seeds', 'cake_pastry', 'biscuit_snack',
              'sugar_sweets', 'ice_cream', 'glutinous_sweet', 'water_tea', 'sweet_drink', 'beer', 'wine', 'rice_wine', 'condiment', 'pickled_veg', 'western_fastfood'}

# Cooking-oil rule: these characters mark a fried dish unless they belong to one of the exempt words (ingredients or fixed names).
OIL_TERMS = '\u7092|\u714e|\u70b8|\u7206|\u7178|\u6e9c|\u7198|\u70e7|\u6cb9|(?<!\u6d0b)\u8471\u70e4'
OIL_EXEMPT = '\u6cb9\u9ea6\u83dc|\u9171\u6cb9|\u869d\u6cb9|\u8017\u6cb9|\u5976\u6cb9|\u9ec4\u6cb9|\u725b\u6cb9\u679c|\u9ebb\u6cb9|\u6cb9\u83dc|\u6cb9\u8c46\u8150|\u6cb9\u9762\u7b4b|\u6cb9\u6761|\u6cb9\u997c|\u6cb9\u58a9\u5b50|\u6cb9\u6d78|\u70e7\u9ea6|\u70e7\u5356|\u70e7\u997c|\u70e7\u9e45|\u53c9\u70e7'
NONOIL_TERMS = '\u84b8|\u716e|\u767d\u707c|\u767d\u5207|\u767d\u65a9|\u6c64|\u7fb9|\u62cc|\u51c9|\u7096|\u5364|\u76d0\u6c34|\u9189|\u7cdf|\u767d\u6c34|\u712f|\u8272\u62c9|\u6c99\u62c9|\u6dae|\u70eb|\u70e4|(?<![\u4eac\u751c\u7092])\u9171(?=[\u725b\u9e2d\u6392\u732a\u7f8a\u9e21\u9999])'
AROMATIC = '\u5c0f\u8471|\u9999\u8471|(?<![\u5927\u4eac\u6d0b])\u8471(?:\u82b1|\u6cb9|\u70e4|\u7206|\u9999)?(?!\u5934)|\u59dc|\u849c\u84c9|\u849c\u6ce5|\u849c\u9999|\u9999\u83dc|\u82ab\u837d|\u5241\u6912|\u82b1\u6912|\u6912\u76d0|\u9ed1\u6912|\u9ed1\u80e1\u6912|\u80e1\u6912|\u5496\u55b1'
# Food components used to recognise composite dishes; matched in this order on the Chinese name, each match removed before the next kind.
PROTEIN = {
    'egg': '(?<!\u65e0)(?:\u9e21|\u9e2d|\u9e45|\u9e4c\u9e51|\u54b8|\u76ae|\u677e\u82b1|\u8336\u53f6|\u5364|\u8377\u5305|\u6c34\u716e|\u767d\u716e)?\u86cb(?!\u7cd5|\u631e|\u9ec4\u9165|\u9165)(?:\u767d|\u9ec4|\u6e05|\u76ae)?',
    'soy': '\u8c46\u8150(?:\u5e72|\u8863|\u76ae|\u7ed3)?|\u8c46\u5e72|\u9999\u5e72|\u5e72\u4e1d|\u767e\u53f6(?:\u7ed3|\u4e1d)?|\u5343\u5f20|\u7d20\u9e21|\u8150\u7af9|\u8c46\u76ae|\u8150\u76ae|\u70e4\u9eb8|\u6cb9?\u9762\u7b4b|\u8c46\u82b1|\u7eb3\u8c46|\u9a6c\u5170\u8c46',
    'processed': '\u9999\u80a0|\u706b\u817f(?:\u80a0|\u7247)?|\u5348\u9910\u8089|\u54b8\u8089|\u814a\u8089|\u57f9\u6839|\u7ea2\u80a0|\u8089\u80a0|\u8089\u677e|\u814a\u80a0|\u54b8\u8e44|\u718f\u8089|(?:\u725b)?\u8089\u5e72',
    'meatball': '(?:\u9e21|\u725b)?\u8089(?:\u5706|\u4e38)(?:\u5b50)?|(?<!\u852c\u83dc)\u4e38\u5b50|\u72ee\u5b50\u5934|(?:\u725b)?\u8089\u997c|\u8d21\u4e38|\u8089\u5377|(?<!\u9152\u917f)\u5706\u5b50',
    'offal': '(?:\u732a|\u9e2d|\u9e21|\u9e45|\u725b|\u7f8a)?(?:\u809d|\u809a(?!\u88c6)|\u5927\u80a0|\u80a5\u80a0|\u80d7|\u80ab|\u80ba)|(?:\u9e2d|\u732a)\u8840|\u8840\u65fa|\u732a\u8033|\u725b\u7b4b|\u8e44\u7b4b|\u725b\u820c|\u8170\u82b1|\u9c7c\u80a0|\u9e2d\u5934|\u9e2d\u8116|\u9e2d\u80a0',
    'shellfish': '\u9c8d\u9c7c|\u9c7f\u9c7c|\u58a8\u9c7c\u4ed4?|\u4e4c\u8d3c|\u7ae0\u9c7c|\u76ee\u9c7c|\u9c7c\u7fc5|\u4f5b\u8df3\u5899|\u6d77\u9c9c(?!\u83c7)|\u86e4\u870a\u8089?|\u82b1\u86e4|\u6587\u86e4|\u86e4|\u86cf\u5b50|\u86cf|\u6247\u8d1d|\u8d1d\u7c7b|\u8d1d|\u7530\u87ba|\u9999\u87ba|\u87ba|\u6cb3\u868c|\u868c|\u751f\u869d|\u869d|\u7261\u86ce|(?<!\u674f)\u9c8d|\u6d77\u53c2|\u9c7f|\u6d77\u8707(?:\u76ae|\u5934|\u4e1d)?|\u82b1\u7532|\u4e09\u9c9c',
    'fish_processed': '\u9c7c\u4e38|\u9c7c\u5706|\u867e\u4e38|\u867e\u6ed1|\u9c7c\u6392|\u87f9\u8db3\u68d2|\u9c7c\u7247\u5e72|\u9c7c\u8c46\u8150',
    'crab': '(?:\u5927\u95f8|\u68ad\u5b50|\u6bdb|\u6cb3|\u6d77|\u818f|\u7687\u5e1d|\u7ea2\u6bdb)?\u87f9(?!\u5473\u83c7)(?:\u8089|\u7c89|\u9ec4)?',
    'shrimp': '(?:\u57fa\u56f4|\u56f4\u57fa|\u6cb3|\u6d77|\u8349|\u767d\u7c73|\u751c|\u5927|\u5c0f\u9f99)?\u867e(?:\u4ec1|\u8089|\u76ae|\u7c73)?',
    'fish': '(?:\u9cab|\u9c88|\u9cb3|\u9cca|\u9cdd|\u9cd7|\u9c91|\u9cd5|\u9ca2|\u9cd9|\u9ca4|\u4e09\u6587|\u5df4\u6c99|\u6ce5\u9cc5|\u6602\u523a|\u9cdc)?\u9c7c(?!\u9999)(?:\u8089|\u7247|\u5757|\u5934|\u5c3e|\u4e1d)?|\u9cab|\u9c88|\u9cb3|\u9cca|\u9cdd|\u9cd7|\u9c91|\u9cd5|\u9ca2|\u9cd9|\u9ca4|\u4e09\u6587|\u5df4\u6c99|\u6ce5\u9cc5|\u6602\u523a|\u9cdc',
    'beef': '\u725b(?!\u5976|\u6cb9|\u5fc3\u83dc|\u809d\u83cc|\u86d9|\u8f67)(?:\u8089|\u8169|\u8171|\u6392|\u5c3e)?(?:\u4e1d|\u7247|\u7c92|\u672b|\u5757)?',
    'mutton': '\u7f8a(?!\u89d2|\u5976)(?:\u8089|\u6392|\u874e\u5b50|\u9aa8\u68d2?)?(?:\u4e1d|\u7247|\u5757)?',
    'chicken': '(?<!\u7d20)(?:\u9e21(?!\u86cb|\u6bdb\u83dc|\u817f\u83c7|\u7cbe|\u679e)|\u9e3d|\u9e4c\u9e51(?!\u86cb)|\u725b\u86d9|\u96ea\u86e4|\u51e4\u722a)(?:\u5b50)?(?:\u80f8|\u817f|\u4e2d\u7fc5|\u7fc5|\u5757|\u722a|\u6392|\u67f3)?(?:\u8089)?(?:\u4e1d|\u7247|\u4e01|\u5757)?',
    'duck': '(?:\u9e2d(?!\u86cb|\u8840|\u68a8|\u80d7|\u80ab|\u80a0)|\u9e45(?!\u86cb|\u809d))(?:\u5b50)?(?:\u817f|\u7fc5|\u8180|\u638c|\u722a|\u80f8|\u812f|\u5757|\u7247|\u4e01)?(?:\u8089)?',
    'pork': '\u732a(?:\u8089|\u6392|\u8e44|\u811a|\u722a)?|\u8089(?:\u4e1d|\u7247|\u4e01|\u672b|\u6cab|\u7cdc)?|\u6392\u9aa8|\u5927\u6392|\u5c0f\u6392|\u6392\u6761|\u8e44\u9ac8|\u8e44\u8180|\u8e44|\u4e94\u82b1|\u91cc\u810a|\u8098|\u7b52\u9aa8|\u9aa8\u5934|\u9aa8',
}
VEG = {
    'pickle': '\u69a8\u83dc|\u54b8\u83dc|\u9171\u83dc|\u6ce1\u83dc|\u9178\u83dc|\u96ea\u83dc|\u6885\u5e72\u83dc|\u6885\u83dc|\u841d\u535c\u5e72|\u83dc\u5e72|\u5927\u5934\u83dc|\u9171\u841d\u535c|\u814c|\u8fa3\u767d\u83dc|\u6a44\u6984\u83dc|\u9178\u7b0b|\u5e72\u83dc',
    'special_leafy': '\u5730\u74dc\u53f6|\u7ea2\u85af\u53f6|\u6728\u8033\u83dc|\u8c4c\u8c46\u82d7|\u8c46\u82d7|\u9e21\u6bdb\u83dc|\u6cb9\u9ea6\u83dc',
    'sprout': '\u9ec4\u8c46\u82bd|\u7eff\u8c46\u82bd|\u8c46\u82bd',
    'fungi': '(?<![\u8328\u6148])\u83c7|\u8611\u83c7?|\u6728\u8033|\u94f6\u8033|\u83cc\u83c7?|\u677e\u8338|\u6d77\u5e26(?:\u7ed3|\u4e1d)?|\u7d2b\u83dc|\u82d4\u6761|\u88d9\u5e26\u83dc?|\u6d77\u82d4|\u7334\u5934|\u7af9\u836a',
    'starchy': '\u571f\u8c46(?:\u4e1d|\u6ce5|\u7247)?|\u9a6c\u94c3\u85af|\u7ea2\u85af|\u5c71\u828b|\u5730\u74dc|\u7d2b\u85af|(?<!\u9b54)\u828b(?:\u827f|\u5934|\u5976)?|\u5c71\u836f|\u85d5(?:\u7247|\u4e01)?|\u7389\u7c73(?:\u7c92)?|\u6817\u5b50|\u677f\u6817|\u83f1\u89d2|\u6728\u85af',
    'legume': '\u6bdb\u8c46|\u8695\u8c46|\u8c4c\u8c46|\u9752\u8c46|\u9ec4\u8c46|\u7ea2\u8c46|\u8d64\u8c46|\u7eff\u8c46|\u9e70\u5634\u8c46|\u7709\u8c46|\u8c46\u5b50',
    'root_stem': '\u80e1\u841d\u535c|\u767d\u841d\u535c|\u841d\u535c|\u80e1\u535c|\u83b4\u7b0b|\u83b4\u82e3|\u832d\u767d|\u897f\u82b9|\u6c34\u82b9|\u82b9\u83dc|\u82b9|\u82a6\u7b0b|\u7f57\u7b0b|\u51ac\u7b0b|\u7af9\u7b0b|\u7b0b|\u6d0b\u8471|\u5927\u8471|\u4eac\u8471|\u849c\u82d7|\u849c\u85b9|\u849c\u82d4|\u849c\u53f0|\u9752\u849c|\u5927\u849c|\u767e\u5408|\u8328\u83c7|\u6148\u83c7|\u8378\u8360|\u9a6c\u8e44|\u9ec4\u82b1\u83dc|\u9ec4\u82b1|\u91d1\u9488\u83dc|\u8568\u83dc',
    'fruit_veg': '\u9ec4\u74dc|\u9752\u74dc|\u51ac\u74dc|\u4e1d\u74dc|\u82e6\u74dc|\u5357\u74dc|\u897f\u846b\u82a6|\u591c\u5f00\u82b1|\u846b\u82a6|\u74e0|\u4f5b\u624b\u74dc|\u756a\u8304|\u897f\u7ea2\u67ff|\u8304\u5b50|\u8304|\u9752\u6912|\u7ea2\u6912|\u5f69\u6912|\u8fa3\u6912|\u751c\u6912|\u8c47\u8c46|\u7f38\u8c46|\u56db\u5b63\u8c46|\u5200\u8c46|\u8c46\u89d2|\u8377\u5170\u8c46|\u6241\u8c46|\u751c\u8c46|\u8c46\u835a|\u79cb\u8475|\u82b8\u8c46|\u9f99\u8c46|\u5723\u5973\u679c',
    'leafy': '\u5c0f\u9752\u83dc|\u9752\u83dc|\u83e0\u83dc|\u6ce2\u83dc|\u751f\u83dc|\u5c0f\u767d\u83dc|\u5927\u767d\u83dc|\u767d\u83dc|\u5377\u5fc3\u83dc|\u5377\u82af\u83dc|\u5305\u83dc|\u7518\u84dd|\u9ec4\u82bd\u83dc|\u9ec4\u7259\u83dc|\u6bdb\u83dc|\u83dc\u5fc3|\u5854\u83dc|\u7a7a\u5fc3\u83dc|\u7a7a\u83dc|\u8579\u83dc|\u833c\u84bf|\u84ec\u84bf|\u82cb\u83dc|\u7c73\u82cb|\u82cb|\u8349\u5934|\u8360\u83dc|\u82a5\u5170|\u82a5\u84dd|\u83dc\u85b9|\u5a03\u5a03\u83dc|\u97ed\u83dc|\u97ed\u9ec4|\u97ed\u82bd|\u97ed|\u676d\u767d\u83dc|\u725b\u5fc3\u83dc|\u82b1\u83dc|\u83dc\u82b1|\u82b1\u6930\u83dc|\u897f\u5170\u82b1|\u897f\u84dd\u82b1|\u852c\u83dc\u53f6|\u9a6c\u5170|\u9999\u693f|\u82e6\u83ca|\u6cb9\u83dc',
    'generic': '\u852c\u83dc|\u7d20\u83dc|\u7092\u83dc|\u4ec0\u9526|\u8364\u7d20|\u679c\u852c|\u6c34\u716e\u83dc|\u70e9\u83dc|\u4e09\u4e1d|\u7092\u7d20',
    'starch_noodle': '\u7c89\u4e1d|\u7c89\u76ae|\u51c9\u76ae|\u9b54\u828b|\u7c73\u7ebf|\u7c73\u7c89|\u5bbd\u7c89|\u7c89\u6761|\u80a0\u7c89',
}
# Dish or food identities that decide the group before component logic; first match wins.
DIRECT = [
    ('unmappable', '\u7a7a\u8179'), ('water_tea', '\u4e2d\u836f'), ('protein_powder', '\u86cb\u767d\u7c89'),
    ('biscuit_snack', '\u5a01\u5316|\u997c\u5e72|\u82cf\u6253|\u71d5\u9ea6\u68d2|\u85af\u7247|\u85af\u6761|\u9505\u5df4|\u81a8\u5316|\u96f6\u98df|\u867e\u7247|\u6d0b\u8471\u5708|\u5170\u82b1\u8c46|\u8106\u7247|\u7c73\u679c|\u96ea\u997c'),
    ('beer', '\u5564\u9152'), ('wine', '\u8461\u8404\u9152|\u7ea2\u9152|\u767d\u9152|\u53c2\u9152'),
    ('glutinous_sweet', '\u9152\u917f|\u6c64\u5706|\u9ebb\u85af|\u7ccd\u7c91|\u9752\u56e2|\u897f\u7c73|\u7c73\u9152\u4e38|\u4e38\u5b50.*\u9152|\u9152.*\u4e38\u5b50'),
    ('rice_wine', '\u9ec4\u9152|\u7c73\u9152|\u82b1\u96d5'),
    ('milk_mixed', '\u5976\u8336|\u62ff\u94c1|\u5496\u5561.*\u5976|\u5976.*\u5496\u5561|\u5976\u5496|\u725b\u5976.*\u9ea6\u7247|\u9ea6\u7247.*\u725b\u5976|\u71d5\u9ea6\u725b\u5976|\u725b\u5976\u52a0|\u725b\u5976.*\u86cb|\u725b\u5976.*\u9762\u5305'),
    ('sweet_drink', '\u53ef\u4e50|\u679c\u6c41|\u6a59\u6c41|\u6930\u5b50\u6c41|\u6930\u6c41|\u6930\u6d46|\u7cd6\u6c34|\u9178\u6885|\u67e0\u6aac\u8336|\u7389\u7c73\u6c41|\u996e\u6599|\u82f9\u679c\u918b|\u679c\u852c\u6d46|\u96ea\u78a7'),
    ('water_tea', '^\u6c34$|\u5496\u5561|\u6e05\u5496|^\u8336$|\u7eff\u8336|\u7ea2\u8336|\u82b1\u8336|\u59dc\u6c41|\u59dc\u6c64|\u67ab\u6597|\u716e\u6c34$|\u529b\u5ea6\u4f38'),
    ('yogurt', '\u9178\u5976(?!\u6cb9)|\u53d1\u9175\u4e73'), ('soy_milk', '\u8c46\u6d46|\u8c46\u5976|\u54b8\u6d46'), ('milk', '\u725b\u5976|\u9c9c\u5976|^\u5976$|\u8212\u5316'),
    ('dry_cereal_powder', '\u5976\u7c89|\u4ee3\u9910|\u8425\u517b\u7c89|\u829d\u9ebb\u7cca\u7c89|\u858f\u7c73\u7c89|\u9ea6\u7247(?!\u7ca5|\u7cca)|\u71d5\u9ea6(?!\u7ca5|\u996d)'),
    ('salad', '\u8272\u62c9|\u6c99\u62c9(?!\u9171)'), ('mixed_veg', '^\u83dc$'),
    ('meat_veg_stirfry', '\u706b\u9505|\u9ebb\u8fa3\u70eb|\u5192\u83dc|\u8364\u83dc|^\u8364\u7d20$'),
    ('meat_veg_steamed', '\u5927\u6742\u70e9|\u5173\u4e1c\u716e'),
    ('sweet_soup', '\u6843\u80f6|\u96ea\u86e4'),
]
SOUP = '\u6c64(?!\u5706|\u5305|\u996d)|\u7fb9|(?<!\u70c2)\u7cca|\u7f57\u5b8b'
STARCHY_DISH = '\u571f\u8c46|\u9a6c\u94c3\u85af|\u5c71\u836f|\u85d5|\u7389\u7c73|(?<!\u9b54)\u828b|\u5357\u74dc|\u7c89\u4e1d|\u6bdb\u8c46|\u6817\u5b50'
LIST = '[\u3001\uff0c,]|\u548c|\u52a0|\u53ca|\u4e0e|\u5171'
SWEET_SOUP = '\u8d64\u8c46|\u7ea2\u8c46|\u7eff\u8c46(?!\u82bd)|\u94f6\u8033|\u767d\u6728\u8033|\u7ea2\u67a3|\u83b2\u5b50|\u83b2\u5fc3|\u767e\u5408|\u6843\u80f6|\u96ea\u86e4|\u829d\u9ebb|\u6838\u6843|\u5357\u74dc\u7cca|\u7cd6\u6c34|\u832f\u82d3|\u5c71\u836f\u7fb9|\u7ea2\u85af\u6c64'
FILLED_VEG = '\u7d20|\u83dc\u5305|\u8c46\u6c99|\u8c46\u5305|\u83dc\u9992\u5934|\u7c89\u4e1d\u5305|\u841d\u535c\u4e1d|\u80e1\u841d\u535c\u5305|\u6885\u5e72\u83dc|\u9999\u83c7\u83dc|\u9999\u83c7\u5305|\u97ed\u83dc|\u9752\u83dc'
STAPLE = [
    ('porridge', '\u7ca5|\u7a00\u996d|\u6ce1\u996d|\u6c64\u996d'),
    ('fried_rice_noodles', '\u7092\u996d|\u7092\u9762|\u7092\u7c73\u7c89|\u7092\u7c89|\u7092\u5e74\u7cd5|\u7092\u7c73\u7ebf|\u7092\u6cb3\u7c89|\u7092\u9762\u7247|\u7092\u610f\u9762'),
    ('fried_wheat', '\u65b9\u4fbf\u9762|\u6ce1\u9762|\u6cb9\u6761|\u6cb9\u997c|\u9ebb\u82b1|\u6cb9\u58a9\u5b50|\u6625\u5377|\u8471\u6cb9\u997c|\u5343\u5c42\u997c|\u7126\u5708|\u624b\u6293\u997c|\u714e\u997c|\u86cb\u997c|\u9171\u9999\u997c|\u82d4\u6761\u997c|\u841d\u535c\u4e1d\u997c|\u76d2\u5b50|\u5408\u5b50'),
    ('fried_dumpling', '\u751f\u714e|\u9505\u8d34|\u714e\u997a|\u714e\u5305|\u714e\u9984\u9968'),
    ('filled', '\u997a|\u9984\u9968|\u4e91\u541e|\u70e7\u9ea6|\u70e7\u5356|\u5c0f\u7b3c|\u5305\u5b50|\u8089\u5305|\u83dc\u5305|\u8c46\u5305|\u8c46\u6c99\u5305|\u7d20\u5305|\u84b8\u5305|\u5c0f\u5305|\u7c89\u4e1d\u5305|\u9999\u83c7\u5305|\u6885\u5e72\u83dc\u5305|\u5357\u74dc\u5305|\u8089\u9992\u5934|\u725b\u8089\u9992\u5934|\u83dc\u9992\u5934'),
    ('western_fastfood', '\u6c49\u5821|\u62ab\u8428|\u6bd4\u8428|\u4e09\u660e\u6cbb|\u70ed\u72d7'),
    ('bread_flatbread', '\u9762\u5305|\u5410\u53f8|\u571f\u53f8|\u829d\u58eb\u5305'),
    ('steamed_bread', '\u9992\u5934|\u82b1\u5377|\u7a9d\u5934|\u53d1\u7cd5|\u7a9d\u7a9d|\u6742\u7cae\u5305'),
    ('cake_pastry', '\u86cb\u7cd5|\u7cd5\u70b9|\u6708\u997c|\u9165(?!\u9e21|\u9e2d)|\u86cb\u631e|\u6ce1\u8299|\u745e\u58eb\u5377|\u897f\u70b9|\u70b9\u5fc3|\u5c71\u6942\u7cd5|\u7eff\u8c46\u7cd5|\u8c46\u7cd5|\u5357\u74dc\u997c|\u725b\u5976\u997c|\u51e4\u68a8|\u86cb\u5377'),
    ('meatball', '^(\u725b)?\u8089\u997c$'),
    ('flatbread', '\u997c'),
    ('rice_cake_dish', '\u5e74\u7cd5(?=.)|.(?=.*\u5e74\u7cd5)'),
    ('glutinous_ricecake', '\u7cbd|\u68d5|\u5e74\u7cd5|\u7c73\u7cd5|\u7cef\u7c73|\u7ca2\u996d|\u996d\u56e2|\u9152\u917f\u7cd5|\u9ed1\u7c73\u7cd5'),
    ('grain_rice', '\u6742\u7cae|\u7cd9\u7c73|\u9ed1\u7c73|\u7ea2\u7c73|\u835e\u9ea6\u996d|\u835e\u9ea6\u7c73|\u835e\u9ea6\u7ea2\u7c73|\u9ad8\u7cb1|\u5c0f\u7c73\u996d|\u85dc\u9ea6|\u858f\u7c73.*\u996d|\u5927\u7c73\u52a0\u5c0f\u7c73|\u858f\u4ec1|\u4e94\u8c37'),
    ('rice', '\u7c73\u996d|\u767d\u996d|^\u996d|\u83dc\u996d|\u62cc\u996d|\u5bff\u53f8|\u4e3b\u98df'),
    ('noodles', '\u9762\u6761|\u9762$|\u62cc\u9762|\u6c64\u9762|\u9762\u7247|\u610f\u9762|\u610f\u5927\u5229\u9762|\u51b7\u9762|\u51c9\u9762|\u70ed\u5e72\u9762|\u835e\u9ea6\u9762|\u5377\u9762|\u6302\u9762|\u9762\u7599\u7629'),
    ('sugar_sweets', '\u5de7\u514b\u529b|\u7cd6\u679c|\u5976\u7cd6|\u592a\u5983|\u829d\u9ebb\u7cd6|\u9ebb\u7cd6|\u8461\u8404\u7cd6|\u767d\u7cd6|\u51b0\u7cd6|\u7ea2\u7cd6|\u8702\u871c|\u679c\u9171|\u84dd\u8393\u9171|\u8349\u8393\u9171|^\u7cd6$|\u725b\u8f67'),
    ('ice_cream', '\u51b0\u6dc7\u6dcb|\u96ea\u7cd5|\u5e03\u4e01|\u679c\u51bb|\u51b0\u68cd|\u51b0\u6fc0\u51cc'),
    ('dried_fruit', '\u67a3|\u8461\u8404\u5e72|\u67b8\u675e|\u6842\u5706\u5e72|\u679c\u5e72|\u5e72\u679c|\u674f\u5e72'),
    ('nuts_seeds', '\u6838\u6843|\u82b1\u751f|\u74dc\u5b50|\u674f\u4ec1|\u8170\u679c|\u677e\u4ec1|\u677e\u5b50|\u575a\u679c|\u829d\u9ebb|\u699b\u5b50|\u5f00\u5fc3\u679c|\u5357\u74dc\u5b50|\u5357\u74dc\u7c7d|\u897f\u74dc\u5b50|\u590f\u5a01\u5937|\u5947\u4e9a\u7c7d|\u4e9a\u5947\u7c7d|\u4e9a\u9ebb\u7c7d|\u5df4\u65e6\u6728'),
    ('fruit_citrus', '\u6a59|\u6a58|\u6854|\u67d1|\u67da|\u67e0\u6aac'),
    ('fruit_melon', '\u897f\u74dc|\u751c\u74dc|\u54c8\u5bc6\u74dc|\u9999\u74dc|\u767d\u5170\u74dc|\u871c\u74dc|\u9ec4\u91d1\u74dc'),
    ('fruit_tropical', '\u9999\u8549|\u8292\u679c|\u8354\u679d|\u9f99\u773c|\u6842\u5706|\u69b4\u83b2|\u83e0\u841d|\u706b\u9f99\u679c|\u5c71\u7af9|\u6930\u5b50|\u725b\u6cb9\u679c|\u767e\u9999\u679c|\u6728\u74dc|\u6768\u6843'),
    ('fruit_temperate', '\u82f9\u679c|\u68a8|(?<!\u6838)\u6843(?!\u80f6)|\u674e\u5b50|\u674f(?!\u9c8d)|\u6a31\u6843|\u8f66\u5398\u5b50|\u8461\u8404|\u63d0\u5b50|\u7315\u7334\u6843|\u5947\u5f02\u679c|\u8349\u8393|\u84dd\u8393|(?<!\u897f\u7ea2)\u67ff|\u6787\u6777|\u5c71\u6942|\u83b2\u96fe|\u4eba\u53c2\u679c|\u4eba\u751f\u679c|\u77f3\u69b4|\u65e0\u82b1\u679c|^\u6c34\u679c$'),
    ('condiment', '^(\u9171\u6cb9|\u918b|\u8fa3\u9171|\u756a\u8304\u9171|\u6c99\u62c9\u9171|\u8272\u62c9\u9171|\u8150\u4e73|\u4e73\u8150|\u7ea2\u8150\u4e73|\u82a5\u672b|\u849c\u84c9|\u9178\u5976\u6cb9|\u8c46\u74e3\u9171|\u751c\u9762\u9171|\u869d\u6cb9|\u9171)$'),
]
VEG_GROUP = {'leafy': 'leafy_veg', 'special_leafy': 'leafy_veg', 'sprout': 'fruit_veg', 'fruit_veg': 'fruit_veg', 'root_stem': 'root_stem_veg', 'starchy': 'starchy_veg', 'legume': 'fresh_legume',
             'fungi': 'fungi_algae', 'generic': 'mixed_veg', 'starch_noodle': 'starch_noodles'}
FRIED_VARIANT = {'leafy_veg': 'leafy_veg_fried', 'fruit_veg': 'fruit_veg_fried', 'root_stem_veg': 'root_stem_veg_fried', 'starchy_veg': 'starchy_veg_fried',
                 'fresh_legume': 'fresh_legume_fried', 'fungi_algae': 'fungi_algae_fried', 'mixed_veg': 'mixed_veg_fried', 'pork': 'pork_fried',
                 'pork_rib': 'pork_rib_fried', 'meatball': 'meatball_fried', 'beef_mutton': 'beef_mutton_fried', 'chicken': 'chicken_fried',
                 'duck_goose': 'duck_fried', 'fish': 'fish_fried', 'shrimp': 'seafood_fried', 'shellfish': 'seafood_fried', 'tofu': 'soy_fried',
                 'soy_dried': 'soy_fried', 'egg': 'egg_fried', 'starch_noodles': 'mixed_veg_fried'}


def read_curation(filename, **kwargs):
    """Read curated rules, decoding escaped source-language matching keys."""
    table = pd.read_csv(CURATION / filename, **kwargs)
    for column in table.columns:
        if pd.api.types.is_string_dtype(table[column]):
            table[column] = table[column].map(
                lambda value: re.sub(r'\\u([0-9a-fA-F]{4})', lambda match: chr(int(match[1], 16)), value)
                if isinstance(value, str) else value
            )
    return table


def number(text):
    text = str(text).strip()
    if text in ('', '\u2014'):
        return np.nan
    return 0.0 if text == 'Tr' else float(re.sub(r'[^\d.]', '', text))


def load_usda(fdc_ids):
    nutrients = pd.read_csv(USDA / 'food_nutrient.csv', usecols=['fdc_id', 'nutrient_id', 'amount'])
    nutrients = nutrients.loc[nutrients.fdc_id.isin(fdc_ids) & nutrients.nutrient_id.isin(USDA_NUTRIENTS)]
    table = nutrients.pivot_table(index='fdc_id', columns='nutrient_id', values='amount').rename(columns=USDA_NUTRIENTS)
    names = pd.read_csv(USDA / 'food.csv', usecols=['fdc_id', 'description']).set_index('fdc_id').description
    table['description'] = names.reindex(table.index)
    assert set(fdc_ids) <= set(table.index) and table[NUTRIENTS[:4]].notna().all().all()
    return table


def load_entries():
    rows = cached_rows()
    classes = read_curation('fiber_classes.csv', keep_default_na=False)
    ratios = {name: float(np.median([number(rows[int(i)]['fiber']) / number(rows[int(i)]['carbohydrate']) for i in ids.split(';')])) if ids else 0.0
              for name, ids in zip(classes.fiber_class, classes.proxy_entry_ids)}
    crosswalk = read_curation('usda_crosswalk.csv', dtype=str, keep_default_na=False).set_index('entry_id')
    usda = load_usda({int(value) for value in pd.concat([crosswalk.fdc_id, crosswalk.fiber_fdc_id]) if value})
    records = []
    for entry in read_curation('entries.csv', dtype=str, keep_default_na=False).itertuples(index=False):
        row = rows[int(entry.entry_id)]
        assert row['name'] == entry.name == crosswalk.at[entry.entry_id, 'cn_name'], (entry.entry_id, row['name'], entry.name)
        reported = {'edible_pct': number(row['edible']), 'water_g': number(row['water']), 'energy_kcal': number(row['energy']) / KJ_PER_KCAL,
                    'protein_g': number(row['protein']), 'fat_g': number(row['fat']), 'carbohydrate_g': number(row['carbohydrate']), 'fiber_g': number(row['fiber'])}
        used = {key: reported[key] for key in NUTRIENTS[:4]}
        for field in filter(None, entry.blank_as_zero.split(';')):
            assert np.isnan(used[f'{field}_g']), (entry.entry_id, field)
            used[f'{field}_g'] = 0.0
        ethanol, reported_cho = float(entry.ethanol_g or 0), used['carbohydrate_g']
        used['carbohydrate_g'] = float(entry.carbohydrate_g_override) if entry.carbohydrate_g_override else reported_cho - ethanol
        used['energy_kcal'] += ((used['carbohydrate_g'] - reported_cho) * CHO_KJ + ethanol * ETHANOL_KJ) / KJ_PER_KCAL
        link = crosswalk.loc[entry.entry_id]
        fdc = int(link.fdc_id) if link.fdc_id else None
        scale = (100 - reported['water_g']) / (100 - usda.at[fdc, 'water_g']) if link.scale_to_cn_water == 'yes' else 1.0
        usda_values = {key: usda.at[fdc, key] * scale for key in NUTRIENTS} if fdc else {}
        if link.fiber_fdc_id:
            usda_fiber, usda_fiber_source = usda.at[int(link.fiber_fdc_id), 'fiber_g'], int(link.fiber_fdc_id)
        else:
            usda_fiber, usda_fiber_source = usda_values.get('fiber_g', np.nan), fdc
        note = entry.note
        if np.isfinite(reported['fiber_g']):
            assert not entry.fiber_class, entry.entry_id
            used['fiber_g'], basis = reported['fiber_g'], 'reported'
        elif entry.fiber_class == 'zero':
            used['fiber_g'], basis = 0.0, 'zero'
        elif np.isfinite(usda_fiber):
            used['fiber_g'], basis = usda_fiber, f'usda:{usda_fiber_source}'
        else:
            used['fiber_g'], basis = ratios[entry.fiber_class] * used['carbohydrate_g'], f'imputed:{entry.fiber_class}'
        if used['fiber_g'] > used['carbohydrate_g']:
            used['fiber_g'] = used['carbohydrate_g']
            note = (note + '; ' if note else '') + 'Fiber exceeds carbohydrate; cap fiber at the carbohydrate value.'
        assert all(np.isfinite(value) for value in used.values()), (entry.entry_id, used)
        if fdc:
            usda_values['fiber_g'] = usda_fiber if np.isfinite(usda_fiber) else 0.0 if entry.fiber_class == 'zero' else used['fiber_g']
            if usda_values['fiber_g'] > usda_values['carbohydrate_g']:
                usda_values['fiber_g'] = usda_values['carbohydrate_g']
                note = (note + '; ' if note else '') + 'USDA-variant fiber exceeds carbohydrate; cap fiber at the carbohydrate value.'
        else:
            usda_values = dict(used)
        cache = CACHE / f'search_ALL_p{row["page"]:03d}.json'
        assert cache.exists()
        records.append({'entry_id': entry.entry_id, 'name': row['name'], 'source': 'China CDC NIN food composition platform', **reported,
                        'query_term': '(blank: full listing)', 'query_page': row['page'], 'retrieved_utc': row['retrieved_utc'], 'cache_file': cache.name,
                        **{f'{key}_used': value for key, value in used.items()}, 'fiber_basis': basis, 'ethanol_g_removed': ethanol or np.nan, 'adjustment_note': note,
                        'usda_fdc_id': fdc, 'usda_description': usda.at[fdc, 'description'] if fdc else '', 'usda_scale': scale if fdc else np.nan,
                        'usda_basis': 'usda' if fdc else 'cn_retained', 'usda_confidence': link.confidence,
                        **{f'{key}_usda': value for key, value in usda_values.items()}})
    for entry_id, link in crosswalk.loc[crosswalk.index.str.startswith('U')].iterrows():
        values = {key: float(np.nan_to_num(usda.at[int(link.fdc_id), key])) for key in NUTRIENTS}
        records.append({'entry_id': entry_id, 'name': link.cn_name, 'source': 'USDA SR Legacy (no China CDC entry)', 'water_g': usda.at[int(link.fdc_id), 'water_g'],
                        **values, **{f'{key}_used': value for key, value in values.items()}, 'fiber_basis': f'usda:{link.fdc_id}', 'adjustment_note': link.rationale,
                        'usda_fdc_id': int(link.fdc_id), 'usda_description': usda.at[int(link.fdc_id), 'description'], 'usda_scale': 1.0, 'usda_basis': 'usda',
                        'usda_confidence': link.confidence, **{f'{key}_usda': value for key, value in values.items()}})
    water = crosswalk.loc[WATER]
    records.append({'entry_id': WATER, 'name': 'Water added during preparation or cooking; not a source-table entry', 'source': 'pseudo-entry', 'edible_pct': 100.0, 'water_g': 100.0,
                    **{key: 0.0 for key in NUTRIENTS}, **{f'{key}_used': 0.0 for key in NUTRIENTS}, 'fiber_basis': 'zero',
                    'adjustment_note': 'Water added to cooked or reconstituted recipes other than rice; all nutrients are zero.', 'usda_fdc_id': int(water.fdc_id),
                    'usda_description': usda.at[int(water.fdc_id), 'description'], 'usda_scale': 1.0, 'usda_basis': 'usda', 'usda_confidence': water.confidence,
                    **{f'{key}_usda': 0.0 for key in NUTRIENTS}})
    entries = pd.DataFrame(records).set_index('entry_id')
    entries['usda_fdc_id'] = entries.usda_fdc_id.astype('Int64')
    return entries, ratios


def expand(recipe, recipes):
    shares = {}
    for part in filter(None, recipe.split(';')):
        item, share = part.split(':')
        item, share = (OIL if item == 'OIL' else item), float(share)
        for sub, weight in (expand(recipes[item[1:]], recipes).items() if item.startswith('@') else [(item, 1.0)]):
            shares[sub] = shares.get(sub, 0.0) + share * weight
    assert abs(sum(shares.values()) - 1) < 1e-9, (recipe, sum(shares.values()))
    return shares


def profile(shares, entries):
    weights = pd.Series(shares)
    table = entries.loc[weights.index]
    values = {key: float(weights @ table[f'{key}_used']) for key in NUTRIENTS}
    fiber = weights * table.fiber_g_used
    kind = table.fiber_basis.str.split(':').str[0]
    for basis in ('reported', 'usda', 'imputed'):
        values[f'fiber_{basis}_share'] = float(fiber[kind == basis].sum() / fiber.sum()) if fiber.sum() > 0 else 0.0
    values['added_oil_g_per_100g'] = 100 * shares.get(OIL, 0.0)
    values.update({f'{key}_usda': float(weights @ table[f'{key}_usda']) for key in NUTRIENTS})
    values['usda_matched_share'] = float(weights[table.usda_basis == 'usda'].sum())
    return values


def describe(shares, entries):
    ordered = sorted(shares.items(), key=lambda item: -item[1])
    return ';'.join(f'{entry}:{share:.4g}' for entry, share in ordered), '; '.join(f'{entries.at[entry, "name"]} {100 * share:.1f}%' for entry, share in ordered)


def method(name):
    if re.search(OIL_TERMS, re.sub(OIL_EXEMPT, '', name)):
        return 'fried_named'
    return 'nonfried_named' if re.search(NONOIL_TERMS, name.replace('\u70e4\u9eb8', '')) else 'unspecified'


def components(name):
    text = re.sub(AROMATIC, '', name)
    found = {}
    for kind, pattern in {**PROTEIN, **VEG}.items():
        hits = re.findall(pattern, name if kind == 'generic' else text)
        if hits:
            found[kind] = hits
            text = re.sub(pattern, '', text)
    return found


def classify(name):
    if not name:
        return 'unmappable', 'unspecified', 'low', 'No source-language name; not classified automatically.'
    cook = method(name)
    for group, pattern in DIRECT:
        if re.search(pattern, name):
            return group, cook, 'medium', ''
    if re.search('\u86cb', name) and re.search('\u84b8|\u7096|\u7fb9', name):
        return 'egg_custard', 'nonfried_named', 'medium', 'Steamed egg custard.'
    if re.search(SOUP, name):
        if re.search('\u9762', name) and not re.search('\u9762\u7b4b', name):
            return 'noodle_meal', 'nonfried_named', 'medium', 'Noodle soup.'
        if re.search(SWEET_SOUP, name):
            return 'sweet_soup', 'nonfried_named', 'medium', 'Assume no added sugar.'
        if re.search(STARCHY_DISH, name):
            return 'starchy_soup', 'nonfried_named', 'medium', 'Starchy soup.'
        found = components(name)
        meat = [kind for kind in PROTEIN if kind not in ('egg', 'soy') and kind in found]
        solids = [kind for kind in found if kind not in meat]
        if meat:
            return ('meat_veg_soup' if solids else 'broth'), 'nonfried_named', 'low', 'Soup with unknown solid-food fraction.'
        return 'veg_soup', 'nonfried_named', 'medium', ''
    found = components(name)
    others = [kind for kind in found if kind not in ('generic',)]
    for group, pattern in STAPLE:
        if not re.search(pattern, name):
            continue
        if group == 'filled':
            group = 'filled_veg' if re.search(FILLED_VEG, name) and not re.search('\u8089|\u867e|\u87f9|\u725b|\u9e21(?!\u86cb)', name) else 'filled_meat'
        elif group == 'flatbread':
            group = 'fried_wheat' if cook == 'fried_named' else 'bread_flatbread'
        elif group == 'rice_cake_dish':
            group = 'fried_rice_noodles' if len(others) else 'glutinous_ricecake'
        elif group == 'noodles' and len(others):
            group = 'noodle_meal'
        elif group == 'meatball' and cook == 'fried_named':
            group = 'meatball_fried'
        return group, ('fried_in_entry' if group in IN_ENTRY_FRIED else cook), 'medium', ''
    n_parts = sum(1 if kind in PROTEIN else len(set(hits)) for kind, hits in found.items())
    non_meat = any(kind in found for kind in ('egg', 'soy', *VEG))
    oil = cook == 'fried_named' or (n_parts >= 2 and cook == 'unspecified' and non_meat and not re.search(LIST, name))
    cooking = 'fried_named' if cook == 'fried_named' else 'fried_dish_type' if oil else cook
    plain = 'nonfried_named' if cook == 'nonfried_named' else 'unspecified'
    veg = [kind for kind in VEG if kind in found]
    meat = [kind for kind in ('processed', 'meatball', 'offal', 'beef', 'mutton', 'chicken', 'duck', 'pork') if kind in found]
    seafood = [kind for kind in ('fish_processed', 'crab', 'shrimp', 'shellfish', 'fish') if kind in found]
    if re.search(STARCHY_DISH, name) and n_parts >= 2 and 'egg' not in found and 'meatball' not in found:
        return ('starchy_mixed_dish', cooking, 'medium', 'Starchy mixed meat-and-vegetable dish.') if oil else ('starchy_mixed_steamed', plain, 'medium', 'Starchy mixed meat-and-vegetable dish.')
    if 'egg' in found:
        if re.search('\u84b8|\u7096|\u7fb9', name):
            return 'egg_custard', 'nonfried_named', 'medium', 'Steamed egg custard.'
        if veg or 'soy' in found:
            return ('egg_veg_stirfry', cooking, 'medium', '') if oil else ('egg', plain, 'medium', '')
        return ('egg_fried', cooking, 'medium', '') if oil else ('egg', plain, 'medium', '')
    if meat and not ('meatball' in meat and len(meat) == 1 and (veg or 'soy' in found)):
        if veg or 'soy' in found or seafood:
            return ('meat_veg_stirfry', cooking, 'medium', '') if oil else ('meat_veg_steamed', plain, 'medium', '')
        kind = meat[0]
        if kind in ('processed', 'offal'):
            if kind == 'offal':
                return ('offal_fried', cooking, 'medium', '') if oil else ('offal', plain, 'medium', '')
            return ('pork_fried', cooking, 'low', 'Map fried processed meat to the stir-fried pork group.') if oil else ('pork_processed', plain, 'medium', '')
        if kind == 'pork':
            if re.search('\u6392|\u8e44|\u8098|\u9ac8|\u811a|\u722a', name):
                base = 'pork_rib'
            elif oil and re.search('\u7ea2\u70e7\u8089|\u4e94\u82b1|\u70e7\u8089|\u6263\u8089|\u56de\u9505|\u4e1c\u5761', name):
                return 'pork_belly_braised', cooking, 'medium', ''
            else:
                base = 'pork'
        else:
            base = {'meatball': 'meatball', 'beef': 'beef_mutton', 'mutton': 'beef_mutton', 'chicken': 'chicken', 'duck': 'duck_goose'}[kind]
        return (FRIED_VARIANT[base], cooking, 'medium', '') if oil else (base, plain, 'medium', '')
    if 'meatball' in meat:
        return ('meatball_fried', cooking, 'medium', '') if cook == 'fried_named' else ('meatball', plain, 'medium', 'Count vegetables in the filling within the meatball entry.')
    if 'soy' in found:
        if seafood:
            return ('seafood_fried', cooking, 'medium', '') if oil else ('tofu', plain, 'medium', '')
        if veg:
            return ('soy_veg_stirfry', cooking, 'medium', '') if oil else ('tofu' if re.search('\u8c46\u8150|\u8c46\u82b1', name) else 'soy_dried', plain, 'medium', '')
        base = 'tofu' if re.search('\u8c46\u8150(?!\u5e72|\u8863|\u76ae)|\u8c46\u82b1', name) and not re.search('\u6cb9\u8c46\u8150', name) else 'soy_dried'
        if re.search('\u6cb9\u8c46\u8150|\u6cb9\u9762\u7b4b', name):
            return 'soy_fried', 'fried_in_entry', 'medium', 'Fried soy products.'
        return (FRIED_VARIANT[base], cooking, 'medium', '') if oil else (base, plain, 'medium', '')
    if seafood:
        kind = seafood[0]
        base = {'fish_processed': 'fish_processed', 'crab': 'crab', 'shrimp': 'shrimp', 'shellfish': 'shellfish', 'fish': 'fish'}[kind]
        if base in ('fish_processed', 'crab'):
            return ('seafood_fried', cooking, 'low', 'Map fried crab or fish balls to the stir-fried shrimp-and-shellfish group.') if oil else (base, plain, 'medium', '')
        return (FRIED_VARIANT[base], cooking, 'medium', '') if oil else (base, plain, 'medium', '')
    if veg:
        if veg == ['pickle']:
            return 'pickled_veg', 'na', 'medium', ''
        groups = {VEG_GROUP.get(kind, 'mixed_veg') for kind in veg if kind != 'pickle'} or {'mixed_veg'}
        base = groups.pop() if len(groups) == 1 else 'mixed_veg'
        if base == 'starch_noodles' and not oil:
            return 'starch_noodles', 'na', 'medium', ''
        confidence = 'low' if 'generic' in veg else 'medium'
        return (FRIED_VARIANT[base], cooking, confidence, '') if oil else (base, plain, confidence, '')
    if re.search(AROMATIC, name):
        return 'root_stem_veg', plain, 'medium', 'Scallion, ginger and garlic vegetables.'
    return 'unmappable', cook, 'low', 'Not classified automatically.'


def key_mappings(frequency, specific, overrides, recipes, group_shares, keys_with_amount, entries):
    rows, info = [], {}
    for rank, item in enumerate(frequency.itertuples(index=False), start=1):
        rule_group, rule_cooking, rule_confidence, rule_note = classify(item.item_zh)
        recipe = ''
        if rank <= SPECIFIC_TOP:
            choice = specific.loc[item.key]
            assert int(choice['rank']) == rank
            group, cooking, confidence, notes, recipe, assignment = choice.group_id, choice.cooking, choice.confidence, choice.notes, choice.recipe, 'specific'
        elif item.key in overrides.index:
            choice = overrides.loc[item.key]
            group, recipe, confidence, notes, assignment = choice.group_id, choice.recipe, choice.confidence, choice.notes, 'override'
            cooking = choice.cooking or rule_cooking
        else:
            group, cooking, confidence, notes, assignment = rule_group, rule_cooking, rule_confidence, rule_note, 'rule'
        if item.key not in keys_with_amount:
            group, recipe, confidence, notes, assignment = 'unmappable', '', 'low', 'No parseable recorded weight; ' + notes, 'no_weight'
        assert group in recipes, (item.key, group)
        oil_rule = ''
        if cooking == 'unspecified' and group != 'unmappable':
            if (group in O1_GROUPS or (group == 'starchy_veg' and '\u571f\u8c46\u4e1d' in item.item_zh)) and not re.search(O1_EXCLUDE, item.item_zh):
                oil_rule = 'O1'
            elif re.search(O2_ITEMS, item.item_zh) and group in ('fish', 'pork_rib', 'beef_mutton'):
                oil_rule = 'O2'
        if oil_rule:
            group, cooking = FRIED_VARIANT[group], 'fried_default'
            if recipe:
                recipe = ';'.join(f'{entry}:{(1 - OIL_SHARE) * share:.6g}' for entry, share in expand(recipe, recipes).items()) + f';OIL:{OIL_SHARE}'
        if not recipe and group in group_shares:
            cooking = 'fried_in_entry' if group in IN_ENTRY_FRIED else cooking
            has_oil = group_shares[group].get(OIL, 0) > 0
            cooking = 'fried_dish_type' if has_oil and cooking not in FRIED_COOKING else cooking
            assert has_oil or cooking not in FRIED_COOKING, (item.key, group, cooking)
        if cooking == 'unspecified' and group in NOT_A_DISH:
            cooking = 'na'
        shares = expand(recipe, recipes) if recipe else group_shares.get(group)
        level = 'unmappable' if group == 'unmappable' else 'specific' if recipe else 'group'
        if level == 'specific' and cooking != 'oil_in_recipe':
            assert np.isclose(100 * shares.get(OIL, 0.0), 100 * OIL_SHARE * (cooking in FRIED_COOKING)), (item.key, cooking)
        oil_rule = oil_rule or ('O4' if re.search('(?<!\u6d0b)\u8471\u70e4', item.item_zh) else OIL_BASIS.get(cooking, 'none'))
        values = profile(shares, entries) if shares else {key: np.nan for key in NUTRIENT_COLUMNS}
        recipe_text, recipe_names = describe(shares, entries) if level == 'specific' else ('', '')
        info[item.key] = {'shares': shares, 'group': group, 'level': level}
        rows.append({'key': item.key, 'item_zh': item.item_zh, 'lines': item.lines, 'rank': rank, 'review_priority': int(rank <= REVIEW_TOP),
                     'mapping_level': level, 'cn_entry_or_recipe': recipe_text, 'recipe_names': recipe_names, 'group_id': group, 'assignment': assignment,
                     **values, 'cooking': cooking, 'oil_rule': oil_rule, 'confidence': confidence, 'notes': notes, 'rule_group': rule_group})
    return pd.DataFrame(rows), info


def line_amounts(recorded):
    overrides = read_curation('amount_overrides.csv', dtype={'amount_g': float}, keep_default_na=False, na_values={'amount_g': ['']})
    lines = recorded.merge(overrides[['record_id', 'time', 'line', 'amount_g', 'key_override', 'amount_source', 'reason']], on=['record_id', 'time', 'line'],
                           how='left', validate='one_to_one', indicator=True)
    assert (lines._merge == 'both').sum() == len(overrides)
    overridden = lines._merge == 'both'
    lines['key'] = np.where(overridden & lines.key_override.fillna('').ne(''), lines.key_override, lines.key)
    lines['amount_g_raw_parsed'] = lines.amount
    lines['amount_g_recorded'] = np.where(overridden, lines.amount_g, lines.amount)
    lines['amount_source'] = np.where(overridden, lines.amount_source.replace({'judgement': 'override'}), np.where(lines.amount.notna(), 'en', 'none'))
    lines['override_reason'] = lines.reason.fillna('')
    shares = lines.loc[lines.amount_g_recorded.notna()].groupby('record_id').amount_g_recorded.agg(lambda values: (values % 5 != 0).mean())
    parsed = lines.loc[lines.amount.notna()].groupby('record_id').amount.agg(lambda values: (values % 5 != 0).mean())
    lines['record_self_reported'] = lines.record_id.map(shares < 0.5).fillna(False).astype(bool)
    flips = sorted(set(shares.index[shares < 0.5]) ^ set(parsed.index[parsed < 0.5]))
    return lines.drop(columns=['_merge', 'amount_g', 'key_override', 'reason']), shares, flips


def line_nutrients(lines, info, recipes, group_shares, entries):
    routes = read_curation('line_routes.csv', keep_default_na=False).set_index(['key', 'item_zh'])
    states = {name: expand(recipe, recipes) for name, recipe in STATE_RECIPES.items()}
    cache = {}

    def values(shares):
        signature = tuple(sorted(shares.items()))
        if signature not in cache:
            cache[signature] = profile(shares, entries)
        return cache[signature]

    records = []
    for row in lines.itertuples(index=False):
        zh = row.item_zh if isinstance(row.item_zh, str) else ''
        key = row.key
        route = routes.loc[(key, zh)] if (key, zh) in routes.index else None
        if route is not None and route.route_key:
            key, route = route.route_key, None
        mapped = info[key]
        shares, group, level = mapped['shares'], mapped['group'], mapped['level']
        if route is not None:
            shares, group, level = expand(route.route_recipe, recipes), route.route_group, 'specific'
        amount = row.amount_g_recorded
        state, used = 'as_mapped', amount
        if level == 'unmappable':
            state = 'unmappable'
        elif row.record_self_reported and row.key in RICE_KEYS | COARSE_KEYS and key == row.key and amount % 5 == 0 and amount <= RAW_STAPLE_MAX:
            state, shares = 'raw_staple', states['raw_rice' if row.key in RICE_KEYS else 'raw_grain']
        elif group == 'noodles' and (re.search('\u5377\u9762|\u6302\u9762', zh) or amount <= 60):
            state, shares = 'dry_noodle', states['dry_noodle']
        elif group in ('noodles', 'noodle_meal') and amount >= 300:
            state, shares = 'noodle_soup', states['noodle_soup']
        elif group in ('filled_meat', 'filled_veg') and re.search('\u9984\u9968|\u4e91\u541e', zh) and amount >= (200 if '\u5c0f\u9984\u9968' in zh else 250):
            state, shares = 'wonton_broth', states['wonton_broth']
        elif group == 'dry_cereal_powder' and re.search('\u9ea6\u7247|\u71d5\u9ea6', zh) and (row.unit == 'ml' or amount >= 150):
            state, shares = 'prepared_cereal', states['prepared_cereal']
        elif re.fullmatch('(\u716e)?\u7389\u7c73(\u68d2)?|\u4e3b\u98df\u7389\u7c73', zh) and amount >= CORN_MIN:
            state, used = 'corn_on_cob', amount * CORN_EDIBLE
        reason = ('unmappable_item' if level == 'unmappable' else 'weight_unreliable' if row.amount_source == 'override' and np.isnan(amount)
                  else 'no_weight' if np.isnan(amount) else 'over_800g' if amount > MAX_LINE_G else '')
        record = {'record_id': row.record_id, 'time': row.time, 'line': row.line, 'key': key, 'item_zh': zh, 'text_zh': row.text_zh, 'unit': row.unit,
                  'amount_g_raw_parsed': row.amount_g_raw_parsed, 'amount_g_recorded': amount, 'amount_g_used': used if not reason else np.nan,
                  'amount_source': row.amount_source, 'override_reason': row.override_reason, 'usable': not reason, 'unusable_reason': reason,
                  'record_self_reported': row.record_self_reported, 'weight_state': state, 'mapping_level': level, 'group_id': group}
        if not reason:
            cooked = mapped['shares'] if state in ('raw_staple', 'dry_noodle') else shares
            plans = {'primary': (shares, ''), 'group_only': (shares if state in STATE_RECIPES_BY_STATE else group_shares[group], ''),
                     'usda': (shares, '_usda'), 'staples_cooked': (cooked, '')}
            for variant, (plan, suffix) in plans.items():
                profile_values = values(plan)
                for nutrient in NUTRIENTS:
                    record[f'{variant}_{nutrient}'] = used * profile_values[f'{nutrient}{suffix}'] / 100
            record['added_oil_g'] = used * shares.get(OIL, 0.0)
            for nutrient in NUTRIENTS:
                oil_part = record['added_oil_g'] * entries.at[OIL, f'{nutrient}_used'] / 100 if nutrient in ('energy_kcal', 'fat_g') else 0.0
                record[f'no_oil_{nutrient}'] = record[f'primary_{nutrient}'] - oil_part
        records.append(record)
    return pd.DataFrame(records), states


def summaries(lines):
    meals = lines.groupby(['record_id', 'time'])
    meal = meals[[f'{variant}_{nutrient}' for variant in VARIANTS for nutrient in NUTRIENTS]].sum().join(meals.usable.all()).join(meals.record_self_reported.first())
    usable = meal.loc[meal.usable]
    day = usable.reset_index().assign(date=lambda frame: frame.time.str[:10])
    days = day.groupby(['record_id', 'date']).agg(meals=('usable', 'size'), self_reported=('record_self_reported', 'first'),
                                                   **{f'{variant}_energy_kcal': (f'{variant}_energy_kcal', 'sum') for variant in VARIANTS},
                                                   **{f'{variant}_carbohydrate_g': (f'{variant}_carbohydrate_g', 'sum') for variant in VARIANTS})
    excluded_days = meal.loc[~meal.usable].reset_index().assign(date=lambda frame: frame.time.str[:10])[['record_id', 'date']].drop_duplicates()
    days = days.drop(index=pd.MultiIndex.from_frame(excluded_days), errors='ignore')
    full = days.loc[days.meals >= 3]
    return meal, usable, full


def main():
    entries, ratios = load_entries()
    groups = read_curation('groups.csv', keep_default_na=False)
    recipes = dict(zip(groups.group_id, groups.recipe))
    group_shares = {group: expand(recipe, recipes) for group, recipe in recipes.items() if recipe}
    group_profiles = {group: profile(shares, entries) for group, shares in group_shares.items()}
    assert all(np.isclose(values['added_oil_g_per_100g'], [0, 100 * OIL_SHARE]).any() for values in group_profiles.values())
    specific = read_curation('specific.csv', keep_default_na=False).set_index('key')
    overrides = read_curation('group_overrides.csv', keep_default_na=False).set_index('key')
    frequency = pd.read_csv(INTERIM / 'food_item_frequency.csv', keep_default_na=False)
    items = pd.read_csv(INTERIM / 'intake_items.csv')
    recorded = items.loc[~items.not_recorded].assign(key=lambda frame: frame.item_en.fillna(frame.text_en).str.lower().str.strip())
    assert set(specific.index) <= set(frequency.key) and set(overrides.index) <= set(frequency.key) and not set(specific.index) & set(overrides.index)
    lines, weighing, flips = line_amounts(recorded)
    keys_with_amount = set(lines.loc[lines.amount_g_recorded.notna(), 'key']) | set(recorded.loc[recorded.amount.notna(), 'key'])
    mapping, info = key_mappings(frequency, specific, overrides, recipes, group_shares, keys_with_amount, entries)
    table, states = line_nutrients(lines, info, recipes, group_shares, entries)
    variant_columns = [f'{variant}_{nutrient}' for variant in VARIANTS for nutrient in NUTRIENTS]
    columns = ['record_id', 'time', 'line', 'key', 'item_zh', 'text_zh', 'unit', 'amount_g_raw_parsed', 'amount_g_recorded', 'amount_g_used', 'amount_source',
               'override_reason', 'usable', 'unusable_reason', 'record_self_reported', 'weight_state', 'mapping_level', 'group_id', 'added_oil_g', *variant_columns]
    table[columns[18:]] = table[columns[18:]].astype(float).round(3)
    table[columns].to_csv(INTERIM / 'intake_line_nutrients.csv', index=False)

    variants = recorded.groupby('key').item_zh.agg(lambda values: '|'.join(f'{name}:{count}' for name, count in values.value_counts().items()))
    state_counts = table.groupby('key').weight_state.agg(lambda values: '; '.join(f'{state}:{count}' for state, count in values.value_counts().items()))
    mapping.insert(2, 'zh_variants', mapping.key.map(variants).fillna(''))
    mapping['line_states'] = mapping.key.map(state_counts).fillna('')
    mapping['lines_usable'] = mapping.key.map(table.groupby('key').usable.sum()).fillna(0).astype(int)
    mapping[NUTRIENT_COLUMNS] = mapping[NUTRIENT_COLUMNS].round(3)
    keep = ['key', 'item_zh', 'zh_variants', 'lines', 'lines_usable', 'rank', 'review_priority', 'mapping_level', 'cn_entry_or_recipe', 'recipe_names', 'group_id',
            'assignment', *NUTRIENT_COLUMNS, 'cooking', 'oil_rule', 'line_states', 'confidence', 'notes', 'rule_group']
    mapping[keep].to_csv(METADATA / 'shanghai_t2dm_food_mapping.csv', index=False)

    lines_by_group = table.groupby('group_id').size()
    group_lines = table.loc[table.mapping_level == 'group'].groupby('group_id').size()
    rows = []
    for group in groups.itertuples(index=False):
        shares = group_shares.get(group.group_id)
        recipe_text, recipe_names = describe(shares, entries) if shares else ('', '')
        values = group_profiles.get(group.group_id, {key: np.nan for key in NUTRIENT_COLUMNS})
        rows.append({'group_id': group.group_id, 'group_zh': group.group_zh, 'group_en': group.group_en, 'recipe': recipe_text, 'recipe_names': recipe_names,
                     **{key: round(values[key], 3) for key in NUTRIENT_COLUMNS}, 'items': int((mapping.group_id == group.group_id).sum()),
                     'lines_all_levels': int(lines_by_group.get(group.group_id, 0)), 'lines_group_level': int(group_lines.get(group.group_id, 0)),
                     'curated_recipe': group.recipe, 'rationale': group.rationale})
    pd.DataFrame(rows).to_csv(METADATA / 'shanghai_t2dm_food_groups.csv', index=False)

    routes = read_curation('line_routes.csv', keep_default_na=False)
    used = ({entry for shares in group_shares.values() for entry in shares} | {entry for mapped in info.values() if mapped['shares'] for entry in mapped['shares']}
            | {entry for shares in states.values() for entry in shares} | {entry for recipe in routes.route_recipe if recipe for entry in expand(recipe, recipes)})
    assert used == set(entries.index), set(entries.index) ^ used
    output = entries.loc[sorted(used, key=lambda entry: (entry == WATER, entry.startswith('U'), int(entry) if entry.isdigit() else 0))].reset_index()
    rounded = [column for column in output.columns if column.endswith(('_g', '_kcal', '_used', '_usda', 'usda_scale'))]
    output[rounded] = output[rounded].astype(float).round(3)
    output['query_page'] = output.query_page.astype('Int64')
    output.to_csv(METADATA / 'shanghai_t2dm_cn_food_entries.csv', index=False)

    meal, usable, full = summaries(table)
    print('Items by level:', mapping.mapping_level.value_counts().to_dict(), '| assignment:', mapping.assignment.value_counts().to_dict(), '| groups:', len(groups))
    print('Records:', {'self_reported': int((weighing < 0.5).sum()), 'hospital_weighed': int((weighing >= 0.5).sum())}, '| flips vs parsed amounts:', flips)
    print('Lines:', len(table), '| usable:', int(table.usable.sum()), '| unusable:', table.unusable_reason.value_counts().to_dict())
    print('Line levels:', table.mapping_level.value_counts().to_dict())
    print('Weight states:', table.weight_state.value_counts().to_dict())
    print('Over 800 g:', table.loc[table.unusable_reason == 'over_800g', ['record_id', 'time', 'item_zh', 'amount_g_recorded']].values.tolist())
    print('Meals:', len(meal), '| usable:', len(usable), '| excluded:', int((~meal.usable).sum()))
    for variant in ('primary', 'staples_cooked'):
        print(f'{variant}: median meal energy {usable[f"{variant}_energy_kcal"].median():.0f} kcal, carbohydrate {usable[f"{variant}_carbohydrate_g"].median():.1f} g;',
              'median full-day energy (>=3 usable meals):', full.groupby('self_reported')[f'{variant}_energy_kcal'].median().round(0).to_dict(),
              'carbohydrate:', full.groupby('self_reported')[f'{variant}_carbohydrate_g'].median().round(0).to_dict(), '| days:', full.self_reported.value_counts().to_dict())
    total = table.loc[table.usable, variant_columns].sum()
    print('Total intake vs primary:', {variant: {nutrient: round(total[f'{variant}_{nutrient}'] / total[f'primary_{nutrient}'], 3) for nutrient in NUTRIENTS} for variant in VARIANTS[1:]})
    print('Added oil per usable meal (g):', round(table.loc[table.usable, 'added_oil_g'].sum() / len(usable), 2), '| by oil rule:',
          table.loc[table.usable].assign(rule=lambda frame: frame.key.map(mapping.set_index('key').oil_rule)).groupby('rule').added_oil_g.sum().round(0).to_dict())
    print('Fiber basis share of eaten food grams and fiber grams:', fiber_shares(table, info, recipes, entries, states))
    print('Cooked rice (287) fiber g/100 g:', round(entries.at['287', 'fiber_g_used'], 3), entries.at['287', 'fiber_basis'])


def fiber_shares(table, info, recipes, entries, states):
    weights = {}
    routes = read_curation('line_routes.csv', keep_default_na=False).set_index(['key', 'item_zh'])
    for row in table.loc[table.usable].itertuples(index=False):
        shares = STATE_SHARES(row, info, recipes, routes, states)
        for entry, share in shares.items():
            weights[entry] = weights.get(entry, 0.0) + row.amount_g_used * share
    frame = pd.Series(weights).to_frame('grams').join(entries[['fiber_basis', 'fiber_g_used']])
    frame['kind'] = np.where(frame.index == WATER, 'added_water', frame.fiber_basis.str.split(':').str[0])
    frame = frame.loc[frame.kind != 'added_water']
    frame['fiber'] = frame.grams * frame.fiber_g_used / 100
    return {'grams': (frame.groupby('kind').grams.sum() / frame.grams.sum()).round(4).to_dict(), 'fiber': (frame.groupby('kind').fiber.sum() / frame.fiber.sum()).round(4).to_dict()}


def STATE_SHARES(row, info, recipes, routes, states):
    if row.weight_state in STATE_RECIPES_BY_STATE:
        return states[STATE_RECIPES_BY_STATE[row.weight_state](row)]
    route = routes.loc[(row.key, row.item_zh)] if (row.key, row.item_zh) in routes.index else None
    return expand(route.route_recipe, recipes) if route is not None and route.route_recipe else info[row.key]['shares']


if __name__ == '__main__':
    main()
