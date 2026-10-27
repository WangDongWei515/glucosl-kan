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
O1_EXCLUDE = r'黄瓜|青瓜|番茄|西红柿|圣女果|南瓜|^姜|^葱$|姜片|蒜蓉|紫菜|银耳|白木耳|魔芋|红豆|赤豆|绿豆(?!芽)|黄豆(?!芽)|鹰嘴豆|脱水|水果萝卜'
O2_ITEMS = r'带鱼|大排|小黄鱼|牛排'
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
OIL_TERMS = r'炒|煎|炸|爆|煸|溜|熘|烧|油|(?<!洋)葱烤'
OIL_EXEMPT = r'油麦菜|酱油|蚝油|耗油|奶油|黄油|牛油果|麻油|油菜|油豆腐|油面筋|油条|油饼|油墩子|油浸|烧麦|烧卖|烧饼|烧鹅|叉烧'
NONOIL_TERMS = r'蒸|煮|白灼|白切|白斩|汤|羹|拌|凉|炖|卤|盐水|醉|糟|白水|焯|色拉|沙拉|涮|烫|烤|(?<![京甜炒])酱(?=[牛鸭排猪羊鸡香])'
AROMATIC = r'小葱|香葱|(?<![大京洋])葱(?:花|油|烤|爆|香)?(?!头)|姜|蒜蓉|蒜泥|蒜香|香菜|芫荽|剁椒|花椒|椒盐|黑椒|黑胡椒|胡椒|咖喱'
# Food components used to recognise composite dishes; matched in this order on the Chinese name, each match removed before the next kind.
PROTEIN = {
    'egg': r'(?<!无)(?:鸡|鸭|鹅|鹌鹑|咸|皮|松花|茶叶|卤|荷包|水煮|白煮)?蛋(?!糕|挞|黄酥|酥)(?:白|黄|清|皮)?',
    'soy': r'豆腐(?:干|衣|皮|结)?|豆干|香干|干丝|百叶(?:结|丝)?|千张|素鸡|腐竹|豆皮|腐皮|烤麸|油?面筋|豆花|纳豆|马兰豆',
    'processed': r'香肠|火腿(?:肠|片)?|午餐肉|咸肉|腊肉|培根|红肠|肉肠|肉松|腊肠|咸蹄|熏肉|(?:牛)?肉干',
    'meatball': r'(?:鸡|牛)?肉(?:圆|丸)(?:子)?|(?<!蔬菜)丸子|狮子头|(?:牛)?肉饼|贡丸|肉卷|(?<!酒酿)圆子',
    'offal': r'(?:猪|鸭|鸡|鹅|牛|羊)?(?:肝|肚(?!裆)|大肠|肥肠|胗|肫|肺)|(?:鸭|猪)血|血旺|猪耳|牛筋|蹄筋|牛舌|腰花|鱼肠|鸭头|鸭脖|鸭肠',
    'shellfish': r'鲍鱼|鱿鱼|墨鱼仔?|乌贼|章鱼|目鱼|鱼翅|佛跳墙|海鲜(?!菇)|蛤蜊肉?|花蛤|文蛤|蛤|蛏子|蛏|扇贝|贝类|贝|田螺|香螺|螺|河蚌|蚌|生蚝|蚝|牡蛎|(?<!杏)鲍|海参|鱿|海蜇(?:皮|头|丝)?|花甲|三鲜',
    'fish_processed': r'鱼丸|鱼圆|虾丸|虾滑|鱼排|蟹足棒|鱼片干|鱼豆腐',
    'crab': r'(?:大闸|梭子|毛|河|海|膏|皇帝|红毛)?蟹(?!味菇)(?:肉|粉|黄)?',
    'shrimp': r'(?:基围|围基|河|海|草|白米|甜|大|小龙)?虾(?:仁|肉|皮|米)?',
    'fish': r'(?:鲫|鲈|鲳|鳊|鳝|鳗|鲑|鳕|鲢|鳙|鲤|三文|巴沙|泥鳅|昂刺|鳜)?鱼(?!香)(?:肉|片|块|头|尾|丝)?|鲫|鲈|鲳|鳊|鳝|鳗|鲑|鳕|鲢|鳙|鲤|三文|巴沙|泥鳅|昂刺|鳜',
    'beef': r'牛(?!奶|油|心菜|肝菌|蛙|轧)(?:肉|腩|腱|排|尾)?(?:丝|片|粒|末|块)?',
    'mutton': r'羊(?!角|奶)(?:肉|排|蝎子|骨棒?)?(?:丝|片|块)?',
    'chicken': r'(?<!素)(?:鸡(?!蛋|毛菜|腿菇|精|枞)|鸽|鹌鹑(?!蛋)|牛蛙|雪蛤|凤爪)(?:子)?(?:胸|腿|中翅|翅|块|爪|排|柳)?(?:肉)?(?:丝|片|丁|块)?',
    'duck': r'(?:鸭(?!蛋|血|梨|胗|肫|肠)|鹅(?!蛋|肝))(?:子)?(?:腿|翅|膀|掌|爪|胸|脯|块|片|丁)?(?:肉)?',
    'pork': r'猪(?:肉|排|蹄|脚|爪)?|肉(?:丝|片|丁|末|沫|糜)?|排骨|大排|小排|排条|蹄髈|蹄膀|蹄|五花|里脊|肘|筒骨|骨头|骨',
}
VEG = {
    'pickle': r'榨菜|咸菜|酱菜|泡菜|酸菜|雪菜|梅干菜|梅菜|萝卜干|菜干|大头菜|酱萝卜|腌|辣白菜|橄榄菜|酸笋|干菜',
    'special_leafy': r'地瓜叶|红薯叶|木耳菜|豌豆苗|豆苗|鸡毛菜|油麦菜',
    'sprout': r'黄豆芽|绿豆芽|豆芽',
    'fungi': r'(?<![茨慈])菇|蘑菇?|木耳|银耳|菌菇?|松茸|海带(?:结|丝)?|紫菜|苔条|裙带菜?|海苔|猴头|竹荪',
    'starchy': r'土豆(?:丝|泥|片)?|马铃薯|红薯|山芋|地瓜|紫薯|(?<!魔)芋(?:艿|头|奶)?|山药|藕(?:片|丁)?|玉米(?:粒)?|栗子|板栗|菱角|木薯',
    'legume': r'毛豆|蚕豆|豌豆|青豆|黄豆|红豆|赤豆|绿豆|鹰嘴豆|眉豆|豆子',
    'root_stem': r'胡萝卜|白萝卜|萝卜|胡卜|莴笋|莴苣|茭白|西芹|水芹|芹菜|芹|芦笋|罗笋|冬笋|竹笋|笋|洋葱|大葱|京葱|蒜苗|蒜薹|蒜苔|蒜台|青蒜|大蒜|百合|茨菇|慈菇|荸荠|马蹄|黄花菜|黄花|金针菜|蕨菜',
    'fruit_veg': r'黄瓜|青瓜|冬瓜|丝瓜|苦瓜|南瓜|西葫芦|夜开花|葫芦|瓠|佛手瓜|番茄|西红柿|茄子|茄|青椒|红椒|彩椒|辣椒|甜椒|豇豆|缸豆|四季豆|刀豆|豆角|荷兰豆|扁豆|甜豆|豆荚|秋葵|芸豆|龙豆|圣女果',
    'leafy': r'小青菜|青菜|菠菜|波菜|生菜|小白菜|大白菜|白菜|卷心菜|卷芯菜|包菜|甘蓝|黄芽菜|黄牙菜|毛菜|菜心|塔菜|空心菜|空菜|蕹菜|茼蒿|蓬蒿|苋菜|米苋|苋|草头|荠菜|芥兰|芥蓝|菜薹|娃娃菜|韭菜|韭黄|韭芽|韭|杭白菜|牛心菜|花菜|菜花|花椰菜|西兰花|西蓝花|蔬菜叶|马兰|香椿|苦菊|油菜',
    'generic': r'蔬菜|素菜|炒菜|什锦|荤素|果蔬|水煮菜|烩菜|三丝|炒素',
    'starch_noodle': r'粉丝|粉皮|凉皮|魔芋|米线|米粉|宽粉|粉条|肠粉',
}
# Dish or food identities that decide the group before component logic; first match wins.
DIRECT = [
    ('unmappable', r'空腹'), ('water_tea', r'中药'), ('protein_powder', r'蛋白粉'),
    ('biscuit_snack', r'威化|饼干|苏打|燕麦棒|薯片|薯条|锅巴|膨化|零食|虾片|洋葱圈|兰花豆|脆片|米果|雪饼'),
    ('beer', r'啤酒'), ('wine', r'葡萄酒|红酒|白酒|参酒'),
    ('glutinous_sweet', r'酒酿|汤圆|麻薯|糍粑|青团|西米|米酒丸|丸子.*酒|酒.*丸子'),
    ('rice_wine', r'黄酒|米酒|花雕'),
    ('milk_mixed', r'奶茶|拿铁|咖啡.*奶|奶.*咖啡|奶咖|牛奶.*麦片|麦片.*牛奶|燕麦牛奶|牛奶加|牛奶.*蛋|牛奶.*面包'),
    ('sweet_drink', r'可乐|果汁|橙汁|椰子汁|椰汁|椰浆|糖水|酸梅|柠檬茶|玉米汁|饮料|苹果醋|果蔬浆|雪碧'),
    ('water_tea', r'^水$|咖啡|清咖|^茶$|绿茶|红茶|花茶|姜汁|姜汤|枫斗|煮水$|力度伸'),
    ('yogurt', r'酸奶(?!油)|发酵乳'), ('soy_milk', r'豆浆|豆奶|咸浆'), ('milk', r'牛奶|鲜奶|^奶$|舒化'),
    ('dry_cereal_powder', r'奶粉|代餐|营养粉|芝麻糊粉|薏米粉|麦片(?!粥|糊)|燕麦(?!粥|饭)'),
    ('salad', r'色拉|沙拉(?!酱)'), ('mixed_veg', r'^菜$'),
    ('meat_veg_stirfry', r'火锅|麻辣烫|冒菜|荤菜|^荤素$'),
    ('meat_veg_steamed', r'大杂烩|关东煮'),
    ('sweet_soup', r'桃胶|雪蛤'),
]
SOUP = r'汤(?!圆|包|饭)|羹|(?<!烂)糊|罗宋'
STARCHY_DISH = r'土豆|马铃薯|山药|藕|玉米|(?<!魔)芋|南瓜|粉丝|毛豆|栗子'
LIST = r'[、，,]|和|加|及|与|共'
SWEET_SOUP = r'赤豆|红豆|绿豆(?!芽)|银耳|白木耳|红枣|莲子|莲心|百合|桃胶|雪蛤|芝麻|核桃|南瓜糊|糖水|茯苓|山药羹|红薯汤'
FILLED_VEG = r'素|菜包|豆沙|豆包|菜馒头|粉丝包|萝卜丝|胡萝卜包|梅干菜|香菇菜|香菇包|韭菜|青菜'
STAPLE = [
    ('porridge', r'粥|稀饭|泡饭|汤饭'),
    ('fried_rice_noodles', r'炒饭|炒面|炒米粉|炒粉|炒年糕|炒米线|炒河粉|炒面片|炒意面'),
    ('fried_wheat', r'方便面|泡面|油条|油饼|麻花|油墩子|春卷|葱油饼|千层饼|焦圈|手抓饼|煎饼|蛋饼|酱香饼|苔条饼|萝卜丝饼|盒子|合子'),
    ('fried_dumpling', r'生煎|锅贴|煎饺|煎包|煎馄饨'),
    ('filled', r'饺|馄饨|云吞|烧麦|烧卖|小笼|包子|肉包|菜包|豆包|豆沙包|素包|蒸包|小包|粉丝包|香菇包|梅干菜包|南瓜包|肉馒头|牛肉馒头|菜馒头'),
    ('western_fastfood', r'汉堡|披萨|比萨|三明治|热狗'),
    ('bread_flatbread', r'面包|吐司|土司|芝士包'),
    ('steamed_bread', r'馒头|花卷|窝头|发糕|窝窝|杂粮包'),
    ('cake_pastry', r'蛋糕|糕点|月饼|酥(?!鸡|鸭)|蛋挞|泡芙|瑞士卷|西点|点心|山楂糕|绿豆糕|豆糕|南瓜饼|牛奶饼|凤梨|蛋卷'),
    ('meatball', r'^(牛)?肉饼$'),
    ('flatbread', r'饼'),
    ('rice_cake_dish', r'年糕(?=.)|.(?=.*年糕)'),
    ('glutinous_ricecake', r'粽|棕|年糕|米糕|糯米|粢饭|饭团|酒酿糕|黑米糕'),
    ('grain_rice', r'杂粮|糙米|黑米|红米|荞麦饭|荞麦米|荞麦红米|高粱|小米饭|藜麦|薏米.*饭|大米加小米|薏仁|五谷'),
    ('rice', r'米饭|白饭|^饭|菜饭|拌饭|寿司|主食'),
    ('noodles', r'面条|面$|拌面|汤面|面片|意面|意大利面|冷面|凉面|热干面|荞麦面|卷面|挂面|面疙瘩'),
    ('sugar_sweets', r'巧克力|糖果|奶糖|太妃|芝麻糖|麻糖|葡萄糖|白糖|冰糖|红糖|蜂蜜|果酱|蓝莓酱|草莓酱|^糖$|牛轧'),
    ('ice_cream', r'冰淇淋|雪糕|布丁|果冻|冰棍|冰激凌'),
    ('dried_fruit', r'枣|葡萄干|枸杞|桂圆干|果干|干果|杏干'),
    ('nuts_seeds', r'核桃|花生|瓜子|杏仁|腰果|松仁|松子|坚果|芝麻|榛子|开心果|南瓜子|南瓜籽|西瓜子|夏威夷|奇亚籽|亚奇籽|亚麻籽|巴旦木'),
    ('fruit_citrus', r'橙|橘|桔|柑|柚|柠檬'),
    ('fruit_melon', r'西瓜|甜瓜|哈密瓜|香瓜|白兰瓜|蜜瓜|黄金瓜'),
    ('fruit_tropical', r'香蕉|芒果|荔枝|龙眼|桂圆|榴莲|菠萝|火龙果|山竹|椰子|牛油果|百香果|木瓜|杨桃'),
    ('fruit_temperate', r'苹果|梨|(?<!核)桃(?!胶)|李子|杏(?!鲍)|樱桃|车厘子|葡萄|提子|猕猴桃|奇异果|草莓|蓝莓|(?<!西红)柿|枇杷|山楂|莲雾|人参果|人生果|石榴|无花果|^水果$'),
    ('condiment', r'^(酱油|醋|辣酱|番茄酱|沙拉酱|色拉酱|腐乳|乳腐|红腐乳|芥末|蒜蓉|酸奶油|豆瓣酱|甜面酱|蚝油|酱)$'),
]
VEG_GROUP = {'leafy': 'leafy_veg', 'special_leafy': 'leafy_veg', 'sprout': 'fruit_veg', 'fruit_veg': 'fruit_veg', 'root_stem': 'root_stem_veg', 'starchy': 'starchy_veg', 'legume': 'fresh_legume',
             'fungi': 'fungi_algae', 'generic': 'mixed_veg', 'starch_noodle': 'starch_noodles'}
FRIED_VARIANT = {'leafy_veg': 'leafy_veg_fried', 'fruit_veg': 'fruit_veg_fried', 'root_stem_veg': 'root_stem_veg_fried', 'starchy_veg': 'starchy_veg_fried',
                 'fresh_legume': 'fresh_legume_fried', 'fungi_algae': 'fungi_algae_fried', 'mixed_veg': 'mixed_veg_fried', 'pork': 'pork_fried',
                 'pork_rib': 'pork_rib_fried', 'meatball': 'meatball_fried', 'beef_mutton': 'beef_mutton_fried', 'chicken': 'chicken_fried',
                 'duck_goose': 'duck_fried', 'fish': 'fish_fried', 'shrimp': 'seafood_fried', 'shellfish': 'seafood_fried', 'tofu': 'soy_fried',
                 'soy_dried': 'soy_fried', 'egg': 'egg_fried', 'starch_noodles': 'mixed_veg_fried'}


def number(text):
    text = str(text).strip()
    if text in ('', '—'):
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
    classes = pd.read_csv(CURATION / 'fiber_classes.csv', keep_default_na=False)
    ratios = {name: float(np.median([number(rows[int(i)]['fiber']) / number(rows[int(i)]['carbohydrate']) for i in ids.split(';')])) if ids else 0.0
              for name, ids in zip(classes.fiber_class, classes.proxy_entry_ids)}
    crosswalk = pd.read_csv(CURATION / 'usda_crosswalk.csv', dtype=str, keep_default_na=False).set_index('entry_id')
    usda = load_usda({int(value) for value in pd.concat([crosswalk.fdc_id, crosswalk.fiber_fdc_id]) if value})
    records = []
    for entry in pd.read_csv(CURATION / 'entries.csv', dtype=str, keep_default_na=False).itertuples(index=False):
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
            note = (note + '；' if note else '') + '纤维高于碳水化合物，按碳水化合物封顶'
        assert all(np.isfinite(value) for value in used.values()), (entry.entry_id, used)
        if fdc:
            usda_values['fiber_g'] = usda_fiber if np.isfinite(usda_fiber) else 0.0 if entry.fiber_class == 'zero' else used['fiber_g']
            if usda_values['fiber_g'] > usda_values['carbohydrate_g']:
                usda_values['fiber_g'] = usda_values['carbohydrate_g']
                note = (note + '；' if note else '') + 'USDA变体纤维高于碳水化合物，按碳水化合物封顶'
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
    records.append({'entry_id': WATER, 'name': '水（冲调/烹煮加水；非成分表条目）', 'source': 'pseudo-entry', 'edible_pct': 100.0, 'water_g': 100.0,
                    **{key: 0.0 for key in NUTRIENTS}, **{f'{key}_used': 0.0 for key in NUTRIENTS}, 'fiber_basis': 'zero',
                    'adjustment_note': '用于米饭以外的熟制/冲调配方补水，所有营养素为0', 'usda_fdc_id': int(water.fdc_id),
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
    return 'nonfried_named' if re.search(NONOIL_TERMS, name.replace('烤麸', '')) else 'unspecified'


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
        return 'unmappable', 'unspecified', 'low', '无中文名，未自动归类'
    cook = method(name)
    for group, pattern in DIRECT:
        if re.search(pattern, name):
            return group, cook, 'medium', ''
    if re.search(r'蛋', name) and re.search(r'蒸|炖|羹', name):
        return 'egg_custard', 'nonfried_named', 'medium', '蒸蛋/炖蛋类'
    if re.search(SOUP, name):
        if re.search(r'面', name) and not re.search(r'面筋', name):
            return 'noodle_meal', 'nonfried_named', 'medium', '汤面'
        if re.search(SWEET_SOUP, name):
            return 'sweet_soup', 'nonfried_named', 'medium', '按不加糖'
        if re.search(STARCHY_DISH, name):
            return 'starchy_soup', 'nonfried_named', 'medium', '含淀粉的汤'
        found = components(name)
        meat = [kind for kind in PROTEIN if kind not in ('egg', 'soy') and kind in found]
        solids = [kind for kind in found if kind not in meat]
        if meat:
            return ('meat_veg_soup' if solids else 'broth'), 'nonfried_named', 'low', '汤：固形物比例未知'
        return 'veg_soup', 'nonfried_named', 'medium', ''
    found = components(name)
    others = [kind for kind in found if kind not in ('generic',)]
    for group, pattern in STAPLE:
        if not re.search(pattern, name):
            continue
        if group == 'filled':
            group = 'filled_veg' if re.search(FILLED_VEG, name) and not re.search(r'肉|虾|蟹|牛|鸡(?!蛋)', name) else 'filled_meat'
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
        return ('starchy_mixed_dish', cooking, 'medium', '含淀粉荤素菜') if oil else ('starchy_mixed_steamed', plain, 'medium', '含淀粉荤素菜')
    if 'egg' in found:
        if re.search(r'蒸|炖|羹', name):
            return 'egg_custard', 'nonfried_named', 'medium', '蒸蛋/炖蛋类'
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
            return ('pork_fried', cooking, 'low', '油炸加工肉按猪肉（炒）组') if oil else ('pork_processed', plain, 'medium', '')
        if kind == 'pork':
            if re.search(r'排|蹄|肘|髈|脚|爪', name):
                base = 'pork_rib'
            elif oil and re.search(r'红烧肉|五花|烧肉|扣肉|回锅|东坡', name):
                return 'pork_belly_braised', cooking, 'medium', ''
            else:
                base = 'pork'
        else:
            base = {'meatball': 'meatball', 'beef': 'beef_mutton', 'mutton': 'beef_mutton', 'chicken': 'chicken', 'duck': 'duck_goose'}[kind]
        return (FRIED_VARIANT[base], cooking, 'medium', '') if oil else (base, plain, 'medium', '')
    if 'meatball' in meat:
        return ('meatball_fried', cooking, 'medium', '') if cook == 'fried_named' else ('meatball', plain, 'medium', '馅中蔬菜按肉圆计')
    if 'soy' in found:
        if seafood:
            return ('seafood_fried', cooking, 'medium', '') if oil else ('tofu', plain, 'medium', '')
        if veg:
            return ('soy_veg_stirfry', cooking, 'medium', '') if oil else ('tofu' if re.search(r'豆腐|豆花', name) else 'soy_dried', plain, 'medium', '')
        base = 'tofu' if re.search(r'豆腐(?!干|衣|皮)|豆花', name) and not re.search(r'油豆腐', name) else 'soy_dried'
        if re.search(r'油豆腐|油面筋', name):
            return 'soy_fried', 'fried_in_entry', 'medium', '油炸豆制品'
        return (FRIED_VARIANT[base], cooking, 'medium', '') if oil else (base, plain, 'medium', '')
    if seafood:
        kind = seafood[0]
        base = {'fish_processed': 'fish_processed', 'crab': 'crab', 'shrimp': 'shrimp', 'shellfish': 'shellfish', 'fish': 'fish'}[kind]
        if base in ('fish_processed', 'crab'):
            return ('seafood_fried', cooking, 'low', '油炸蟹/鱼丸按虾贝（炒）组') if oil else (base, plain, 'medium', '')
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
        return 'root_stem_veg', plain, 'medium', '葱姜蒜类'
    return 'unmappable', cook, 'low', '未能自动归类'


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
            group, recipe, confidence, notes, assignment = 'unmappable', '', 'low', '记录无可解析重量；' + notes, 'no_weight'
        assert group in recipes, (item.key, group)
        oil_rule = ''
        if cooking == 'unspecified' and group != 'unmappable':
            if (group in O1_GROUPS or (group == 'starchy_veg' and '土豆丝' in item.item_zh)) and not re.search(O1_EXCLUDE, item.item_zh):
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
        oil_rule = oil_rule or ('O4' if re.search(r'(?<!洋)葱烤', item.item_zh) else OIL_BASIS.get(cooking, 'none'))
        values = profile(shares, entries) if shares else {key: np.nan for key in NUTRIENT_COLUMNS}
        recipe_text, recipe_names = describe(shares, entries) if level == 'specific' else ('', '')
        info[item.key] = {'shares': shares, 'group': group, 'level': level}
        rows.append({'key': item.key, 'item_zh': item.item_zh, 'lines': item.lines, 'rank': rank, 'review_priority': int(rank <= REVIEW_TOP),
                     'mapping_level': level, 'cn_entry_or_recipe': recipe_text, 'recipe_names': recipe_names, 'group_id': group, 'assignment': assignment,
                     **values, 'cooking': cooking, 'oil_rule': oil_rule, 'confidence': confidence, 'notes': notes, 'rule_group': rule_group})
    return pd.DataFrame(rows), info


def line_amounts(recorded):
    overrides = pd.read_csv(CURATION / 'amount_overrides.csv', dtype={'amount_g': float}, keep_default_na=False, na_values={'amount_g': ['']})
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
    routes = pd.read_csv(CURATION / 'line_routes.csv', keep_default_na=False).set_index(['key', 'item_zh'])
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
        elif group == 'noodles' and (re.search(r'卷面|挂面', zh) or amount <= 60):
            state, shares = 'dry_noodle', states['dry_noodle']
        elif group in ('noodles', 'noodle_meal') and amount >= 300:
            state, shares = 'noodle_soup', states['noodle_soup']
        elif group in ('filled_meat', 'filled_veg') and re.search(r'馄饨|云吞', zh) and amount >= (200 if '小馄饨' in zh else 250):
            state, shares = 'wonton_broth', states['wonton_broth']
        elif group == 'dry_cereal_powder' and re.search(r'麦片|燕麦', zh) and (row.unit == 'ml' or amount >= 150):
            state, shares = 'prepared_cereal', states['prepared_cereal']
        elif re.fullmatch(r'(煮)?玉米(棒)?|主食玉米', zh) and amount >= CORN_MIN:
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
    groups = pd.read_csv(CURATION / 'groups.csv', keep_default_na=False)
    recipes = dict(zip(groups.group_id, groups.recipe))
    group_shares = {group: expand(recipe, recipes) for group, recipe in recipes.items() if recipe}
    group_profiles = {group: profile(shares, entries) for group, shares in group_shares.items()}
    assert all(np.isclose(values['added_oil_g_per_100g'], [0, 100 * OIL_SHARE]).any() for values in group_profiles.values())
    specific = pd.read_csv(CURATION / 'specific.csv', keep_default_na=False).set_index('key')
    overrides = pd.read_csv(CURATION / 'group_overrides.csv', keep_default_na=False).set_index('key')
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

    routes = pd.read_csv(CURATION / 'line_routes.csv', keep_default_na=False)
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
    routes = pd.read_csv(CURATION / 'line_routes.csv', keep_default_na=False).set_index(['key', 'item_zh'])
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
