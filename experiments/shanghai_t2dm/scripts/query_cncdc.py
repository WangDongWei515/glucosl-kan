from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import html
import os
import json
import re
import sys
import time


STAGE = Path(__file__).resolve().parents[3]
CACHE = STAGE / 'data/interim/shanghai_t2dm/cncdc_query_cache'
SEARCH_URL = 'https://nlc.chinanutri.cn/fq/FoodInfoQueryAction!queryFoodInfoList.do'
DETAIL_URL = 'https://nlc.chinanutri.cn/fq/foodinfo/{}.html'
# Column order of a search-result row, verified against detail pages by verify_columns().
COLUMNS = ['entry_id', 'image', 'name', 'alias', 'name_en', 'edible', 'water', 'energy', 'protein', 'fat', 'cholesterol', 'ash', 'carbohydrate', 'fiber',
           'carotene', 'vitamin_a', 'vitamin_e', 'thiamin', 'riboflavin', 'niacin', 'vitamin_c', 'ca', 'p', 'k', 'na', 'mg', 'fe', 'zn', 'se', 'cu', 'mn', 'i',
           'sfa', 'mufa', 'pufa', 'fatty_acid_total']
DETAIL_LABELS = {'食部(Edible)': 'edible', '水分(Water)': 'water', '能量(Energy)': 'energy', '蛋白质(Protein)': 'protein', '脂肪(Fat)': 'fat',
                 '胆固醇(Cholesterol)': 'cholesterol', '灰分(Ash)': 'ash', '碳水化合物(CHO)': 'carbohydrate', '总膳食纤维(Dietary fiber)': 'fiber',
                 '硫胺素(Thiamin)': 'thiamin', '核黄素(Riboflavin)': 'riboflavin', '烟酸(Niacin)': 'niacin', '钙(Ca)': 'ca', '磷(P)': 'p', '钾(K)': 'k',
                 '钠(Na)': 'na', '镁(Mg)': 'mg', '铁(Fe)': 'fe', '锌(Zn)': 'zn', '硒(Se)': 'se', '铜(Cu)': 'cu', '锰(Mn)': 'mn'}
VERIFY_IDS = [287, 1313, 1339, 988]
PAUSE_SECONDS = 1.5
REQUESTS = {'count': 0, 'last': 0.0}


def fetch(url, data=None):
    if os.environ.get('GLUCOSL_OFFLINE') == '1':
        raise RuntimeError('Food-composition cache is incomplete. Run prepare with network access or provide --cn-cache-dir.')
    wait = REQUESTS['last'] + PAUSE_SECONDS - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    request = Request(url, data=urlencode(data).encode() if data else None, headers={'User-Agent': 'Mozilla/5.0 (research; nutrient mapping)'})
    try:
        with urlopen(request, timeout=60) as response:
            status, body = response.status, response.read().decode('utf-8')
    finally:
        REQUESTS['count'] += 1
        REQUESTS['last'] = time.monotonic()
    if status != 200:
        raise RuntimeError(f'HTTP {status} for {url}; stopping')
    return body


def cache_file(prefix, term, page=None):
    safe = re.sub(r'[\\/:*?"<>|\s]+', '_', term) or 'ALL'
    return CACHE / (f'{prefix}_{safe}' + (f'_p{page:03d}' if page else '') + '.json')


def search_page(term, page):
    path = cache_file('search', term, page)
    if not path.exists():
        params = {'categoryOne': '0', 'categoryTwo': '0', 'foodName': term, 'pageNum': str(page), 'field': '0', 'flag': '0'}
        body = fetch(SEARCH_URL, params)
        try:
            response = json.loads(body)
        except json.JSONDecodeError:
            raise RuntimeError(f'Non-JSON response for {term!r} page {page}; stopping: {body[:200]}')
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'query_term': term, 'page': page, 'endpoint': SEARCH_URL, 'params': params,
                                    'retrieved_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'response': response}, ensure_ascii=False, indent=1))
    return json.loads(path.read_text())


def search(term):
    first = search_page(term, 1)
    return [first] + [search_page(term, page) for page in range(2, int(first['response']['totalPages']) + 1)]


def detail(entry_id):
    path = cache_file('detail', str(entry_id))
    if not path.exists():
        url = DETAIL_URL.format(entry_id)
        body = fetch(url)
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'entry_id': entry_id, 'url': url, 'retrieved_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'html': body},
                                   ensure_ascii=False, indent=1))
    return json.loads(path.read_text())


def cached_rows():
    rows = {}
    for path in sorted(CACHE.glob('search_*.json')):
        cached = json.loads(path.read_text())
        for row in cached['response']['list']:
            assert len(row) == len(COLUMNS), (path.name, row)
            rows.setdefault(row[0], {**dict(zip(COLUMNS, row)), 'query_term': cached['query_term'], 'page': cached['page'], 'retrieved_utc': cached['retrieved_utc']})
    return rows


def verify_columns(rows):
    report = []
    for entry_id in VERIFY_IDS:
        page = detail(entry_id)['html']
        name = html.unescape(re.search(r'<title>(.*?)-食物营养成分查询平台', page).group(1)).strip()
        values = {DETAIL_LABELS[label]: html.unescape(value).strip() for label, value in
                  re.findall(r'class="td_left">\s*([^<]+?)\s*</td>\s*<td>([^<]*)</td>', page) if label in DETAIL_LABELS}
        row = rows[entry_id]
        mismatches = {field: (row[field], value) for field, value in values.items() if row[field] != value}
        assert name == row['name'] and len(values) == len(DETAIL_LABELS) and not mismatches, (entry_id, name, mismatches)
        report.append(f'{entry_id} {name}: {len(values)} labelled fields identical to search-row columns')
    return report


def main():
    terms = sys.argv[1:] or ['']
    for term in terms:
        pages = search(term)
        print(f'{term or "(all foods)"}: {len(pages)} pages, {sum(len(page["response"]["list"]) for page in pages)} rows', flush=True)
    rows = cached_rows()
    if not sys.argv[1:]:
        print('\n'.join(verify_columns(rows)))
    print(f'{len(rows)} distinct entries cached; {REQUESTS["count"]} HTTP requests made in this run')


if __name__ == '__main__':
    main()
