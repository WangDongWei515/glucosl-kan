from pathlib import Path
import re

import numpy as np
import pandas as pd


STAGE = Path(__file__).resolve().parents[1]
SOURCE = STAGE / 'data/raw/shanghai_t2dm/v3/extracted'
# Every T2DM file has these 11 columns in this order; header spellings vary, so columns are mapped by position.
RECORD_COLUMNS = ['time', 'cgm_mgdl', 'cbg_mgdl', 'ketone_mmoll', 'diet_en', 'diet_zh', 'insulin_sc', 'oral_agents', 'csii_bolus_iu', 'csii_basal_iu_h', 'insulin_iv']
INSULIN_COLUMNS = ['insulin_sc', 'csii_bolus_iu', 'csii_basal_iu_h', 'insulin_iv']
SUMMARY_FIELDS = {
    'Patient Number': 'record_id', 'Gender (Female=1, Male=2)': 'sex_code', 'Age (years)': 'age', 'BMI (kg/m2)': 'bmi',
    'Duration of diabetes (years)': 'diabetes_years', 'Comorbidities': 'comorbidities', 'Hypoglycemic Agents': 'agents', 'Other Agents': 'other_agents',
    'Fasting Plasma Glucose (mg/dl)': 'fasting_glucose_mgdl', '2-hour Postprandial Plasma Glucose (mg/dl)': 'pp2h_glucose_mgdl',
    'Fasting C-peptide (nmol/L)': 'fasting_cpeptide_nmoll', 'Fasting Insulin (pmol/L)': 'fasting_insulin_pmoll', 'HbA1c (mmol/mol)': 'hba1c_mmolmol',
    'Glycated Albumin (%)': 'glycated_albumin_pct', 'Total Cholesterol (mmol/L)': 'cholesterol_mmoll', 'Triglyceride (mmol/L)': 'triglycerides_mmoll',
    'High-Density Lipoprotein Cholesterol (mmol/L)': 'hdl_mmoll', 'Low-Density Lipoprotein Cholesterol (mmol/L)': 'ldl_mmoll',
}
# Conversions to the units used in CGMacros: NGSP master equation, insulin 6.0 pmol/L per uIU/mL, cholesterol 38.67 and triglycerides 88.57 mg/dL per mmol/L.
CONVERSIONS = {
    'hba1c_pct': ('hba1c_mmolmol', lambda value: 0.09148 * value + 2.152),
    'insulin': ('fasting_insulin_pmoll', lambda value: value / 6.0),
    'cholesterol': ('cholesterol_mmoll', lambda value: value * 38.67),
    'hdl': ('hdl_mmoll', lambda value: value * 38.67),
    'ldl': ('ldl_mmoll', lambda value: value * 38.67),
    'triglycerides': ('triglycerides_mmoll', lambda value: value * 88.57),
}
DRUG_CLASSES = {
    'insulin': r'insulin|novolin|humulin|gansulin|gensulin|\b\d{2}r\b|\d{2}/\d{2}|glargine|detemir|degludec|aspart|lispro|glulisine|lantus|levemir|tresiba',
    'metformin': r'metformin',
    'alpha_glucosidase_inhibitor': r'acarbose|voglibose|miglitol',
    'sulfonylurea': r'gliclazide|glimepiride|glipizide|gliquidone|glibenclamide|glyburide',
    'glinide': r'glinide',
    'dpp4_inhibitor': r'gliptin',
    'thiazolidinedione': r'glitazone',
    'sglt2_inhibitor': r'gliflozin',
    'glp1_receptor_agonist': r'glutide|exenatide|lixisenatide',
}
NOT_RECORDED = {'data not available', '未记录'}
ITEM_PATTERN = re.compile(r'^(?P<name>.*?)[\s,，:：]*(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>g|ml|克|毫升)[\s.。]*$', re.IGNORECASE)
AMOUNT_PATTERN = re.compile(r'\d+(?:\.\d+)?\s*(?:g|ml|m|克|毫升)?', re.IGNORECASE)


def drug_classes(text):
    text = str(text).lower() if pd.notna(text) else ''
    return {name for name, pattern in DRUG_CLASSES.items() if re.search(pattern, text)}


def unmatched_agent_terms(text):
    terms = [term.strip(' \xa0') for term in re.split(r'[,，;；]', str(text).lower())] if pd.notna(text) else []
    return [term for term in terms if term and term != 'none' and not drug_classes(term)]


def read_summary():
    summary = pd.read_excel(SOURCE / 'Shanghai_T2DM_Summary.xlsx')
    summary.columns = [' '.join(str(column).split()) for column in summary.columns]
    assert set(SUMMARY_FIELDS) <= set(summary.columns)
    table = summary[list(SUMMARY_FIELDS)].rename(columns=SUMMARY_FIELDS)
    assert table.record_id.is_unique and table.record_id.str.fullmatch(r'\d{4}_\d+_\d{8}').all()
    table.insert(1, 'patient_id', table.record_id.str[:4])
    # Missing laboratory values are written as "/" in the summary sheet.
    numeric = [field for field in table.columns if field not in ['record_id', 'patient_id', 'comorbidities', 'agents', 'other_agents']]
    table[numeric] = table[numeric].apply(pd.to_numeric, errors='coerce')
    table['sex'] = table.sex_code - 1
    for output, (source, convert) in CONVERSIONS.items():
        table[output] = convert(pd.to_numeric(table[source], errors='coerce'))
    table['homa_ir'] = table.insulin * table.fasting_glucose_mgdl / 405
    return table


def read_record(path):
    frame = pd.read_excel(path)
    assert len(frame.columns) == len(RECORD_COLUMNS) and str(frame.columns[0]).strip() == 'Date', path.name
    frame.columns = RECORD_COLUMNS
    frame['time'] = pd.to_datetime(frame.time, errors='raise')
    for column in ['cgm_mgdl', 'cbg_mgdl', 'ketone_mmoll', 'csii_bolus_iu', 'csii_basal_iu_h']:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    return frame


def record_paths():
    return sorted((SOURCE / 'Shanghai_T2DM').glob('*.xls*'))


def parse_intake(english, chinese):
    lines_en = [line.strip() for line in str(english).split('\n') if line.strip()] if pd.notna(english) else []
    lines_zh = [line.strip() for line in str(chinese).split('\n') if line.strip()] if pd.notna(chinese) else []
    paired = len(lines_en) == len(lines_zh)
    items = []
    for number, line in enumerate(lines_en):
        match = ITEM_PATTERN.match(line)
        text_zh = lines_zh[number] if paired else np.nan
        items.append({
            'line': number, 'text_en': line, 'text_zh': text_zh, 'zh_line_count_matches': paired,
            'item_zh': AMOUNT_PATTERN.sub('', text_zh).strip().strip(',，.。').strip() if paired else np.nan,
            'not_recorded': line.lower() in NOT_RECORDED, 'parsed': bool(match and match['name']),
            'item_en': match['name'].strip() if match else np.nan,
            'amount': float(match['amount']) if match else np.nan,
            'unit': {'克': 'g', '毫升': 'ml'}.get(match['unit'], match['unit'].lower()) if match else np.nan,
        })
    return items
