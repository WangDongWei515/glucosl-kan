from pathlib import Path
import csv
root = Path(__file__).resolve().parents[1]
with (root / 'results/glucosl/evaluation/metrics.csv').open() as stream:
    rows = list(csv.DictReader(stream))
for model in ['glucosl_kan', 'xgboost', 'constant']:
    row = next(r for r in rows if r['set'] == 'shanghai_B' and r['task'] == 'regression' and r['model'] == model and r['metric'] == 'macro_mae')
    print(f"{model}: MAE {float(row['estimate']):.2f}, 95% CI {float(row['low']):.2f} to {float(row['high']):.2f} mg/dL*h")
with (root / 'results/glucosl/evaluation/differences.csv').open() as stream:
    rows = list(csv.DictReader(stream))
row = next(r for r in rows if r['set'] == 'shanghai_B' and r['task'] == 'regression' and r['comparison'] == 'glucosl_kan minus xgboost')
print(f"GlucoSL-KAN minus XGBoost: {float(row['estimate']):.2f}, 95% CI {float(row['low']):.2f} to {float(row['high']):.2f}")
