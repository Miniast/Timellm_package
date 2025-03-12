import pandas as pd
import json

with open('./result/validate_result.json', 'r') as f:
    data = json.load(f)

for record in data:
    pred = record['pred']
    true = record['true']
    # Batch size = 2

