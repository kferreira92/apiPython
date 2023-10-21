from features.preprocessing import preprocess
from model.models import XGB
import pandas as pd

model_dir = 'model/artifacts/xgboost.pkl'
scaler_dir = 'model/artifacts/scaler.pkl'
entry = {
        'sex': 'Male',
        'age': 30,
        'height': 180,
        'weight': 80,
        'waistline': 89.0,
        'sight_left': 0.9,
        'sight_right': 1.2,
        'hear_left': 1.0,
        'hear_right': 1.0,
        'SBP': 130.0,
        'DBP': 82.0,
        'BLDS': 106.0,
        'tot_chole': 228.0,
        'HDL_chole': 55.0,
        'LDL_chole': 148.0,
        'triglyceride': 121.0,
        'hemoglobin': 15.8,
        'urine_protein': 1.0,
        'serum_creatinine': 0.9,
        'SGOT_AST': 20.0,
        'SGOT_ALT': 36.0,
        'gamma_GTP': 27.0,
        'SMK_stat_type_cd': 3.0
        }

df = preprocess(entry)

boost = XGB(model_dir,scaler_dir).realiza_previsao(df)
print(boost)
# print(df)
