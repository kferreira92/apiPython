import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass



@dataclass(frozen=True)
class StaticAtts:

    categ_features = ['sight_left_class', 'sight_right_class',
                        'blood_pressure_class', 'blood_glucose_class',
                        'serum_creatinine_class', 'SGOT_AST_class',
                        'SGOT_ALT_class', 'gamma_GTP_class']
    
    encoder = pickle.load(open('features/artifacts/ordinal.pkl' , 'rb'))

def get_bmi(row):
    weight = row["weight"]
    height = row["height"]
    return weight / (height ** 2)

def get_blood_pressure_ratio(row):
    sbp = row["SBP"]
    dbp = row["DBP"]
    return sbp / dbp

def get_hdl_cholestorol_ratio(row):
    hdl_chole = row["HDL_chole"]
    total_chole = row["tot_chole"]
    return hdl_chole / total_chole

def get_ldl_cholestorol_ratio(row):
    ldl_chole = row["LDL_chole"]
    total_chole = row["tot_chole"]
    return ldl_chole / total_chole

def classify_eyesight(val):
    if val > 1: 
        return "Good"
    elif val > 0.5: 
        return "Average"
    elif val > 0.25: 
        return "Poor"
    else:
        return "Very Poor"

def classify_blood_pressure(row):
    sbp = row["SBP"]
    dbp = row["DBP"]
    if sbp < 120 and dbp < 80:
        return "Normal"
    elif sbp < 130 and dbp < 80:
        return "Elevated"
    elif sbp < 140 and dbp < 90:
        return "Hypertension Stage 1"
    elif sbp < 180 and dbp < 120:
        return "Hypertension Stage 2"
    else:
        return "Hypertensive Crisis"

def classify_blood_glucose(row):
    blood_glucose = row["BLDS"]
    if blood_glucose < 100:
        return "Normal"
    elif blood_glucose < 125:
        return "Pre-Diabetes"
    else:
        return "Diabetes"

def classify_serum_creatinine(row):
    serum_creatinine = row["serum_creatinine"]
    if serum_creatinine < 2.7:
        return "Normal"
    else:
        return "Abnormal"

def classify_SGOT_AST(row):
    sgot_ast = row["SGOT_AST"]
    if sgot_ast < 40:
        return "Normal"
    else:
        return "Abnormal"

def classify_SGOT_ALT(row):
    sgot_alt = row["SGOT_ALT"]
    if sgot_alt < 40:
        return "Normal"
    else:
        return "Abnormal"

def classify_gamma_GTP(row):
    gamma_gtp = row["gamma_GTP"]
    sex = row["sex"]
    if sex == "Male":
        if gamma_gtp >= 11 and gamma_gtp <= 63:
            return "Normal"
        else:
            return "Abnormal"
    else:
        if gamma_gtp >= 8 and gamma_gtp <= 35:
            return "Normal"
        else:
            return "Abnormal"

def preprocess(df):
    df = pd.DataFrame(data = df, index=[0])
    df['sex'] = np.where(df['sex'].values=='Female',0,1)
    df["bmi"] = df.apply(get_bmi, axis=1)
    df["BP_ratio"] = df.apply(get_blood_pressure_ratio, axis=1)
    df["HDL_ratio"] = df.apply(get_hdl_cholestorol_ratio, axis=1)
    df["LDL_ratio"] = df.apply(get_ldl_cholestorol_ratio, axis=1)
    df["sight_left_class"] = df["sight_left"].apply(classify_eyesight)
    df["sight_right_class"] = df["sight_right"].apply(classify_eyesight)
    df["blood_pressure_class"] = df.apply(classify_blood_pressure, axis=1)
    df["blood_glucose_class"] = df.apply(classify_blood_glucose, axis=1)
    df["serum_creatinine_class"] = df.apply(classify_serum_creatinine, axis=1)
    df["SGOT_AST_class"] = df.apply(classify_SGOT_AST, axis=1)
    df["SGOT_ALT_class"] = df.apply(classify_SGOT_ALT, axis=1)
    df["gamma_GTP_class"] = df.apply(classify_gamma_GTP, axis=1)

    df[StaticAtts.categ_features] = StaticAtts.encoder.transform(df[StaticAtts.categ_features])

    return df
