import os
# import sys
# sys.path.append("..")
import numpy as np
import time
from model.models import XGB
from pydantic import BaseModel
from fastapi import status, HTTPException, FastAPI
from features.preprocessing import preprocess 
from dataclasses import dataclass



class Input(BaseModel):
    """
    Classe que vai ser utilizada pela API para determinar as variáveis de entrada e seus tipos.
    """
    sex: str
    age: int
    height: int
    weight: int
    waistline: float
    sight_left: float
    sight_right: float
    hear_left: float
    hear_right: float
    SBP: float
    DBP: float
    BLDS: float
    tot_chole: float
    HDL_chole: float
    LDL_chole: float
    triglyceride: float
    hemoglobin: float
    urine_protein: float
    serum_creatinine: float
    SGOT_AST: float
    SGOT_ALT: float
    gamma_GTP: float
    SMK_stat_type_cd: float

@dataclass(frozen=True)
class Dir:
    """
    Classe com o caminho dos diretórios que contem artefatos que serão utilizados para fazer a previsão.
    """
    model_dir = 'model/artifacts/xgboost.pkl'
    scaler_dir = 'model/artifacts/scaler.pkl'

class BooseProbability(BaseModel):
    """
    Classe que retorna a probabilidade da predição.
    """
    
    probability: float



model = XGB(Dir.model_dir,Dir.scaler_dir)

app = FastAPI()
@app.post('/boose/predict', response_model=BooseProbability, status_code=status.HTTP_200_OK)
async def run_model(input: Input) -> float:
    """
    Recebe os dados

    Transforma em um dicionário

    Preprocessa e realiza a predição

    """
    dataframe = preprocess(input.dict())
    y_hat = model.realiza_previsao(dataframe)

    return BooseProbability(probability=y_hat)

@app.get('/')
async def root():
    return 'Drinking API'
