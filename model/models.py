import joblib
import pickle
import numpy as np
from xgboost import XGBClassifier
from abc import ABC, ABCMeta, abstractmethod

class XGBoostModel(ABC, metaclass=ABCMeta):
    """
    Classe base.
    """

    @abstractmethod
    def realiza_previsao(self, dataset) -> float:
        """Verifica se os dados fornecidos configuram um cenário de alcoolismo."""
        pass

class XGB(XGBoostModel):
    """

    Parâmetros
    ----------
    __modelo : modelo importado utilizando joblib
        O modelo treinado fornecido que contém os pesos a serem utilizados pelo algoritmo selecionado.
    """

    def __init__(self, diretorio_modelo: str, diretorio_scaler:str) -> None:
        """Método que inicializa a classe já importando o modelo fornecido no diretório especificado"""
        self.__modelo = pickle.load(open(f'{diretorio_modelo}' , 'rb'))
        self._scaler = joblib.load(diretorio_scaler)
    
    def realiza_previsao(self, dataset) -> float:
        """
        Verifica se os dados fornecidos configuram um cenário de alcoolismo.
    
        Parâmetros
        ----------
        dataset
        Contém informações da saude do indivíduo.
        
        Retorno
        ----------
        float:
        Float que indica a probabilidade do indivíduo ser alcoólatra.

        """
        dataset = self._scaler.transform(dataset)
        result = self.__modelo.predict_proba(dataset)
        return float(result[0][1])

# class ModelFactory(ABC, metaclass=ABCMeta):
#     """
#     Abstract Factory que declara um conjunto de métodos que devem retornar
#     diferentes objetos para criação de modelos preditivos a depender dos
#     algoritmos, parâmetros e tecnologias utilizadas.
#     """
    
#     @abstractmethod
#     def cria_xgboost(self) -> XGBoostModel:
#         """Método base para criar modelos preditivos utilizando xgboost"""
#         pass

# class XGBoostFactory(ModelFactory):
#     """
#     Factory que cria uma família de produtos que pertencem a uma mesma variante. A ideia desta classe é grantir que
#     os modelos resultantes sejam compatíveis com as regras e métodos necessários.
#     """

#     def cria_xgboost(self, caminho_modelo: str, caminho_scaler:str) -> XGB:
#         """Método que cria uma classe de modelos que utilizem Isolation Forest via scikit-learn"""
#         return XGB(caminho_modelo,caminho_scaler)