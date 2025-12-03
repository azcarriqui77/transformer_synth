import pandas as pd
import numpy as np
import os
from .data import get_timeseries_by_date
from datetime import datetime, timedelta


def get_segmentation_alarms(path: str, n_days: int) -> pd.DataFrame:
    """Debemos definir como calcular las alarmas por segmentación.
    Ahora mismo esta funcion devuelve siempre un vector de alarma 
    vacío. 

    Args:
        path (str): _description_

    Returns:
        pd.DataFrame: DataFrame con los valores de alarma para cada región.
    """
    return None


def get_temp_alarms(temperatures: pd.DataFrame, prediction: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Esta función analiza cada una de las regiones para calcular la desviación de temperatura 
    en tanto por ciento entre el valor registrado y el valor predicho por el modelo. Si esta desviación
    supera el umbral establecido, generamos una alarma en esa región. 

    Args:
        temperatures (pd.DataFrame): Valores reales de temperatura registrados para cada región.
        prediction (pd.DataFrame): Valores de temperatura predichos para cada región.
        cfg (dict): Parametros de configuración adicionales.

    Returns:
        pd.DataFrame: Alarmas generadas para cada región.
    """
    alarms = pd.DataFrame()
    F = pd.DataFrame()
    
    # Análizamos todas las regiones. Si la desviación de temperatura es
    # mayor del umbral establecido, generamos una alarma en esa región.
    for region in temperatures:
        registered = temperatures[region].values[0]
        predicted = prediction[region].values[0]
    
        if predicted != 'NaN':
            # Porcentual:
            # dev = abs((predicted * 100 / registered) - 100)
            
            # Absoluta:
            # dev = abs(predicted - registered)
            # alarms[region] = [1] if dev > cfg['max_deviation'] else [0]
            
            # Normalizada:
            k = 2
            F_value = np.maximum(0, (registered**k - predicted**k) / (registered**k + predicted**k))
            alarms[region] = [1] if F_value > cfg['alarm_threshold'] else [0]
            F[region] = [round(F_value,2)]
            
        else:
            # Si estamos aquí es que no existen predicciones, por lo que enviamos en el vector 
            # de alarmas un código de error:
            # CÓDIGO 40 - No existen predicciones.
            alarms[region] = [40]
            F[region] = [40]

    return alarms.set_index(temperatures.index), F.set_index(temperatures.index)
