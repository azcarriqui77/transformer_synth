import os
import glob
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from modules.data import append_timeseries

def medfilt(x, k=5):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i+1)] = x[j:]
        y[-j:, -(i+1)] = x[-1]
    return np.median(y, axis=1)

def remove_spikes(series, threshold=3):
    for i in range(1, len(series)):
        # Si la diferencia entre el valor actual y el anterior excede el umbral, reemplazar
        if abs(series.iloc[i] - series.iloc[i - 1]) > threshold:
            series.iloc[i] = series.iloc[i - 1]
    return series


def train_ar(train: pd.Series, order:int) -> int:
    model = AutoReg(train, lags=order, old_names=False)
    return model.fit()


def get_predictions(timestamp: str, temperatures: pd.DataFrame, order:int, cfg:dict) -> bool:
    # Generamos un DataFrame vacío de predicciones:
    empty_predictions = pd.DataFrame('NaN', index = np.arange(1), columns = temperatures.columns.values)
    empty_predictions.set_index(temperatures.index, inplace=True)
    
    print('Llamada a get_predictions: ', timestamp)
    query_time = pd.to_datetime(timestamp, format='%Y-%m-%d-%H-%M-%S')
    # Comprobamos si existe la carpeta de predicciones y sino la creamos:
    predictions_path = os.path.join(cfg['folder'], f'order_{order}')
    os.makedirs(predictions_path, exist_ok=True)

    # Comporbamos si hay archivos de predicciones. Si no los hay, tenemos que generarlos:
    csv_filename = sorted(glob.glob(os.path.join(predictions_path, '*.csv')))
    
    # Eliminamos archivos de predicciones antiguos:
    if len(csv_filename) > 50:
        predictions_to_remove = csv_filename[:-50]
        for file in predictions_to_remove:
            os.remove(file)
        
    if csv_filename:
        # Leemos el último archivo de predicciones disponibles:
        print('Leemos último archivo de predicciones disponible: ', csv_filename[-1])
        predictions = pd.read_csv(
            csv_filename[-1], index_col='timestamp', parse_dates=True)

        # Comprobamos si el punto temporal actual está entre los intervalos predichos:
        start_date = predictions.index.values[0]
        end_date = predictions.index.values[-1]
        print('Mostramos los intervalos. Start: ', start_date, 'End: ' , end_date, 'Query: ', query_time)
        if start_date <= query_time <= end_date:
            # Buscamos la predicción más cercana:
            closest_index = predictions.index.get_indexer(
                [query_time], method='nearest')
            print('Devolvemos el índice más cercano: ', closest_index)
            return predictions.iloc[closest_index]
        else:
            # Generamos nuevas predicciones.
            # En caso de que la generación de predicciones falle no devolvemos nada:
            print('El query time no está entre start_date y end_date. Llamada recursiva a get predictions si generate_predictions es TRUE.')
            return get_predictions(query_time, temperatures, order, cfg) if generate_predictions(query_time, order, cfg) else empty_predictions
    else:
        # Generamos nuevas predicciones:
        # En caso de que la generación de predicciones falle no devolvemos nada:
        print('No hay archivos de predicciones. Llamada recursiva a get predictions si generate_predictions es TRUE.')
        return get_predictions(query_time, temperatures, order, cfg) if generate_predictions(query_time, order, cfg) else empty_predictions


def generate_predictions(query_time: pd.Timestamp, order:int, cfg: dict):
    print('Llamada a generate predictions. QUERY: ', query_time)
    # Inicializamos la configuración:
    days_to_load = cfg['model']['days_of_data']
    data_path = cfg['temperatures_folder']

    # Inicializamos los dataframes para el entrenamiento y las predicciones:
    training_data = pd.DataFrame()
    prediction = pd.DataFrame()

    # Comporobamos si hay suficientes datos para entrenar nuestros modelos:
    csv_files = sorted(glob.glob(os.path.join(data_path, '*.csv')))
    if len(csv_files) < days_to_load:
        print('No hay suficientes datos para generar predicciones.')
        return False
    else:
        # Generamos los datos de entrenamiento cargando los últimos csv
        # y concatenandolos en un solo df:
        csv_files = csv_files[-days_to_load:]
        for path in csv_files:
            csv_file = pd.read_csv(
                path, index_col='timestamp', parse_dates=True)
            training_data = pd.concat([training_data, csv_file], axis=0)

        # Bug fixed: Nos aseguramos que hay suficientes datos para entrenar:
        n_samples, regions = training_data.shape
        
        print('Tamaño del set de entrenamiento:', n_samples, ' Orden: ', order)
        if (n_samples/2)+1 < order:
            print('No existen en los archivos diarios suficientes muestras de temperatura para entrenar')
            return False
        
        # Aplicamos el filtro de mediana para eliminar glitches:
        for region in training_data:
            training_data[region] = medfilt(training_data[region].values, 5)
            training_data[region] = training_data[region].bfill()
        
        # Eliminamos saltos bruscos:
        training_data = training_data.apply(remove_spikes)
        
        # Aplicamos suavizado:
        training_data = training_data.apply(lambda x: x.rolling(window=5, min_periods=1, center=True).mean())
        
        # Eliminamos la última muestra de temperatura (es la actual)
        training_data.drop(training_data.tail(1).index,inplace=True)
        
        # Entrenamos el modelo:
        # original_index = training_data.index.values
        training_data = training_data.reset_index(drop=True)

        # Calculamos el número de muestras a predecir en función de la ventana de predicción
        # y de la frequencia de adquisición de imágenes.
        freq = cfg['model']['freq']
        prediction_length = cfg['model']['prediction_length']
        prediction_length = round(prediction_length / freq)

        # Generamos predicciones desde este instante temporal en adelante.
        for region in training_data:
            print('Entrenando:', region)
            
            try:
                trained_model = train_ar(training_data[region], order)
                prediction[region] = trained_model.predict(
                    start=len(training_data), end=len(training_data) + prediction_length-1)
                prediction = prediction.round(2)
                print("Predicción generada")
                # Filtro para evitar predicciones anómalas inestables:
                if any(abs(x) > (200 + 273.15) for x in prediction[region]): # Las temperaturas de entrenamiento en grados Kelvin
                    print('Error en el re-entreno. Posible inestabilidad de las predicciones.')
                    return False
            except Exception as error:
                 print('Error en el re-entreno. No se han generado predicciones: ', error)
                 return False

        # Generamos y actualizamos el nuevo índice de tiempos:
        freq_dates = cfg['model']['freq_dates']
        datelist = pd.date_range(
            query_time, periods=prediction_length, freq=freq_dates).tolist()
        prediction = prediction.set_index(pd.to_datetime(datelist))

        # Guardamos el archivo de predicciones:
        file_name = pd.to_datetime(prediction.index.values[-1])
        file_name = file_name.strftime('%Y-%m-%d-%H-%M-%S') + '.csv'
        prediction.to_csv(os.path.join(
            cfg['folder'], f'order_{order}', file_name), index=True, index_label='timestamp')
        return True


        
