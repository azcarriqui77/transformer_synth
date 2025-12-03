import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime


def get_timeseries_by_date(path: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Esta función importa las series temporales de temperatura que se encuentren 
    en el rango de fechas que se le pasen como argumento. Para ello genera un DataFrame
    de pandas en el que se devolverá la serie temporal completa de cada región. El
    índice del dataFrame son fechas datetime.

    Args:
        path (str): Ruta en la que se encuentran las series temporales a cargar.
        start_date (_type_): Fecha de inicio en formato: yyyy-mm-dd
        end_date (_type_): Fecha de fin en formato: yyyy-mm-dd

    Returns:
        pd.DataFrame: DataFrame que contiene cada una de las series temporales.
    """
    # Inicializamos la lista donde almacenaremos los dataFrames diarios:
    data_frames = []

    # Convertimos las fechas que hemos pasado como argumento:
    if type(start_date) is str:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Lista completa de archivos:
    files = sorted(os.listdir(path))
    csv_files = [file for file in files if file.endswith('.csv')]

    valid_date_formats = ["%Y-%m-%d", "%Y-%m-%d-%H-%M-%S"]
    # Iteramos sobre todos los archivos csv:
    for file in csv_files:
        for date_format in valid_date_formats:
            try:
                date = datetime.strptime(file.split('.')[0], date_format)
                if start_date <= date <= end_date:
                    csv_path = os.path.join(path, file)
                    df = pd.read_csv(csv_path)
                    df.set_index('timestamp', inplace=True)
                    data_frames.append(df)
            except ValueError:
                pass

    # Si no coinciden las fechas:
    if not data_frames:
        print("No se han encotrado series temporales en las fechas solicitadas o el formato de fecha introducido no es correcto.")
        return None

    return pd.concat(data_frames, ignore_index=False)


def get_last_image(path: str, file_filter: str = '*.npy', folder_filter: str = '*') -> tuple[np.ndarray, str]:    
    """Esta función carga la última imágen disponible y la devuelve 
    junto con el timestamp en el que esta fue adquirida.

    Args:
        path (str): Ruta a la carpeta de matrices.

    Returns:
        tuple[np.ndarray, str]: Array con la imagen vectorizada
        y el correspondiente timestamp.
    """
    # Lista de ordenada de subcarpetas y nos quedamos con la última:
    folder = get_subfolders(path, folder_filter)[-1]
    # Lista ordenada de archivos y nos quedamos con el último:
    image_path = sorted(glob.glob(os.path.join(folder, file_filter)))[-1]
    # Cargamos la imagen y calculamos su timestamp:
    return np.load(image_path), get_timestamp(image_path)


def get_masks(path: str) -> pd.DataFrame:
    """Esta funcion lee el archivo .csv o .npy con las máscaras y devuelve los datos 
    en un DataFrame.

    Args:
        path (str): Ruta la carpeta contenedora.

    Returns:
        pd.DataFrame: DataFrame con las máscaras.
    """
    if os.path.exists(os.path.join(path, 'template-mask.npy')):
        mask = np.load(os.path.join(path, 'template-mask.npy'))
        df_masks = pd.DataFrame()
        for i in range(int(np.max(mask) + 1)):
            df_masks[f'roi_{i}'] = (mask == i).flatten()
    else:
        df_masks = pd.read_csv(os.path.join(path, 'masks.csv'))
    
    return df_masks



def get_last_data(path: str, days: int = None, filter: str = '*.npy') -> tuple[np.ndarray, list]:
    """Esta función devuevle un array bidimensional de numpy el cual
    contiene por filas las imagenes vectorizadas correspondientes al 
    número de días que le pasamos como argumento.

    Args:
        path (str): Ruta a la carpeta que contiene las imágenes.
        days (int, optional): Número de días a cargar. 
        Si no se le pasa ningún día, carga todas las disponibles.

    Returns:
        tuple[np.ndarray, list]: Array de numpy bidimensional con las imágenes 
        vectorizadas por filas. Lista con los timestamps correspondientes.
    """
    # Inicialización de la matriz de datos:
    data = None

    # Lista de carpetas diarias:
    subfolders = get_subfolders(path)
    # Si no hay suficientes como para cubrir los días de entrenamiento
    # leemos las que tengamos:
    if days is not None and days < len(subfolders):
        subfolders = subfolders[-days:]

    # Obtenemos las imágenes para cada día:
    for folder in subfolders:
        data_folder, times_folder = get_files(folder, filter)
        if data is None:
            data = data_folder
            timestamps = times_folder
        else:
            data = np.vstack([data, data_folder])
            timestamps += times_folder
    return data, timestamps


def get_subfolders(path: str, filter:str = '*') -> list:
    """Esta función devuelve una lista ordenada de directorios contenidos en
    la ruta que pasamos como argumento. Se asegura de devolver solo los directorios
    y no archivos.

    Args:
        path (str): Ruta contenedora.

    Returns:
        list: Lista ordenada de directorios.
    """
    return sorted(glob.glob(os.path.join(path, filter)))


def get_files(path: str, filter: str = '*.npy') -> np.ndarray:
    """Esta función devuevle un array bidimensional de numpy el cual
    contiene por files las imagenes vectorizadas contenidas en la ruta 
    que le pasamos como argumento.

    Args:
        path (str): Ruta a la carpeta contenedora de las imágenes.
        filter (str, optional): Filtro para cargar archivos. Por defento: '*.npy'.

    Returns:
        np.ndarray: Array de numpy bidimensional con las imágenes.
    """
    data = None
    timestamps = []
    print(f'Loading data: {path}')
    files = sorted(glob.glob(os.path.join(path, filter)))
    for i, file in enumerate(files):
        if data is not None:
            data[i] = np.load(file)
            timestamps.append(get_timestamp(file))
        else:
            # Preasignación para una carga MUCHO más rápida.
            temp = np.load(file)
            data = np.ndarray([len(files), len(temp)])
            data[i] = temp
            timestamps = [get_timestamp(file)]
    return data, timestamps


def get_timestamp(path: str) -> str:
    """Esta función devuelve a partir del nombre de archivo el momento
    temporal en el que la imagen fue adquirida.

    Args:
        path (str): Ruta al archivo de imagen.

    Returns:
        str: Cadena de caracteres correspondiente a la marca temporal.
    """
    file_name = os.path.basename(path)
    file_parts = file_name.split('_')
    file_parts = file_parts[-2] + '-' + file_parts[-1].removesuffix('.npy')
    return file_parts


def append_timeseries(timeseries: pd.DataFrame, path: str) -> None:
    """Esta función añade una fila a un archivo .csv donde se almacenan
    el historico de datos de forma diaria. Si no existe el archivo lo crea.

    Args:
        timeseries (pd.DataFrame): Valores a almacenar en la última fila.
        path (str): Ruta de guardado.
    """
    # Generamos el nombre del archivo para el historico diario:
    # timestamp = timeseries.index.date[0]
    daily_file = timeseries.index[0].strftime('%Y-%m-%d') + '.csv'
    daily_file_path = os.path.join(path, daily_file)

    # Si no existe el archivo diario lo creamos:
    if not os.path.exists(daily_file_path):
        save_dfdata(daily_file, path, timeseries, True)
        return

    # Añadimos una línea al archivo existente:
    timeseries.to_csv(daily_file_path, mode='a', header=False)


def save_npydata(fname: str, folder: str, data: np.ndarray) -> None:
    """Esta función guarda un array de nompy en la carpeta especificada
    con el nombre de archivo especificado. Si no existe la carpeta la crea.

    Args:
        file_name (str): Nombre del archivo a guardar.
        folder (str): Ruta de guardado.
        data (ndarray): Array de datos a guardar.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    np.save(os.path.join(folder, fname), data)


def save_dfdata(fname: str, folder: str, data: pd.DataFrame, idx: bool = True) -> None:
    """Esta funcion guarda un DataFrame en la carpera especificada con el nombre de 
    archivo especificado. Si no existe la carpeta la crea. Además podemos indicarle
    si queremos que nos guarde el vector de índices o no.

    Args:
        fname (str): Nombre del archivo a guardar.
        folder (str): Ruta de guardado.
        data (pd.DataFrame): El DataFrame con los catos
        idx (bool, optional): Booleano para almacenar o no el vector de índices. 
        Por defecto lo almacena.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    data.to_csv(os.path.join(folder, fname), index=idx)


def save_alarms_table(alarms: pd.DataFrame, predicted:pd.DataFrame, temperatures:pd.DataFrame, cfg) -> None:
    
    # El nombre del archivo será el timestamp de las alarmas:
    path = cfg.api_tables_folder
    prefix = cfg.ct_id + '_' + cfg.camera_id + '_'
    filename = prefix + alarms.index.strftime('%Y-%m-%d_%H-%M-%S').values[0] + '.csv'
    # Generamos la tabla:
    alarms_table = pd.concat([temperatures, predicted, alarms])
    alarms_table = alarms_table.reset_index(drop=True)
    
    # IMPORTANTE: Enviamos alarma si no se ha enviado una alarma en la hora anterior.
    # Para ello vamos a mirar todas las tablas que tengamos en busca de una alarma 
    # en la misma región:
    # CÓDIGO PRODUCCIÓN:
    files = sorted(glob.glob(os.path.join(path, 'marked-*.csv')),reverse=True)
    # TEST EN LOCAL:
    # files = sorted(glob.glob(os.path.join(path, '*.csv')),reverse=True)
    
    # Nos quedamos con las que correspondan a la ultima hora mas o menos:
    files_1_hour = round(49 / cfg.acquisition_frequency)
    if len(files) > files_1_hour:
        files = files[:files_1_hour]
    
    for file in files:
        try:
            csv_table = pd.read_csv(file)
            for region in csv_table:
                # Si existe alguna alarma anterior en los archivos temporales
                # para esa región, no enviamos más alarmas.
                if csv_table[region][2] == 1.0:
                    alarms_table[region][2] = 0.0
                    alarms[region] = 0.0
        except Exception as e:
            print('No se han podido leer tablas temporales')
            
    # Forzamos una alarma siempre que la temperatura registrada por la cámara supere los 100 grados,
    # independientemente de si tenemos predicciones o no, de si se han enviado antes alarmas o no.
    for region in alarms_table:
        if alarms_table[region][0] > 100.0:
            alarms_table[region][2] = 1.0
    
    # Guardamos la nueva tabla:
    try:
        save_dfdata(filename, path, alarms_table, idx=False )
    except:
        print('Ha ocurrido un error y no se ha podido guardar la tabla de alarmas')
    
    # Para no consumir muchos recursos, eliminamos los archivos antiguos:
    # CÓDIGO PRODUCCIÓN:
    files = sorted(glob.glob(os.path.join(path, 'marked-*.csv')),reverse=True)
    # TEST EN LOCAL:
    # files = sorted(glob.glob(os.path.join(path, '*.csv')),reverse=True)
    
    if len(files) > 49:
        files_to_remove = files[49:]
        for file in files_to_remove:
            os.remove(file)
            
    # Devolvemos el nuevo vector de alarmas enviadas:        
    return alarms
        
    
