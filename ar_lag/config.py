import os

# Información de los centro de distribución y las cámaras:
ct_id = 'SYN_B'
acquisition_frequency = 5
freq_dates = '5T'
max_deviation = 15
alarm_threshold = 0.1

# Tamaño de las imágenes:
height = 192
width = 256

# CONFIGURACIÓN DE LOS MODELOS DE PREDICCIÓN:
ar_model = {
    'order': [10, 25, 50, 100, 200, 288, 400],    # Orden del modelo autoregresivo
    'prediction_length': int(60*0.5),                # Ventana de predicción en minutos.
    'days_of_data': 4,                      # Dias de datos a cargar para entrenar.
    'freq': acquisition_frequency,          # Frecuencia de adquisición de imágenes en minutos
    'freq_dates': freq_dates                # Frecuencia muestreo temporal de date_range
}

# Rutas:
project_directory = '/Volumes/Crucial X9/resisto_synth/ar_lag'
images_folder = os.path.join(project_directory, 'volumes','matrices')
images_folder_filter = ct_id + '*'
file_format = '.npy'
camera_filter = '*' + ct_id + '*' + file_format

# Las series temporales deben almacenarse en una carpeta por cámara que tenga el CT:
pl = ar_model['prediction_length']
timeserires_folder = os.path.join(project_directory, 'output', 'timeseries', ct_id, f'pw_{pl}min')
masks_folder = os.path.join(project_directory, 'volumes', 'masks', 'CT49014')

temperatures_folder = os.path.join(timeserires_folder, 'temperatures')
alarms_folder = os.path.join(timeserires_folder, 'alarms')
predictions_folder = os.path.join(timeserires_folder, 'predictions')
predicted_folder = os.path.join(timeserires_folder, 'predicted')
f_value_folder = os.path.join(timeserires_folder, 'f_value')

# CONFIGURACIÓN DEL MODELO DE PREDICCIÓN Y ALARMAS
predictions = {
    'model': ar_model,
    'folder': predictions_folder,
    'temperatures_folder': temperatures_folder,
    'max_deviation': max_deviation,
    'alarm_threshold': alarm_threshold
}
