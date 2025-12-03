import os

# Información de los centro de distribución y las cámaras:
ct_id = 'SYN_C'
camera_id = ''
acquisition_frequency = 5
freq_dates = '5T'
max_deviation = 15
alarm_threshold = 0.1

# Tamaño de las imágenes:
height = 192
width = 256

# Rutas:
project_directory = '/Volumes/Crucial X9/resisto_synth/predictions'
images_folder = os.path.join(project_directory, 'volumes','matrices', camera_id)
images_folder_filter = ct_id + '*'
file_format = '.npy'
camera_filter = '*' + ct_id + '_' + camera_id + '*' + file_format

# Las series temporales deben almacenarse en una carpeta por cámara que tenga el CT:
timeserires_folder = os.path.join(project_directory, 'output', 'timeseries', ct_id)
api_tables_folder = os.path.join(project_directory, 'output', 'api_tables', ct_id)
masks_folder = os.path.join(project_directory, 'volumes', 'masks', 'CT49014')

temperatures_folder = os.path.join(timeserires_folder, 'temperatures')
alarms_folder = os.path.join(timeserires_folder, 'alarms')
alarms_sent_folder = os.path.join(timeserires_folder, 'alarms_sent')
regions_folder = os.path.join(timeserires_folder, 'region_size')
predictions_folder = os.path.join(timeserires_folder, 'predictions')
predicted_folder = os.path.join(timeserires_folder, 'predicted')
f_value_folder = os.path.join(timeserires_folder, 'f_value')

# CONFIGURACIÓN DE LOS MODELOS DE PREDICCIÓN:
lstm_model = {}
sarima_model = {}
ar_model = {
    'order': 50,    # Orden del modelo autoregresivo
    'prediction_length': 30,                # Ventana de predicción en minutos.
    'days_of_data': 4,                      # Dias de datos a cargar para entrenar.
    'freq': acquisition_frequency,          # Frecuencia de adquisición de imágenes en minutos
    'freq_dates': freq_dates                # Frecuencia muestreo temporal de date_range
}

# CONFIGURACIÓN DEL MODELO DE PREDICCIÓN Y ALARMAS
predictions = {
    'model': ar_model,
    'folder': predictions_folder,
    'temperatures_folder': temperatures_folder,
    'max_deviation': max_deviation,
    'alarm_threshold': alarm_threshold
}


# CONFIGURACIÓN DE LA SEGMENTACIÓN AUTOMÁTICA:
segmentation_period = 7         # Periodo en días para recalcular las máscaras.
otsu_levels = 5                 # Número de niveles de segmentación OTSU.
compute_mser = True             # Aplicamos MSER a la segmentacion.
tmpl_mode = 'mean'              # Forma de calcular la imagen plantilla (mean/max)
