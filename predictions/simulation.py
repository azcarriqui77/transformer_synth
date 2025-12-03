"""
Este script mueve las imágenes generadas en la carpeta de simulación a la carpeta de volúmenes
y llama a main.py para procesarlas una a una.
Con ello conseguimos simular la llegada de nuevas imágenes cada intervalo de tiempo y el 
procesamiento en tiempo real que ocurre en los servidores del proyecto RESISTO.
"""

import os
import shutil
import importlib
from modules.data import get_subfolders
from modules.alarms import get_temp_alarms
from modules.models import get_predictions
from modules.segment import get_temperatures
from modules.data import get_last_image, get_masks, append_timeseries


# Lista de carpetas diarias:
volumes_folder = '/Volumes/Crucial X9/resisto_synth/predictions/volumes/matrices'
simulation_folder = '/Volumes/Crucial X9/resisto_synth/synthetic_model/output/SYN_C/synth_plus_anom_images'

subfolders = get_subfolders(simulation_folder) 

# Ver las imágenes restantes y almacenar las rutas en una lista
for day_folder in subfolders:
    if day_folder.startswith('.'):
        continue
    else:
        print(f"Procesando carpeta: {day_folder}")
    images_list = sorted(os.listdir(day_folder))
    images_list = [file for file in images_list if file.endswith('.npy') and not file.startswith('.')]
    
    # Generamos la carpeta del nuevo día:
    day_folder_name = day_folder.split('/')[-1]
    new_day_folder = os.path.join(volumes_folder, day_folder_name)
    try:
        os.mkdir(new_day_folder)
    except OSError as e:
        pass

    for image in images_list:
        # Movemos la imagen
        print(f"Copiando la imagen {image}")
        origin = os.path.join(day_folder,image)
        destination = os.path.join(new_day_folder,image)
        shutil.copy(origin, destination)

        print("Iniciando proceso RESISTO")
        cfg = importlib.import_module('config')

        # Cargamos la última imagen disponible y las máscaras:
        image, timestamp = get_last_image(cfg.images_folder, cfg.camera_filter, cfg.images_folder_filter)
        image = image.flatten() # Las imágenes se leen como arrays bidimensionales, y el código las considera arrays unidimensionales
        masks = get_masks(cfg.masks_folder)
        print('Obtenida imagen:', timestamp)

        # Extraemos la temperatura media de cada región y almacenamos en el histórico:
        temperatures = get_temperatures(image, [timestamp], masks)
        append_timeseries(temperatures, cfg.temperatures_folder)
        print('Obtenidos valores de temperatura reales.')
        # Hasta aquí todo parece ir bien

        # Cargargamos las predicciones para este instante temporal:
        predicted = get_predictions(timestamp, temperatures, cfg.predictions)
        append_timeseries(predicted, cfg.predicted_folder)
        print('Obtenidos valores de temperatura predichos.')

        # Calculamos las alarmas en función de las predicciones
        alarms, F = get_temp_alarms(temperatures, predicted, cfg.predictions)
        append_timeseries(alarms, cfg.alarms_folder)
        append_timeseries(F, cfg.f_value_folder)
        print('Generadas las alarmas.')

        print("Terminado proceso")

        os.remove(destination)  # Borramos la imagen
        