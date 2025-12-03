import os

prefix = 'SYN_C' # prefix for output directories and files

# %% Define paths and directories
ruta = '/Volumes/Crucial X9/resisto_synth/synthetic_model'
pathdata = os.path.join(ruta, 'volumes', 'matrices')          # path to input data matrices
pathtemplate = os.path.join(ruta, 'volumes', 'masks')  # path to segmentation masks
pathout = os.path.join(ruta, 'output', prefix)    # output path
images_path = os.path.join(pathout, 'synthetic_images')  # path to save synthetic images

generate_alarms = True
palarm = 0.1 # probability of alarm occurrence in a day
generate_images = True
image_size = (192, 256)  # image size (height, width)

# Parámetros simulación para main2
# opcion1 = False # True: temperatura ambiente constante a lo largo del día (temperatura media del día). False: temperatura ambiente variable a lo largo del día (fichero CSV)
# load_images = False # True: cargar imágenes para extraer temperaturas medias. False: cargar temperaturas medias desde un fichero CSV

# %% Parámetros termodinámicos del sistema térmico
# Ley de enfriamiento de Newton y de Stefan-Boltzmann
# Se asume en los clusters que el Area es un tercio de la masa. (comentario original)

# Comentarios sobre emisividades (valores de referencia)
# Metales        T [ºC]      emiss
# Aluminio       170         0,05
# Acero         -70...700    0,06...0,25
# Cobre          300..700    0,015...0,025
# Cobre oxidado  130         0,73

# Comentarios sobre calor específico (valores de referencia)
# Sustancia 	Calor específico (J/kg·K)
# Acero 	    460
# Aluminio 	    880
# Cobre 	    390
# Estaño 	    230
# Hierro 	    450
# Mercurio 	    138
# Oro 	        130
# Plata 	    235
# Plomo 	    130
# Sodio 	    1300

# Thermal model parameters
termo_params = {
    'm': 0.5,      # kg
    'A': 0.2, # Area in m^2   (12 cm^2 -> 12 / 100^2 m^2)
    'alpha': 0.1,           # coeficiente de recepción de calor para la cámara
    'sigma': 5.67e-8,       # W / (m^2 * K^4)  Stefan-Boltzmann
    'emiss': 0.25,          # emisividad
    'U': 25.0,              # W / (m^2 * K) Coeficiente de transferencia de calor (convección)
    'Cp': 460.0,     # J / (kg * K) Calor específico (0.5 * 1000 según original)
    'beta': 5.0           # W / % heater
}

