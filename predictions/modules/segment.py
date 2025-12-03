from skimage.filters import threshold_multiotsu
from skimage.segmentation import slic
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import cv2


def check_masks():
    # Comprobar que existen las máscaras:
    # Si no existen generarlas:
    # Si existen, comporobar cuando fué la ultima vez que se generaron
    # Si han pasado días suficientes, re-calcular
    pass


def define_otsu_masks(image: np.ndarray, levels: int, compute_mser: bool) -> pd.DataFrame:
    """Esta función aplica el algoritmo de OTSU multinivel a la imagen que se le pasa a su entrada.
    A continuación, para cada región definida por el algoritmo le aplicamos MSER para conseguir una 
    segmentación más precisa.

    Args:
        image (np.ndarray): Imagen de entrada a segmentar
        levels (int): Número de regiones OTSU a definir.
        compute_mser (bool): Flag para computar el algoritmo MSER.

    Returns:
        pd.DataFrame: DataFrame con cada una de las regiones de salida.
    """
    print('Defining multilevel OTSU masks...')

    # Definimos los umbrales:
    thresholds = threshold_multiotsu(image, levels)

    # Generamos las regiones:
    regions = np.digitize(image, bins=thresholds)

    masks = pd.DataFrame()

    # Generamos cada región OTSU:
    for region_idx in range(levels):
        region_name = 'region_' + str(region_idx + 1)
        mask = regions == region_idx
        masks[region_name] = mask.flatten()

    # Representamos los resultados:
    # import matplotlib.pyplot as plt  
    # fig_1 = plt.figure() 
    # plt.imshow(regions,cmap='rainbow')
    # plt.show()   
    # plt.axis('off')
    # fig_2 = plt.figure()
    # plt.hist(image.ravel(), bins=255, orientation='horizontal', color='black')
    # plt.ylabel('Temperature (°C)')
    # plt.xlabel('# of pixels')
    # for thresh in thresholds:
    #     plt.axhline(thresh, color='orange')
    # print('Done!')

    # Aplicamos MSER si es necesario:
    if compute_mser:
        masks = mser(masks, image.shape)

    return masks


def mser(otsu_regions: pd.DataFrame, size: tuple) -> pd.DataFrame:
    """Esta función computa el algoritmo MSER a cada una de las regiones que se le pasan a su entrada. 

    Args:
        otsu_regions (pd.DataFrame): DataFrame que contiene cada una de las regiones definidas con OTSU.
        size (tuple): Tamaño de las imágenes para hacer el resize neesario.

    Returns:
        pd.DataFrame: DataFrame con cada una de las regiones de salida.
    """
    print('Defining MSER regions...')

    masks = pd.DataFrame()
    mser_region_idx = 0
    final_img = np.full(size, 0) # Para hacer plots de la segmentación
    # Iteramos sobre cada una de las regiones definidas por OTSU
    for otsu_region in otsu_regions:
        otsu_region = otsu_regions[otsu_region].values
        otsu_region = otsu_region.astype(int)
        otsu_region = otsu_region.reshape(size)

        # Seleccionamos el número de clusters por region y el tamaño maximo y minimo de regiones:
        n_clusters = 1
        max_area = otsu_region.sum()//n_clusters
        min_area = otsu_region.sum()//10*n_clusters

        # Mostramos la mascara de la región:
        # plt.imshow(region)

        # Transformamos la imagen a 8 bit para trabajar con openCV
        otsu_region = np.array(otsu_region * 255, dtype=np.uint8)

        # Extraemos las regiones MSER:
        detector = cv2.MSER_create(
            min_area=min_area, max_area=max_area, delta=2, max_variation=0.25)
        mser_regions, _ = detector.detectRegions(otsu_region)

        # Generamos una máscara para cada region mser (importante: dentro de la otsu -> eliminar zonas negras)
        for mser_region in mser_regions:
            final = np.full(size, False)
            
            for pixel_list in mser_region:
                final[pixel_list[1]][pixel_list[0]] = True
                
                if sum(otsu_region[final]): # Para hacer plots de la segmentación
                    final_img[pixel_list[1]][pixel_list[0]] = mser_region_idx + 1
                
            if sum(otsu_region[final]):
                mser_region_idx += 1
                region_name = 'region_' + str(mser_region_idx)
                masks[region_name] = final.flatten()

    # import matplotlib.pyplot as plt    
    # plt.imshow(final_img,cmap='rainbow')
    # plt.show()
      
    print('Done!')
    return masks


def define_pca_masks(image: np.ndarray, levels: int) -> pd.DataFrame:
    assert image.ndim > 2
    model = PCA(n_components=levels)
    model.fit(image)

    masks = pd.DataFrame()

    # Generamos cada región:
    for region_idx in range(levels):
        region_name = 'region_' + str(region_idx + 1)
        # Este threshold podría ser relevante.
        mask = model.components_[region_idx] > 0.01
        masks[region_name] = mask.flatten()
    return masks


def define_slic_masks(image: np.ndarray, rois: dict) -> pd.DataFrame:
    # Computamos SLIC superpixels:
    super = slic(image, n_segments=35,
                 compactness=0.01, sigma=1, channel_axis=None)

    masks = pd.DataFrame()

    # Generamos las máscaras:
    for name in rois:
        masks[name] = generate_slic_mask(image=super, ids=rois[name]).flatten()
    return masks


def generate_slic_mask(image: np.ndarray, ids: list) -> np.ndarray:
    """This function generate a binary mask for a especific image and a pre-defined set of indexes.

    Args:
        image (np.ndarray): Bidimensional image containing the idx of each pixel.
        ids (list): List of idxs to generate the complete mask.

    Returns:
        np.ndarray: _description_
    """
    mask = np.zeros_like(image, dtype=bool)
    for id in ids:
        mask = np.logical_or(mask, image == id)
    return mask


def reshape_img(vector: np.ndarray, height: int, width: int) -> np.ndarray:
    """Esta función recostruye una imagen vectorizada y la devuelve como una matriz
    bidimensional de tamaño height x width

    Args:
        vector (np.ndarray): Imagen vectorizada
        height (int): Número de píxeles de alto
        width (int): Número de píxeles de ancho

    Returns:
        np.ndarray: Imagen reconstruida.
    """
    return np.reshape(vector, (height, width))


def get_temperatures(images: np.ndarray, timestamps: list, regions: pd.DataFrame) -> pd.DataFrame:
    """Esta función devuelve un DataFrame cuyo indice corresponde al timestamp 
    de adquisición de las imágenes y en cada columna tenemos la serie temporal de temperaturas
    para cada región. 

    Args:
        images (np.ndarray): Array bidimensional (o unidimnesional cuando se trata de una sola imágen).
        timestamps (list): Lista con los timestamps de adquisición de las imágenes.
        regions (pd.DataFrame): DataFrame con las máscaras a aplicar.

    Returns:
        pd.DataFrame: Cada una de las series temporales de temperatura.
    """
    temperatures = pd.DataFrame()
    temperatures['timestamp'] = pd.to_datetime(
        timestamps, format='%Y-%m-%d-%H-%M-%S')

    # Extraemos la temperatura media para cada región:
    for region in regions:
        mask = regions[region].values.astype(bool)
        if images.ndim > 1:
            for image in images:
                # Seleccionamos todos los pixeles:
                all_pixels = image[mask]
                # Seleccionamos solo el 5% de los píxeles más calientes:
                n_elements = int(len(image[mask]) * 0.05)
                top_percent = np.sort(image[mask])[-n_elements:]
                
                temperatures[region] = np.array([round(np.mean(all_pixels), 2)])
        else:
            # Seleccionamos todos los pixeles:
            all_pixels = images[mask]
            # Seleccionamos solo el 5% de los píxeles más calientes:
            n_elements = int(len(images[mask]) * 0.05)
            top_percent = np.sort(images[mask])[-n_elements:]
            
            temperatures[region] = np.array([round(np.mean(all_pixels), 2)])
    return temperatures.set_index('timestamp')


def get_region_size(regions: pd.DataFrame, timestamp: str) -> pd.DataFrame:
    """Esta función recibe como argumento un DataFrame con las máscaras para cada región
    y devueleve el tamaño (número de píxeles) de cada región.

    Args:
        masks (pd.DataFrame): DataFrame con las máscaras.

    Returns:
        pd.DataFrame: DataFrame con los tamaños.
    """
    sizes = pd.DataFrame()
    for region in regions:
        sizes[region] = [regions[region].values.sum()]
    sizes['timestamp'] = pd.to_datetime(
        [timestamp], format='%Y-%m-%d-%H-%M-%S')
    sizes.set_index('timestamp', inplace=True)
    return sizes


def generate_template_image(images: np.ndarray, mode: str) -> np.ndarray:
    """Esta función devuelve una imagen sintética que usaremos para definir 
    las máscaras para la segmentación. 

    Args:
        images (np.ndarray): Array bidimensional con todas las imágenes vectorizadas.
        mode (str): Modo de generación: media/máximos, pueden definirse otras alternativas.

    Returns:
        np.ndarray: Imágen sintética vectorizada.
    """
    if mode == 'mean':
        template = np.mean(images, axis=0)
    elif mode == 'max':
        template = images.max(axis=0)
    return template
