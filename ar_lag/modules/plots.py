import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def generate_segmented_template(template: np.ndarray, masks: pd.DataFrame) -> np.ndarray:
    # Digitalize masks for representation:
    w, h = np.shape(template)
    template = np.zeros_like(template.flatten())

    weight = 1
    for mask in masks:
        template += masks[mask] * weight
        weight += 1
    return np.reshape(template, [w, h])


def generate_masks_plots(masks: pd.DataFrame, template: np.ndarray, path: str) -> None:

    # Create folder if it do not exist:
    folder = os.path.join(path, 'png')
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Generate and save the template image:
    fig, ax1 = plt.subplots(1, 1)
    im = ax1.imshow(template, cmap='jet')
    ax1.set_title('Template')
    im.set_clim(min(template.flatten()), max(template.flatten()))
    cb = plt.colorbar(im)
    cb.set_label('Temperature (ºC)')
    fig.savefig(os.path.join(folder, 'template.png'))

    # Generate and save the segmented image:
    fig, ax1 = plt.subplots(1, 1)
    im = ax1.imshow(generate_segmented_template(template, masks), cmap='jet')
    ax1.set_title('Segmented template')
    fig.savefig(os.path.join(folder, 'segmented_image.png'))

    # Generate and save the masks individually:
    w, h = np.shape(template)
    for mask_name, mask in masks.items():
        if mask_name != 'index':
            # Plotting the mask.
            fig, ax1 = plt.subplots(1, 1)
            im = ax1.imshow(np.reshape(mask, [w, h]), cmap='jet')
            ax1.set_title(mask_name)
            fig.savefig(os.path.join(folder, mask_name + '.png'))


def plot_image(image: np.ndarray, timestamp: str, cmap: str='magma') -> None:
    """Esta función representa una sola imagen que le pasemos como argumento.

    Args:
        image (np.ndarray): Array que contiene la imagen.
        timestamp (str): Cadena con la fecha de captura de la imagen.
    """
    
    # Definimos la figura:
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    
    # Representamos la imagen:
    figure = ax.imshow(image, cmap=cmap)
    ax.axis('off')
    colorbar = fig.colorbar(figure, ax=ax,extend='both', shrink=0.85, location='bottom')
    colorbar.minorticks_on()
    colorbar.ax.set_title('Temperature (ºC)')
    
    # Ajustamos el layout y representamos:
    plt.title([timestamp])
    plt.tight_layout()
    plt.show()
    