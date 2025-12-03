from modules.segment import reshape_img, define_otsu_masks, generate_template_image
from modules.data import get_last_data, save_npydata, save_dfdata
from modules.plots import generate_masks_plots
import config as cfg
import numpy as np

if __name__ == '__main__':

    # Cargamos las imágenes para generar las máscaras:
    images, _ = get_last_data(
        cfg.images_folder, cfg.segmentation_period, cfg.camera_filter)

    # Generamos una imagen sintética (media/máximos) para calcular las máscaras:
    template = generate_template_image(images, cfg.tmpl_mode)

    # Reconstruimos la imagen y guardamos:
    template = reshape_img(template, cfg.height, cfg.width)
    save_npydata(cfg.tmpl_image_fn, cfg.masks_folder, template)

    # Definimos las máscaras mediante el método de OTSU multinivel + MSER:
    masks = define_otsu_masks(template, cfg.otsu_levels, cfg.compute_mser)
    save_dfdata(cfg.masks_fn, cfg.masks_folder, masks, idx=False)

    # Generamos los PNGs:
    generate_masks_plots(masks, template, cfg.masks_folder)
