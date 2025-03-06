import glob
import os

import skimage
import numpy as np
from src.transformations.Model import Model


class InstanceSegmentation:
    def __init__(self):
        self.model = Model()

    def get_needed_padding(self, height, width, patch_height, patch_width):
        mod_height = height % patch_height  # Todo why modulo instead of difference?
        mod_width = width % patch_width
        pad_height = patch_height - mod_height if mod_height != 0 else 0
        pad_width = patch_width - mod_width if mod_width != 0 else 0
        return pad_height, pad_width

    def extract_patches(self, image, patch_size):
        patches = []

        # Calculate padding needed
        height, width = image.shape[:2]
        patch_height, patch_width = patch_size
        pad_height, pad_width = self.get_needed_padding(height, width, patch_height, patch_width)

        # Pad the image
        image = np.pad(image, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

        for y in range(0, height + pad_height, patch_height):
            for x in range(0, width + pad_width, patch_width):
                patch = image[y:y + patch_height, x:x + patch_width]
                patches.append(patch)

        return patches

    def merge_patches(self, patches, image_shape):
        height, width = image_shape[:2]

        patch_height, patch_width = patches[0].shape

        # Calculate padding needed
        pad_height = patch_height - height % patch_height # Todo: Duplicate code
        pad_width = patch_width - width % patch_width

        output_image = np.zeros((height + pad_height, width + pad_width), dtype=np.uint8)
        patch_height, patch_width = patches[0].shape[:2]

        patch_index = 0
        for y in range(0, height, patch_height):
            for x in range(0, width, patch_width):
                patch = patches[patch_index]
                output_image[y:y + patch_height, x:x + patch_width] = patch[:patch_height, :patch_width]
                patch_index += 1

        output_image = output_image[:height, :width]

        return output_image.astype(np.uint8)

    def window_segmentation(self, mask, patch_size=(320, 320)):
        patches = self.extract_patches(mask, patch_size)
        segmented_image = self.merge_patches(patches, mask.shape)
        return segmented_image

    def cleanup_segmentation_mask(self, raw_mask):
        # label image
        labeled_image, num = skimage.measure.label(raw_mask, return_num=True)

        # clear border
        cleared_border = skimage.segmentation.clear_border(labeled_image)

        # thresholding
        cleaned_image = cleared_border.copy()
        props = skimage.measure.regionprops(cleaned_image)
        area_threshold = 2000

        c_mask = np.zeros(cleaned_image.shape)
        rr, cc = skimage.draw.disk(
            (int(np.floor(cleaned_image.shape[0] / 2) + 52), int(np.floor(cleaned_image.shape[1] / 2) + 2)), 1450,
            shape=cleaned_image.shape)
        c_mask[rr, cc] = 1

        for prop in props:
            if prop.area < area_threshold:
                cleaned_image[cleaned_image == prop.label] = 0
            coords = prop.coords

            # If any of the coordinates of the region falls outside the circle, remove the region
            if np.any(c_mask[coords[:, 0], coords[:, 1]] == 0):
                cleaned_image[cleaned_image == prop.label] = 0

        return cleaned_image > 0

    def create_segmentation_masks(self, mask_file_names, result_path):
        for mask_file_name in mask_file_names:
            mask = np.load(mask_file_name)
            segmented_image = self.window_segmentation(mask)
            final_mask = self.cleanup_segmentation_mask(segmented_image)
            final_mask = final_mask * 255
            final_mask = final_mask.astype(np.uint8)
            result_filename = os.path.join(result_path, mask_file_name.split('/')[-1])
            np.save(str(result_filename), final_mask) #Todo currently breakes file instead of creating a new one

    def run(self, data_path, result_path):
        mask_file_names = self.get_mask_file_names(data_path)
        segmentation_masks = self.create_segmentation_masks(mask_file_names, result_path)
        return segmentation_masks

    def get_mask_file_names(self, data_path):
        mask_filenames_structure = f'{data_path}/*mask*.npy'
        mask_file_names = glob.glob(mask_filenames_structure)
        return mask_file_names
            