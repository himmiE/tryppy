import numpy as np
import skimage
from tryppy.transformations.model import Model


class MaskExtraction:
    def __init__(self, weights_path):
        self.model = None
        self.load_model(weights_path)

    def change_model(self, model_name="default"):
        self.model_name = model_name
        self.load_model()

    def get_masks(self, patches):
        image_ids = list(patches.keys())
        patch_data = [patches[k] for k in image_ids]

        patches_array = np.stack(patch_data)
        segmentation = self.model.predict(patches_array)
        result = {k: segmentation[i] for i, k in enumerate(image_ids)}
        return result

    def info(self):
        model_loaded = ""
        if self.model:
            model_loaded = "not yet "
        #files = len(list(pathlib.Path(self.data_path).glob('*.img')))

        print(f"You are currently aiming to use the {self.model_name} model.")
        print(f"The model has {model_loaded}been loaded.")
        #print(f"The specified path for your image data is: {self.data_path}")
        #print(f"The directory could {data_path_exists}be found. It contains {files} image files")
        return

    def load_model(self, weights_path):
        self.model = Model(weights_path)
        self.model.load_model()

    def get_needed_padding(self, height, width, patch_height, patch_width):
        mod_height = height % patch_height
        mod_width = width % patch_width
        pad_height = patch_height - mod_height if mod_height != 0 else 0
        pad_width = patch_width - mod_width if mod_width != 0 else 0
        return pad_height, pad_width

    def extract_patches(self, image, patch_size):
        patches = {}
        image = image[:, :, 0]

        # Calculate padding needed
        height, width = image.shape[:2]
        patch_height, patch_width = patch_size
        pad_height, pad_width = self.get_needed_padding(height, width, patch_height, patch_width)

        # Pad the image
        image = image.squeeze()
        image = np.pad(image, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

        for y in range(0, height + pad_height, patch_height):
            for x in range(0, width + pad_width, patch_width):
                patch_name = f"{y}_{x}"
                patch = image[y:y + patch_height, x:x + patch_width]
                patches[patch_name] = patch
        return patches

    def merge_patches(self, patches, image_shape):
        height, width = image_shape[:2]

        any_value = next(iter(patches.values()))
        patch_height, patch_width = any_value.shape
        pad_height, pad_width = self.get_needed_padding(height, width, patch_height, patch_width)

        output_image = np.zeros((height + pad_height, width + pad_width), dtype=np.uint8)
        any_value = next(iter(patches.values()))
        patch_height, patch_width = any_value.shape[:2]

        for y in range(0, height, patch_height):
            for x in range(0, width, patch_width):
                patch_name = f"{y}_{x}"
                patch = patches[patch_name]
                output_image[y:y + patch_height, x:x + patch_width] = patch[:patch_height, :patch_width]

        output_image = output_image[:height, :width]
        return output_image.astype(np.uint8)

    def window_segmentation(self, image, patch_size=(320, 320)):
        patches = self.extract_patches(image, patch_size)
        partial_masks = self.get_masks(patches)
        segmented_image = self.merge_patches(partial_masks, image.shape)
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

    def run(self, images):
        segmentation_masks = {}
        for image_id, image in images.items():
            segmented_image = self.window_segmentation(image)
            final_mask = self.cleanup_segmentation_mask(segmented_image)
            final_mask = final_mask * 255
            final_mask = final_mask.astype(np.uint8)
            segmentation_masks[image_id] = final_mask
        return segmentation_masks



