import numpy as np
import skimage
import skimage.io

class InstanceSegmentation:
    def __init__(self, fileHandler):
        self.fileHandler = fileHandler

    def run(self, images, masks={}):
        crops = {}
        mask_crops = {}

        if not masks:
            mask_folder_name = "mask"
            masks = self.fileHandler.get_input_files(input_folder_name=mask_folder_name)
        for image_id, image in images.items():
            crops[image_id], mask_crops[image_id] = self.crop_images_and_masks(image, masks[image_id], image_id)
        return crops, mask_crops

    def create_crops_from_mask(self, image, mask, min_threshold, max_threshold):
        labeled_mask = skimage.measure.label(mask)
        props = skimage.measure.regionprops(labeled_mask)

        coords = []  # Storage for cropped masks
        centroids = []
        padding = []
        for prop in props:
            if prop.area < min_threshold or prop.area > max_threshold:
                continue
            center_r, center_c = prop.centroid
            start_r = int(center_r - 320 // 2)
            start_c = int(center_c - 320 // 2)

            # Determine padding sizes
            pad_top = max(0, -start_r)
            pad_bottom = max(0, start_r + 320 - image.shape[0])
            pad_left = max(0, -start_c)
            pad_right = max(0, start_c + 320 - image.shape[1])

            # Update start_r and start_c to account for the padding
            start_r += pad_top
            start_c += pad_left

            padding.append((pad_top, pad_bottom, pad_left, pad_right))
            coords.append((start_r, start_c))  # Append the cropped mask
            centroids.append(prop.centroid)

        return padding, coords, centroids  # Return both cropped images and masks

    def crop_images_and_masks(self, image, mask, image_id):
        crops = {}
        mask_crops = {}

        padding, coords, centroids = self.create_crops_from_mask(image, mask, 100, 10000)

        # Save each cropped image and cropped mask to path_for_crops
        for j, (start_r, start_c) in enumerate(coords):
            pad_top, pad_bottom, pad_left, pad_right = padding[j]
            padded_image = image.copy()
            padded_mask = mask.copy()
            padded_image = np.pad(padded_image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
            padded_mask = np.pad(padded_mask, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
            cropped_image = padded_image[start_r:start_r + 320, start_c:start_c + 320, :]
            cropped_mask = padded_mask[start_r:start_r + 320, start_c:start_c + 320]  # Crop the mask
            cropped_mask = self.reduce_mask(cropped_mask)
            labeled_mask = skimage.measure.label(cropped_mask)
            cleared_border = skimage.segmentation.clear_border(labeled_mask)
            if cleared_border.max() == 0:
                continue
            name = f"{image_id}_{j}"
            mask_crops[name] = cleared_border
            crops[name] = cropped_image
        return crops, mask_crops

    def reduce_mask(self, mask):
        """
        This function converts a mask to a contour by focusing on the central region.
        Args:
            mask_path (str): The path to the mask image.
        Returns:
            ndarray: A reduced image where only the region of the center label is kept.
        """
        # Read the image from the mask path

        labeled_mask = skimage.measure.label(mask)

        # Find the center point of the image
        center_point = np.array([int(np.floor(mask.shape[0] / 2)), int(np.floor(mask.shape[1] / 2))])

        props = skimage.measure.regionprops(labeled_mask)

        label = 0
        min_distance = 5000000
        for prop in props:
            distance_to_center = np.min(np.sum((np.array(prop.centroid) - center_point) ** 2, axis=0))
            if distance_to_center < min_distance:
                min_distance = distance_to_center
                label = prop.label

        # Create a reduced image where only the region with the center label is kept
        reduced_image = labeled_mask == label
        return reduced_image





            