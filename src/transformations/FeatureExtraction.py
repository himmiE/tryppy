import glob
import math
import os.path

import numpy as np
import scipy
from scipy.integrate import quad
import skimage
from scipy.signal import find_peaks
import cv2 #opencv-python
import warnings

from spatial_efd import spatial_efd

from src.feature_extraction_visualizer import FeatureExtractionVisualizer


class FeatureExtraction:
    def __init__(self, path):
        self.path = path
        self.mask_images = self.get_src_images(path)
        pass

    def normalize_coordinates(self, xt, yt):
        lengths = np.sqrt(np.diff(xt) ** 2 + np.diff(yt) ** 2)
        total_length = np.sum(lengths)
        xt_normalized = xt / total_length
        yt_normalized = yt / total_length
        return xt_normalized, yt_normalized

    def get_window_from_list(self, list_len, index, window_size):
        half_window = window_size // 2
        rest = window_size % 2
        indices = range(index-half_window, index+half_window+rest)
        result = [i % list_len for i in indices]
        return result

    def get_outline_from_image(self, image_path):
        # Load image and get contour
        image = skimage.io.imread(image_path)
        image = skimage.morphology.area_closing(image, 10)

        contour = skimage.measure.find_contours(image, 0.8)[0]
        coeffs = spatial_efd.CalculateEFD(contour[:, 0], contour[:, 1], harmonics=20)
        xt, yt = spatial_efd.inverse_transform(coeffs, harmonic=20, n_coords=10000)

        return xt, yt


    def calculate_curvature(self, xt, yt, window_size=3, show=0):
        # Normalize the coordinates
        xt, yt = self.normalize_coordinates(xt, yt)

        # Ensure the coordinates are numpy arrays
        xt = np.array(xt)
        yt = np.array(yt)

        curvatures = []
        half_window = window_size // 2

        for i in range(len(xt)):
            # Define the window range
            x_window = xt[self.get_window_from_list(len(xt), i, window_size)]
            y_window = yt[self.get_window_from_list(len(yt), i, window_size)] #ToDO thorough testing!

            # Calculate first derivatives
            dx = np.gradient(x_window)
            dy = np.gradient(y_window)

            # Calculate second derivatives
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)

            # Calculate curvature at the central point of the window
            denominator = np.power((dx[half_window] ** 2 + dy[half_window] ** 2), 1.5) # magic nr?
            if denominator != 0:
                curvature = (dx[half_window] * ddy[half_window] - dy[half_window] * ddx[half_window]) / denominator
            else:
                curvature = 0
            curvatures.append(curvature)

        return np.array(curvatures)

    def find_relevant_minima(self, xt, yt, first_threshold=-20, second_threshold=-45, min_distance=4000, show=0):
        min_value_below_second_threshold = None
        smallest_remaining_index = None

        curvatures = self.calculate_curvature(xt, yt)
        minima_indices, _ = find_peaks(-curvatures)
        minima_values = curvatures[minima_indices]

        # Find minima below first threshold
        first_threshold_indices = minima_indices[minima_values <= first_threshold]
        first_threshold_values = minima_values[minima_values <= first_threshold]

        # Find minima below second threshold
        second_threshold_indices = first_threshold_indices[first_threshold_values <= second_threshold]
        second_threshold_values = first_threshold_values[first_threshold_values <= second_threshold]

        if len(second_threshold_values) > 0:
            min_value_below_second_threshold = second_threshold_indices[np.argmin(second_threshold_values)]

        def is_far_enough(idx1, idx2, threshold):
            distance_forward = (idx2 - idx1) % len(xt)
            distance_backward = (idx1 - idx2) % len(xt)
            smaller_distance = min(distance_backward, distance_forward)
            return smaller_distance >= threshold

        # Find other minima below first threshold that are some minimum distance away
        remaining_indices = [idx for idx in first_threshold_indices if
                             min_value_below_second_threshold is None
                             or is_far_enough(idx, min_value_below_second_threshold,min_distance)]
        remaining_values = curvatures[remaining_indices]

        if len(remaining_values) > 0:
            smallest_remaining_index = remaining_indices[np.argmin(remaining_values)]

        return min_value_below_second_threshold, smallest_remaining_index

    def get_midline(self, mask, prefilter_radius, min_length_pruning, show=0):
        pth_skeleton = self.mask_pruned_skeleton(mask, prefilter_radius,
                                            min_length_pruning)  # MAGIC NUMBERS: Radius for pre-filtering, length for pruning branches
        neighbours = scipy.ndimage.convolve(pth_skeleton, [[1, 1, 1], [1, 0, 1], [1, 1, 1]]) * pth_skeleton
        termini_count = np.count_nonzero(neighbours == 1)
        midline_count = np.count_nonzero(neighbours == 2) # not used?
        branches_count = np.count_nonzero(neighbours > 2)

        # trace, if a single line (two termini, zero branches)
        if termini_count == 2 and branches_count == 0:
            termini = neighbours.copy()
            termini[termini > 1] = 0
            termini_y, termini_x = skimage.morphology.local_maxima(termini, indices=True, allow_borders=False)
            # trace from index 0
            midline = [[termini_y[0], termini_x[0]]]
            v = pth_skeleton[midline[-1][0], midline[-1][1]]
            while v > 0:
                v = 0
                # mark visited pixels by setting to 0
                pth_skeleton[midline[-1][0], midline[-1][1]] = 0
                # for all neighbours...
                for a in range(-1, 2):  # a is delta in x
                    for b in range(-1, 2):  # b is delta in y
                        # if a skeleton pixel, step in that direction
                        if pth_skeleton[midline[-1][0] + b, midline[-1][1] + a] == 1:
                            midline.append([midline[-1][0] + b, midline[-1][1] + a])
                            v = pth_skeleton[midline[-1][0], midline[-1][1]]
                            # break inner loop on match
                            break
                    # break outer loop with inner
                    else:
                        continue
                    break
        else:
            return None

        return midline

    def mask_pruned_skeleton(self, mask, prefilter_radius, prune_length):
        skeleton = skimage.morphology.skeletonize(mask)
        skeleton = skeleton.astype(np.uint8)
        # make a neighbour count skeleton, 1 = terminus, 2 = arm, >2 = branch point
        neighbours = scipy.ndimage.convolve(skeleton, [[1, 1, 1], [1, 0, 1], [1, 1, 1]]) * skeleton
        # filter for 1 neighbour only, ie terminus image, and use to list termini
        termini = neighbours.copy()
        termini[termini > 1] = 0
        termini_y, termini_x = skimage.morphology.local_maxima(termini, indices=True, allow_borders=False)
        # prune skeleton
        for t in range(len(termini_x)):
            length = 0
            cx, cy = termini_x[t], termini_y[t]
            v = neighbours[cy, cx]
            while length < prune_length + 2 and v > 0 and v < 3:
                v = 0
                # mark visited pixels with 2, if removable (not a branch)
                if neighbours[cy, cx] < 3:
                    skeleton[cy, cx] = 2
                # for all neighbours...
                for a in range(-1, 2):
                    for b in range(-1, 2):
                        # if a skeleton pixel, step in that direction
                        if (a != 0 or b != 0) and skeleton[cy + b, cx + a] == 1:
                            length += 1
                            v = neighbours[cy, cx]
                            cy += b
                            cx += a
                            # break inner loop on match
                            break
                    # break outer loop with inner
                    else:
                        continue
                    break
            # if short enough then prune by replacing visited pixels (2) with 0
            if length < prune_length:
                skeleton[skeleton == 2] = 0
            else:
                skeleton[skeleton == 2] = 1
        # re-skeletonise, to handle messy branch points left over
        skeleton = skimage.morphology.medial_axis(skeleton, return_distance=False).astype(np.uint8)
        return skeleton

    # Correct arc length calculation
    def arc_length(self, fx, fy, a, b):
        fx_der = fx.derivative()
        fy_der = fy.derivative()
        length, _ = quad(lambda t: np.sqrt(fx_der(t) ** 2 + fy_der(t) ** 2), a, b)
        return length

    def find_point_on_boundary(self, boundary, point, distance):
        """
        Finds a point on the boundary that is at a specified distance from a given point on the boundary.
        If no point is found at the specified distance, find the point closest to that distance.

        :param boundary: A list of tuples representing the boundary points (x, y).
        :param point: A tuple representing the point on the boundary (x, y).
        :param distance: The distance from the given point to the target point on the boundary.
        :return: A tuple representing the point on the boundary (x, y) at or closest to the specified distance.
        """

        def euclidean_distance(p1, p2):
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        boundary_distances = [abs(euclidean_distance(point, boundary_point)-distance) for boundary_point in boundary]
        min_distance = min(boundary_distances)
        min_distance_index = boundary_distances.index(min_distance)
        closest_point = boundary[min_distance_index]
        if closest_point > 1e-6:
            warnings.warn("Closest point found on boundary is no exact match.")

        return closest_point  # Return the closest point if no exact match is found

    # Generic function to find the first intersection based on direction
    def find_first_intersection(self, normal_x, normal_y, x0, y0, boundary_x, boundary_y, direction):
        closest_intersection = None
        min_distance = float('inf')

        for i in range(len(boundary_x) - 1): # why not the last point?
            x1, y1 = boundary_x[i], boundary_y[i]
            x2, y2 = boundary_x[i + 1], boundary_y[i + 1]

            # Line equation for the boundary segment
            a1 = y2 - y1
            b1 = x1 - x2
            c1 = a1 * x1 + b1 * y1

            # Line equation for the normal
            a2 = normal_y
            b2 = -normal_x
            c2 = a2 * x0 + b2 * y0

            # Determinant
            det = a1 * b2 - a2 * b1

            if abs(det) < 1e-10:
                continue  # Lines are parallel, no intersection

            # Intersection coordinates
            ix = (b2 * c1 - b1 * c2) / det
            iy = (a1 * c2 - a2 * c1) / det

            # Check if intersection is within the boundary segment
            if min(x1, x2) <= ix <= max(x1, x2) and min(y1, y2) <= iy <= max(y1, y2):
                # Check the direction of the intersection with respect to the normal
                if direction * ((ix - x0) * normal_x + (iy - y0) * normal_y) > 0:
                    distance = np.sqrt((ix - x0) ** 2 + (iy - y0) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_intersection = (ix, iy)

        return closest_intersection

    # Function to compute intersections and distances
    def compute_intersections_and_distances(self, x_new, y_new, normal_x, normal_y, boundary_x, boundary_y, direction):
        intersections = []
        distances = []
        for i in range(len(x_new)):
            intersection = self.find_first_intersection(normal_x[i], normal_y[i], x_new[i], y_new[i], boundary_x, boundary_y,
                                                   direction)
            if intersection:
                intersections.append(intersection)
                distance = np.sqrt((intersection[0] - x_new[i]) ** 2 + (intersection[1] - y_new[i]) ** 2)
                distances.append(distance)
            else:
                intersections.append((float('NaN'), float('NaN')))
                distances.append(0.0)
        return intersections, distances

    def create_coordinates(self, midline_points, normals, max_distance, num_vertical_splits):
        vertical_distance = max_distance / num_vertical_splits
        vertical_coordinates_pos = []
        vertical_coordinates_neg = []

        for i, midline_point in enumerate(midline_points):
            normal = normals[i]
            curr_point = midline_point
            pos_coords = [curr_point]
            neg_coords = [curr_point]

            for j in range(num_vertical_splits):
                next_point_pos = curr_point + normal * vertical_distance
                pos_coords.append(next_point_pos)
                curr_point = next_point_pos

            curr_point = midline_point
            for j in range(num_vertical_splits):
                next_point_neg = curr_point - normal * vertical_distance
                neg_coords.append(next_point_neg)
                curr_point = next_point_neg

            vertical_coordinates_pos.append(pos_coords)
            vertical_coordinates_neg.append(neg_coords)

        # Combine the coordinates into cells
        cells = []
        for pos_coords, neg_coords in zip(vertical_coordinates_pos, vertical_coordinates_neg):
            for i in range(len(pos_coords) - 1):  # Assuming pos_coords and neg_coords are of the same length
                cell = [
                    pos_coords[i],
                    pos_coords[i + 1],
                    neg_coords[i + 1],
                    neg_coords[i]
                ]
                cells.append(cell)

        vertices = np.array(cells).reshape(-1, 2)

        # Debugging: Print out some of the cells
        print("Cells (First 5):", cells[:5])  # Print first 5 cells for inspection

        return vertices, vertical_coordinates_pos, vertical_coordinates_neg, cells

    def sum_pixel_intensities(self, image, vertices):
        intensity_sums = []

        for i in range(0, len(vertices), 3):
            polygon = np.array([[int(pt[0]), int(pt[1])] for pt in vertices[i:i + 3]], np.int32)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)
            intensity_sum = cv2.sumElems(image * (mask // 255))[0]
            intensity_sums.append(intensity_sum)

        return intensity_sums

    def get_src_images(self, source_image_directory):
        if os.path.isdir(source_image_directory):
            image_filename_structure = f'{source_image_directory}/*.png'
            all_files = glob.glob(image_filename_structure)
            return all_files
        else:
            # Todo Throw exception
            pass

    @classmethod
    def run(self, images):
        for image in images:
            outline = self.get_outline_from_image(image
                                                  )