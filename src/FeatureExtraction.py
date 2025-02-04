import math

import numpy as np
import scipy
from scipy.integrate import quad
import skimage
from optree.integration import numpy
from scipy.signal import find_peaks
import cv2 #opencv-python
from src.feature_extraction_visualizer import FeatureExtractionVisualizer


class FeatureExtraction:
    def __int__(self, image_size, source_images):
        self.images_size = image_size
        self.visualizer = FeatureExtractionVisualizer("")
        self.source_images = ""

    def normalize_coordinates(self, xt, yt):
        # Calculate the total length of the boundary
        lengths = np.sqrt(np.diff(xt) ** 2 + np.diff(yt) ** 2)
        total_length = np.sum(lengths)

        # Normalize coordinates
        xt_normalized = xt / total_length
        yt_normalized = yt / total_length

        return xt_normalized, yt_normalized

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
            if i < half_window:
                x_window = np.concatenate((xt[-(half_window - i):], xt[:i + half_window + 1]))
                y_window = np.concatenate((yt[-(half_window - i):], yt[:i + half_window + 1]))
            elif i >= len(xt) - half_window:
                x_window = np.concatenate((xt[i - half_window:], xt[:(half_window - (len(xt) - i - 1))]))
                y_window = np.concatenate((yt[i - half_window:], yt[:(half_window - (len(yt) - i - 1))]))
            else:
                x_window = xt[i - half_window:i + half_window + 1]
                y_window = yt[i - half_window:i + half_window + 1]

            # Calculate first derivatives
            dx = np.gradient(x_window)
            dy = np.gradient(y_window)

            # Calculate second derivatives
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)

            # Calculate curvature at the central point of the window
            denominator = np.power((dx[half_window] ** 2 + dy[half_window] ** 2), 1.5)
            if denominator != 0:
                curvature = (dx[half_window] * ddy[half_window] - dy[half_window] * ddx[half_window]) / denominator
            else:
                curvature = 0
            curvatures.append(curvature)

        return np.array(curvatures)

    def find_relevant_minima(self, xt, yt, show=0):
        curvatures = self.calculate_curvature(xt, yt)
        minima_indices, _ = find_peaks(-curvatures)
        minima_values = curvatures[minima_indices]

        # Filter minima above -20
        filtered_minima_indices = minima_indices[minima_values <= -20]
        filtered_minima_values = minima_values[minima_values <= -20]

        # Find minima below -50
        below_neg_50_indices = filtered_minima_indices[filtered_minima_values <= -45]
        below_neg_50_values = filtered_minima_values[filtered_minima_values <= -45]

        if len(below_neg_50_values) > 0:
            smallest_below_neg_50_index = below_neg_50_indices[np.argmin(below_neg_50_values)]
        else:
            smallest_below_neg_50_index = None

        def is_far_enough(idx1, idx2, threshold):
            distance_forward = (idx2 - idx1) % len(xt)
            distance_backward = (idx1 - idx2) % len(xt)
            return distance_forward >= threshold and distance_backward >= threshold

        # Find other minima below -20 that are at least 350 coordinates away
        remaining_indices = [idx for idx in filtered_minima_indices if
                             smallest_below_neg_50_index is None or is_far_enough(idx, smallest_below_neg_50_index,
                                                                                  4000)]
        remaining_values = curvatures[remaining_indices]

        if len(remaining_values) > 0:
            smallest_remaining_index = remaining_indices[np.argmin(remaining_values)]
        else:
            smallest_remaining_index = None

        return smallest_below_neg_50_index, smallest_remaining_index

    def get_midline(self, mask, prefilter_radius, min_length_pruning, show=0):
        pth_skeleton = self.mask_pruned_skeleton(mask, prefilter_radius,
                                            min_length_pruning)  # MAGIC NUMBERS: Radius for prefiltering, length for pruning branches
        neighbours = scipy.ndimage.convolve(pth_skeleton, [[1, 1, 1], [1, 0, 1], [1, 1, 1]]) * pth_skeleton
        termini_count = np.count_nonzero(neighbours == 1)
        midline_count = np.count_nonzero(neighbours == 2)
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
        # filter for 1 neigbour only, ie terminus image, and use to list termini
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
                # mark visited pixels with 2, if removeable (not a branch)
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
        # reskeletonise, to handle messy branch points left over
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

        closest_point = None
        closest_distance_diff = float('inf')

        for boundary_point in boundary:

            current_distance = euclidean_distance(point, boundary_point)
            distance_diff = abs(current_distance - distance)

            if distance_diff < 1e-6:  # Allowing a small tolerance
                return boundary_point

            if distance_diff < closest_distance_diff:
                closest_distance_diff = distance_diff
                closest_point = boundary_point

        return closest_point  # Return the closest point if no exact match is found

    # Generic function to find the first intersection based on direction
    def find_first_intersection(self, normal_x, normal_y, x0, y0, boundary_x, boundary_y, direction):
        closest_intersection = None
        min_distance = float('inf')

        for i in range(len(boundary_x) - 1):
            x1, y1 = boundary_x[i], boundary_y[i]
            x2, y2 = boundary_x[i + 1], boundary_y[i + 1]

            # Line equation for the boundary segment
            A1 = y2 - y1
            B1 = x1 - x2
            C1 = A1 * x1 + B1 * y1

            # Line equation for the normal
            A2 = normal_y
            B2 = -normal_x
            C2 = A2 * x0 + B2 * y0

            # Determinant
            det = A1 * B2 - A2 * B1

            if abs(det) < 1e-10:
                continue  # Lines are parallel, no intersection

            # Intersection coordinates
            ix = (B2 * C1 - B1 * C2) / det
            iy = (A1 * C2 - A2 * C1) / det

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