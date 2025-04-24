import glob
import math
import os.path

import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.integrate import quad
import skimage
from scipy.signal import find_peaks
import cv2 #opencv-python
import warnings
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fsolve
from shapely.geometry import Polygon


from spatial_efd import spatial_efd



class FeatureExtraction:
    def __init__(self):
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


    def get_contour(self, image):
        image = skimage.morphology.area_closing(image, 10)
        contours = skimage.measure.find_contours(image, 0.8)

        best_contour = None
        largest_area = 0

        for c in contours:
            try:
                poly = Polygon(c[:, ::-1])  # (y,x) → (x,y)
                if poly.is_valid and poly.area > largest_area:
                    largest_area = poly.area
                    best_contour = c
            except Exception as e:
                print(f"skipped contour: {e}")

        contour = best_contour
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
            y_window = yt[self.get_window_from_list(len(yt), i, window_size)]

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

        def integrand(t):
            return float(np.sqrt(fx_der(t) ** 2 + fy_der(t) ** 2))

        length = quad(integrand, a, b)[0]
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
        vertical_coordinates = []

        for i, midline_point in enumerate(midline_points):
            normal = normals[i]
            curr_point = midline_point
            pos_coords = [curr_point]
            neg_coords = []

            for j in range(num_vertical_splits):
                next_point_pos = curr_point + normal * vertical_distance
                pos_coords.append(next_point_pos)
                curr_point = next_point_pos

            curr_point = midline_point
            for j in range(num_vertical_splits):
                next_point_neg = curr_point - normal * vertical_distance
                neg_coords.append(next_point_neg)
                curr_point = next_point_neg

            vertical_coordinates.append(neg_coords[::-1] + pos_coords)

        vc = vertical_coordinates
        cells = [[vc[i][j], vc[i+1][j], vc[i][j+1], vc[i+1][j+1]] for j in range(num_vertical_splits-1) for i in range(len(vc)-1)]

        # Combine the coordinates into cells
        '''cells = []
        for pos_coords, neg_coords in zip(vertical_coordinates_pos, vertical_coordinates_neg):
            for i in range(len(pos_coords) - 1):  # Assuming pos_coords and neg_coords are of the same length
                cell = [
                    pos_coords[i],
                    pos_coords[i + 1],
                    neg_coords[i + 1],
                    neg_coords[i]
                ]
                cells.append(cell)'''

        vertices = np.array(cells).reshape(-1, 2)

        # Debugging: Print out some of the cells
        print("Cells (First 5):", cells[:5])  # Print first 5 cells for inspection

        return cells

    def get_grid(self, contour, endpoints, image_shape):

        xt, yt = contour
        distances = []

        grid_size = 320
        mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
        rr, cc = skimage.draw.polygon(yt, xt, image_shape)
        mask[rr, cc] = 1

        midline = self.get_midline(mask, 2, 50)
        if not midline:
            return None

        midline = np.array(midline)
        # midline = midline/scaling_factor
        coords_ml = midline.copy()
        start_point_ml = coords_ml[0]
        end_point_ml = coords_ml[-1]

        head, tail = endpoints
        #if not smallest_below_neg_50_index or not smallest_remaining_index:
        #    break
        smallest_below_neg_50 = (yt[head], xt[head])
        smallest_remaining = (yt[tail], xt[tail])

        # Calculate distances to start_point_ml
        dist_to_start_below_neg_50 = np.linalg.norm(smallest_below_neg_50 - start_point_ml)
        dist_to_start_remaining = np.linalg.norm(smallest_remaining - start_point_ml)
        distances.append(min(dist_to_start_below_neg_50, dist_to_start_remaining))

        # Determine which point is closer to start_point_ml
        if dist_to_start_below_neg_50 < dist_to_start_remaining:
            extended_coords_ml = np.insert(coords_ml, 0, smallest_below_neg_50, axis=0)
            extended_coords_ml = np.append(extended_coords_ml, [smallest_remaining], axis=0)
        else:
            extended_coords_ml = np.insert(coords_ml, 0, smallest_remaining, axis=0)
            extended_coords_ml = np.append(extended_coords_ml, [smallest_below_neg_50], axis=0)

        # Smooth the extended midline
        window_length = 5  # Must be an odd number, try different values
        polyorder = 3  # Try different values
        smoothed_x_ml = savgol_filter(extended_coords_ml[:, 1], window_length, polyorder)
        smoothed_y_ml = savgol_filter(extended_coords_ml[:, 0], window_length, polyorder)

        # Fit a function to the smoothed coordinates using UnivariateSpline
        spline_x_ml = UnivariateSpline(np.arange(extended_coords_ml.shape[0]), smoothed_x_ml, s=5)
        spline_y_ml = UnivariateSpline(np.arange(extended_coords_ml.shape[0]), smoothed_y_ml, s=5)
        new_points_x_ml = spline_x_ml(np.linspace(0, len(extended_coords_ml) - 1, 1000))
        new_points_y_ml = spline_y_ml(np.linspace(0, len(extended_coords_ml) - 1, 1000))

        total_length = self.arc_length(spline_x_ml, spline_y_ml, 0, len(extended_coords_ml) - 1)
        num_points = 50
        arc_lengths = np.linspace(0, total_length, num=num_points)

        # Find the t values that correspond to these equidistant arc lengths
        t_new = np.zeros(num_points)
        t_new[0] = 0

        for i in range(1, num_points):
            def objective(t):
                return self.arc_length(spline_x_ml, spline_y_ml, 0, t) - arc_lengths[i]

            t_new[i] = fsolve(objective, t_new[i - 1])[0]

        # Sample the splines at these t values
        x_new = spline_x_ml(t_new)
        y_new = spline_y_ml(t_new)

        neighborhood_size = 5

        normals_x = []
        normals_y = []

        for i in range(len(t_new)):
            # Make sure to handle boundaries correctly
            t_neighborhood = t_new[max(0, i - neighborhood_size):min(len(t_new), i + neighborhood_size + 1)]

            # Compute tangents in the neighborhood
            dx_neighborhood = spline_x_ml.derivative()(t_neighborhood)
            dy_neighborhood = spline_y_ml.derivative()(t_neighborhood)

            # Average the tangents
            avg_dx = np.mean(dx_neighborhood)
            avg_dy = np.mean(dy_neighborhood)

            # Normalize the averaged tangent vector
            avg_tangent_magnitude = np.sqrt(avg_dx ** 2 + avg_dy ** 2)
            avg_tangent_x = avg_dx / avg_tangent_magnitude
            avg_tangent_y = avg_dy / avg_tangent_magnitude

            # Compute the normal vector from the averaged tangent vector
            normal_x = -avg_tangent_y
            normal_y = avg_tangent_x

            normals_x.append(normal_x)
            normals_y.append(normal_y)

        # Convert list of normals to a numpy array for easier handling
        normal_x = np.array(normals_x)
        normal_y = np.array(normals_y)

        # Compute the first intersection for each normal vector in both directions
        all_intersections, distances = self.compute_intersections_and_distances(x_new, y_new, normal_x, normal_y, xt, yt,
                                                                           1)
        opposite_intersections, opposite_distances = self.compute_intersections_and_distances(x_new, y_new, normal_x,
                                                                                         normal_y, xt, yt, -1)

        intersection_matrix = np.concatenate(
            (np.expand_dims(np.array(all_intersections), 0), np.expand_dims(np.array(opposite_intersections), 0)),
            0)
        distances_matrix = np.concatenate(
            (np.expand_dims(np.array(distances), 0), np.expand_dims(np.array(opposite_distances), 0)), 0)

        # Normalize distances
        max_distance = max(np.max(distances), np.max(opposite_distances))
        if max_distance > 0:
            distances_matrix = distances_matrix / max_distance

        midline_points = np.array([x_new, y_new]).T  # Assuming x_new and y_new are 1D arrays of the same length
        normals = np.array([normal_x, normal_y]).T  # Assuming normal_x and normal_y are 1D arrays of the same length

        num_vertical_splits = 3  #ToDo adjust in config

        cells = self.create_coordinates(midline_points,normals, max_distance, num_vertical_splits)

        return cells

    def get_src_images(self, source_image_directory):
        if os.path.isdir(source_image_directory):
            image_filename_structure = f'{source_image_directory}/*.png'
            all_files = glob.glob(image_filename_structure)
            return all_files
        else:
            # Todo Throw exception
            pass

    def run(self, images):
        curvatures = dict()
        endpoints = dict()
        grids = dict()
        for name, image in images.items():
            contour_x, contour_y = self.get_contour(image)
            curvatures[name] = self.calculate_curvature(contour_x, contour_y)
            endpoints[name] = self.find_relevant_minima(contour_x, contour_y)
            grid = self.get_grid((contour_x, contour_y), endpoints[name], (320,320))
            if not grid:
                continue
            else:
                grids[name] = grid
