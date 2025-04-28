import glob
import math
import os.path
import sys

import networkx as nx
import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from networkx.classes import neighbors
from scipy.integrate import quad
import skimage
from scipy.signal import find_peaks
import cv2 #opencv-python
import warnings
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fsolve
from shapely.geometry import Polygon, LineString, Point
from scipy.optimize import minimize
from functools import partial
from skimage.draw import polygon
from skimage import measure, morphology

from spatial_efd import spatial_efd



class FeatureExtraction:
    def __init__(self, gridsize):
        self.gridsize = gridsize
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

    def plot(self, maske, contour=None, curvature=None, endpoints=None, midline=None, grid=None):
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        title =""

        ax[0].imshow(maske.T, cmap='gray', origin='upper')
        ax[0].set_title('Maske')
        ax[0].grid(True)
        ax[0].axis('off')
        if contour is not None:
            ax[1].grid(True)
            title = "Contour"
            if curvature is None:
                ax[1].plot(contour[0], contour[1], color='cyan', linewidth=2, label='Contour')
            else:
                ax[1].plot(contour[0], contour[1], color='cyan', linewidth=2)
                points = np.array([contour[0], contour[1]]).T
                segments = [points[i:i + 2] for i in range(len(points) - 1)]
                lc = LineCollection(segments, cmap='viridis', linewidth=3, array=curvature,
                                    norm=plt.Normalize(vmin=np.min(curvature), vmax=np.max(curvature)), label='Contour/Curvature')
                ax[1].add_collection(lc)

        if endpoints is not None:
            title = title + ", enpoints"
            start_idx, end_idx = endpoints
            x_start, y_start = contour[0][start_idx], contour[1][start_idx]
            x_end, y_end = contour[0][end_idx], contour[1][end_idx]
            ax[1].scatter([x_start, x_end], [y_start, y_end], color='red', marker='o', s=40, label='Endpoints')

        if midline is not None:
            title = title + ", midline"
            y, x = zip(*midline)
            ax[1].plot(y, x, color='orange', linewidth=2, label='Midline')

        if grid is not None:
            title = title + ", grid"

        ax[1].invert_yaxis()

        #if any([contour is not None, endpoints is not None, midline is not None]):
        #   ax[1].legend(loc='lower right', fontsize='small')
        ax[1].set_title(title)
        ax[1].axis('off')
        ax[1].set_aspect('equal', 'box')

        for ax in fig.get_axes():
            print(ax.get_position())
        plt.show()


    def get_contour(self, image):
        image = skimage.morphology.area_closing(image, 10)
        contours = skimage.measure.find_contours(image, 0.8)

        if not contours:
            print("No contours found, using 0.5 as area threshold")
            contours = skimage.measure.find_contours(image, 0.5)


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

        if best_contour is None:
            best_contour = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]) # TODO: find basic contour to return

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

    def find_endpoints(self, contour_x, contour_y, curvature, midline, threshold=True):
        head = None
        tail = None
        if threshold:
            head, tail = self.find_minima_with_threshold(contour_x, contour_y, curvature)
        if head is None or tail is None:
            head, tail = self.find_minima_with_midline(contour_x, contour_y, curvature, midline)
        return head, tail


    def extend_line_to_contour(self, start_point, direction, contour_x, contour_y, length=100):
        contour_points = list(zip(contour_x, contour_y))
        poly = Polygon(contour_points)

        # Shapely erwartet (x, y)
        line = LineString([
            (start_point[1], start_point[0]),
            (start_point[1] + length * direction[1], start_point[0] + length * direction[0])
        ])

        intersection = line.intersection(poly.boundary)

        if intersection.is_empty:
            return None

        if intersection.geom_type == 'Point':
            return np.array([intersection.y, intersection.x])
        elif intersection.geom_type == 'MultiPoint':
            points = np.array([[p.y, p.x] for p in intersection])
            dists = np.linalg.norm(points - start_point, axis=1)
            return points[np.argmin(dists)]
        else:
            print("no intersection could be found between the linear midline extension and the contour")
            return None

    def find_minima_with_midline(self, xt, yt, midline, threshold=-20):
        mask = np.zeros((self.gridsize, self.gridsize), dtype=np.uint8)

        midline = np.array(midline)
        start_point = midline[0]
        end_point = midline[-1]

        start_vec = start_point - midline[1]
        end_vec = end_point - midline[-2]

        start_dir = start_vec / np.linalg.norm(start_vec)
        end_dir = end_vec / np.linalg.norm(end_vec)

        extended_start = self.extend_line_to_contour(start_point, start_dir, xt, yt)
        extended_end = self.extend_line_to_contour(end_point, end_dir, xt, yt)

        curvature = self.calculate_curvature(xt, yt)
        minima_indices, _ = find_peaks(-curvature)
        minima_values = curvature[minima_indices]

        # Find minima below first threshold
        filtered_indices = minima_indices[minima_values <= threshold]
        filtered_values = minima_values[minima_values <= threshold]

        minima_coords = np.stack([yt[filtered_indices], xt[filtered_indices]], axis=1)

        dists_to_start = np.linalg.norm(minima_coords - extended_start, axis=1)
        dists_to_end = np.linalg.norm(minima_coords - extended_end, axis=1)

        alpha = 1.0  # weight of the curvature in relation to the distance

        score_start = dists_to_start - alpha * filtered_values
        score_end = dists_to_end - alpha * filtered_values

        best_start_idx = filtered_indices[np.argmin(score_start)]  # Index des besten Startpunkts
        best_end_idx = filtered_indices[np.argmin(score_end)]  # Index des besten Endpunkts

        return best_start_idx, best_end_idx

    def find_minima_with_threshold(self, xt, yt, curvature, first_threshold=-20, second_threshold=-45, show=0):
        min_value_below_second_threshold = None
        smallest_remaining_index = None

        min_distance = int(len(curvature)*0.4)
        minima_indices, _ = find_peaks(-curvature)
        minima_values = curvature[minima_indices]

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
        remaining_values = curvature[remaining_indices]

        if len(remaining_values) > 0:
            smallest_remaining_index = remaining_indices[np.argmin(remaining_values)]

        return min_value_below_second_threshold, smallest_remaining_index

    def fill_mask(selfself, mask, max_area):
        filled_mask = mask.copy()
        labeled_mask, num_objects = measure.label(filled_mask, connectivity=2, return_num=True)
        for region in measure.regionprops(labeled_mask):
            if region.area < max_area:
                coords = region.coords
                filled_mask[coords[:, 0], coords[:, 1]] = 1
        return filled_mask

    def get_midline(self, mask):
        filled_mask = morphology.remove_small_holes(mask, area_threshold=100)
        skeleton = skimage.morphology.skeletonize(filled_mask)
        midline = self.skeleton_to_midline(skeleton)
        return midline

    def skeleton_to_midline(self, skeleton):
        # skeleton to graph
        G = nx.Graph()
        rows, cols = np.where(skeleton)
        for y, x in zip(rows, cols):
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx_ = y + dy, x + dx
                    if (0 <= ny < skeleton.shape[0]) and (0 <= nx_ < skeleton.shape[1]):
                        if skeleton[ny, nx_]:
                            G.add_edge((y, x), (ny, nx_))

        #remove cycles
        T = nx.minimum_spanning_tree(G)
        # find longest path through skeleton
        nodes = list(T.nodes)
        longest_path = []
        max_length = 0
        for node in nodes:
            lengths = nx.single_source_dijkstra_path_length(T, node)
            farthest_node, length = max(lengths.items(), key=lambda x: x[1])
            if length > max_length:
                max_length = length
                longest_path = nx.dijkstra_path(T, node, farthest_node)
        return longest_path

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
        cells = [[[vc[i][j], vc[i+1][j], vc[i][j+1], vc[i+1][j+1]] for j in range(num_vertical_splits-1)] for i in range(len(vc)-1)]

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
        #print("Cells (First 5):", cells[:5])  # Print first 5 cells for inspection

        return cells

    def get_grid(self, contour, midline, endpoints, image_shape):

        xt, yt = contour
        distances = []

        grid_size = 320
        mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
        rr, cc = skimage.draw.polygon(yt, xt, image_shape)
        mask[rr, cc] = 1

        midline = np.array(midline)
        coords_ml = midline.copy()
        start_point_ml = coords_ml[0]
        end_point_ml = coords_ml[-1]

        head, tail = endpoints
        #if not smallest_below_neg_50_index or not smallest_remaining_index:
        #    break
        smallest_below_neg_50 = (yt[head], xt[head])
        smallest_remaining = (yt[tail], xt[tail])
        #print(xt)
        #print(yt)
        #print(head)
        #print(tail)
        #print(smallest_below_neg_50)
        # Calculate distances to start_point_ml
        dist_to_start_below_neg_50 = np.linalg.norm(smallest_below_neg_50 - start_point_ml)
        if isinstance(dist_to_start_below_neg_50, np.ndarray) and dist_to_start_below_neg_50.shape == (2, 1, 10000):
            print(dist_to_start_below_neg_50)
            pass
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
        print(extended_coords_ml[:, 1])
        print(extended_coords_ml[:, 0])

        if len(extended_coords_ml[:, 1]) < window_length:
            return None

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
            #self.plot(image)
            contour_x, contour_y = self.get_contour(image)

            if contour_x is None:
                ValueError("no contour was calculated")
                continue
            #self.plot(image, contour=(contour_x, contour_y))

            curvatures[name] = self.calculate_curvature(contour_x, contour_y)
            if curvatures[name] is None:
                ValueError("no curvature was calculated")
                continue
            #self.plot(image, contour=(contour_x, contour_y), curvature=curvatures[name])

            midline = self.get_midline(image)
            if midline is None:
                ValueError("no midline was calculated")
                continue
            #self.plot(image, contour=(contour_x, contour_y), curvature=curvatures[name], midline=midline)
            #save midline?

            endpoints[name] = self.find_endpoints(contour_x, contour_y, curvatures[name], midline)
            if endpoints[name] is None:
                ValueError("no endpoints were calculated")
                continue
            self.plot(image, contour=(contour_x, contour_y), curvature=curvatures[name], midline=midline, endpoints=endpoints[name])
            grid = self.get_grid((contour_x, contour_y), midline, endpoints[name], (320,320))

            if grid is None:
                ValueError("no grid was calculated")
                continue
            else:
                grids[name] = grid
