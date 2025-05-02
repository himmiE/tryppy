import glob
import math
import os.path
import random

import matplotlib
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from scipy.integrate import quad, IntegrationWarning
import skimage
from scipy.signal import find_peaks
import warnings
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fsolve
from shapely import affinity
from shapely.geometry import Polygon, LineString
from skimage import measure, morphology
from skimage.draw import polygon
from skimage.measure import regionprops
from tqdm import tqdm

from src.file_handler import FileHandler
from spatial_efd import spatial_efd


class FeatureExtraction:
    def __init__(self, config, file_handler):
        self.file_handler = file_handler
        self.config = config
        self.image_size = config['tasks']['feature_extraction']['image_size']
        matplotlib.use('TkAgg')

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



    def plot(self, mask, contour=None, curvature=None, endpoints=None, midline=None,
             extended_midline=None, shape=None, grid=None, show=True, save=False, name=""):
        to_plot = self.config['tasks']['feature_extraction']['to_plot']
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        title =""

        ax[0].imshow(mask.T, cmap='gray', origin='upper')
        ax[0].set_title('Mask')
        ax[0].axis('off')

        labeled_mask = skimage.measure.label(mask)
        props = regionprops(labeled_mask)
        min_row, min_col, max_row, max_col = props[0].bbox

        if 'contour' in to_plot and contour is not None:

            #align with mask
            contour_x = contour[0] + min_col
            contour_y = contour[1] + min_row

            if 'curvature' in to_plot and curvature is None:
                title = "Contour"
                ax[1].plot(contour_x, contour_y, color='cyan', linewidth=2, label='Contour')
            else:
                title = "Curvature"
                ax[1].plot(contour_x, contour_y, color='cyan', linewidth=2)
                points = np.array([contour_x, contour_y]).T
                segments = [points[i:i + 2] for i in range(len(points) - 1)]
                lc = LineCollection(segments, cmap='viridis', linewidth=3, array=curvature,
                                    norm=plt.Normalize(vmin=np.min(curvature), vmax=np.max(curvature)), label='Curvature')
                ax[1].add_collection(lc)

        if 'endpoints' in to_plot and endpoints and contour:
            title = title + ", enpoints"
            start_idx, end_idx = endpoints
            x_start, y_start = contour_x[start_idx], contour_y[start_idx]
            x_end, y_end = contour_x[end_idx], contour_y[end_idx]
            ax[1].scatter([x_start, x_end], [y_start, y_end], color='red', marker='o', s=30, label='Endpoints')

        if 'midline' in to_plot and midline is not None:
            title = title + ", midline"
            if extended_midline:
                extended_x = extended_midline[1] + min_col
                extended_y = extended_midline[0] + min_row
                ax[1].plot(extended_x, extended_y, color='orange', linewidth=3, label='Extended-Midline')

            x, y = zip(*midline)
            x = np.array(x) + min_col
            y = np.array(y) + min_row
            ax[1].plot(x, y, color='red', linewidth=1, label='Midline')

        if 'shape' in to_plot and shape is not None:
            distances_matrix, midline_intersection_points, _, _, all_intersections, opposite_intersections = shape
            #plt.plot(new_points_x_ml, new_points_y_ml)
            plt.scatter(midline_intersection_points[:, 1]+min_col, midline_intersection_points[:, 0]+min_row,
                        color='red', label='Equidistant Points', s=5)
            #plt.quiver(x_new, y_new, -normal_x, -normal_y, color='r', angles='xy', scale_units='xy', label='Normals')
            for ix, iy in all_intersections:
                plt.plot(ix+min_col, iy+ min_row, 'bo', ms=4)
            for ix, iy in opposite_intersections:
                plt.plot(ix+min_col, iy+ min_row, 'go', ms=4)

        if 'grid' in to_plot and grid is not None:
            title = title + ", grid"
            for row in grid:
                for box in row:
                    polygon = matplotlib.patches.Polygon(box, closed=True, edgecolor='black', facecolor='none')
                    ax[1].add_patch(polygon)



        if any([contour is not None, endpoints is not None, midline is not None]):
           ax[1].legend(loc='lower right', fontsize='small')
        ax[1].set_title(title)
        ax[1].set_aspect('equal', 'box')
        plt.tight_layout()

        if save:
            self.file_handler.save_plot("plots", name, plt)

        if show:
            plt.show(block=True)
        else:
            plt.close(fig)
        return plt


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
            return None, None

        contour = best_contour
        coeffs = spatial_efd.CalculateEFD(contour[:, 0], contour[:, 1], harmonics=20)
        xt, yt = spatial_efd.inverse_transform(coeffs, harmonic=20, n_coords=10000)

        shifted_xt = xt - np.min(xt)
        shifted_yt = yt - np.min(yt)

        return shifted_xt, shifted_yt

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

    def find_endpoints(self, contour_x, contour_y, curvature, midline, threshold=False):
        head = None
        tail = None
        if threshold:
            head, tail = self.find_minima_with_threshold(contour_x, contour_y, curvature)
        if head is None or tail is None:
            head, tail = self.find_minima_with_midline(contour_x, contour_y, midline)
        return head, tail


    def extend_line_to_contour(self, start_point, direction, contour_x, contour_y, midline):
        contour_points = list(zip(contour_x, contour_y))
        poly = Polygon(contour_points)
        min_x, min_y, max_x, max_y = poly.bounds
        length = math.hypot(max_x - min_x, max_y - min_y)/10

        # Shapely erwartet (x, y)
        line = LineString([
            (start_point[0], start_point[1]),
            (start_point[0] + direction[0], start_point[1] + direction[1])
            #(start_point[1] + length * direction[1], start_point[0] + length * direction[0])
        ])
        long_line = affinity.scale(line, xfact=length, yfact=length, origin=tuple(start_point))

        intersection = long_line.intersection(poly.boundary)

        # Plot
        '''fig, ax = plt.subplots()
        x, y = poly.exterior.xy
        ax.plot(x, y, label='Polygon')

        x, y = zip(*midline)
        x = np.array(x)
        y = np.array(y)
        ax.plot(x, y, color='red', linewidth=1, label='Midline')

        x, y = long_line.xy
        ax.plot()
        ax.plot(x, y, color='orange', label='verlängerte Linie')

        if not intersection.is_empty:
            if intersection.geom_type == 'Point':
                ax.plot(*intersection.xy, 'ro', label='Schnittpunkt')
            elif intersection.geom_type == 'MultiPoint':
                for pt in intersection.geoms:
                    ax.plot(pt.x, pt.y, 'ro')

        ax.set_aspect('equal')
        ax.legend()
        plt.show()'''

        if intersection.is_empty:
            return None

        if intersection.geom_type == 'Point':
            return np.array([intersection.y, intersection.x])
        elif intersection.geom_type == 'MultiPoint':
            points = np.array([[p.y, p.x] for p in intersection.geoms])
            dists = np.linalg.norm(points - start_point, axis=1)
            return points[np.argmin(dists)]
        else:
            print("no intersection could be found between the linear midline extension and the contour")
            return None

    def find_minima_with_midline(self, xt, yt, midline, threshold=-20):
        midline = np.array(midline)
        start_point = midline[0]
        end_point = midline[-1]

        #print(f"midline-start: {start_point}, end: {end_point}")

        start_vec = start_point - midline[5]
        end_vec = end_point - midline[-6]

        #start_dir = start_vec / np.linalg.norm(start_vec)
        #end_dir = end_vec / np.linalg.norm(end_vec)

        extended_start = self.extend_line_to_contour(start_point, start_vec, xt, yt, midline)
        extended_end = self.extend_line_to_contour(end_point, end_vec, xt, yt, midline)

        curvature = self.calculate_curvature(xt, yt)
        minima_indices, _ = find_peaks(-curvature)
        minima_values = curvature[minima_indices]

        # Find minima below first threshold
        filtered_indices = minima_indices[minima_values <= threshold]
        filtered_values = minima_values[minima_values <= threshold]

        minima_coords = np.stack([yt[filtered_indices], xt[filtered_indices]], axis=1)

        dists_to_start = np.linalg.norm(minima_coords - extended_start, axis=1)
        dists_to_end = np.linalg.norm(minima_coords - extended_end, axis=1)

        alpha = 0.5  # weight of the curvature in relation to the distance

        score_start = dists_to_start + alpha * filtered_values
        score_end = dists_to_end + alpha * filtered_values

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
            print(len(xt))
            print(len(curvature))
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

    def mask_from_contour(self, contour_x, contour_y):
        # Koordinaten in int konvertieren
        rr, cc = polygon(contour_x, contour_y, (self.image_size, self.image_size))

        # Leere Maske und Füllen
        filled_mask = np.zeros((self.image_size, self.image_size), dtype=bool)
        filled_mask[rr, cc] = True
        return filled_mask

    def get_midline(self, contour_x, contour_y):
        filled_mask = self.mask_from_contour(contour_x, contour_y)
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
            fx_value = fx_der(t) # Wert von fx_der als Skalar extrahieren
            fy_value = fy_der(t)
            return float(np.hypot(fx_value, fy_value))
            #return float(np.sqrt(fx_der(t) ** 2 + fy_der(t) ** 2))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)
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
        cells = [[[vc[i][j], vc[i+1][j], vc[i+1][j+1], vc[i][j+1]]
                  for j in range(num_vertical_splits-1)] for i in range(len(vc)-1)]
        return cells

    def get_shape_vector(self, contour, extended_midline):
        #2xn featurevector with distance from boundaries on tangents of midline to the midline
        num_points = self.config['tasks']['feature_extraction']['num_points']

        smoothed_y_ml, smoothed_x_ml = extended_midline
        xt, yt = contour

        # Fit a function to the smoothed coordinates using UnivariateSpline
        spline_x_ml = UnivariateSpline(np.arange(smoothed_x_ml.shape[0]), smoothed_x_ml, s=5)
        spline_y_ml = UnivariateSpline(np.arange(smoothed_y_ml.shape[0]), smoothed_y_ml, s=5)
        #new_points_x_ml = spline_x_ml(np.linspace(0, len(smoothed_x_ml) - 1, 1000))
        #new_points_y_ml = spline_y_ml(np.linspace(0, len(smoothed_y_ml) - 1, 1000))

        total_length = self.arc_length(spline_x_ml, spline_y_ml, 0, len(smoothed_x_ml) - 1)
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
        midline_intersection_points = np.array([y_new, x_new]).T
        neighborhood_size = 10

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

            # contour
            '''plt.plot(xt, yt, color='blue', linewidth=2)
            plt.axis('equal')  # optional, um die Achsen gleichmäßig zu skalieren
            # midline-points
            plt.scatter(x_new, y_new, color='red', label='Equidistant Points', s=5)
            # tangents
            plt.quiver(x_new[i], y_new[i], avg_tangent_x, avg_tangent_y,
                       angles='xy', scale_units='xy', scale=0.1, color='red')
            plt.quiver(x_new[i], y_new[i], normal_x, normal_y,
                       angles='xy', scale_units='xy', scale=0.1, color='green')
            plt.show()'''

        # Convert list of normals to a numpy array for easier handling
        normal_x = np.array(normals_x)
        normal_y = np.array(normals_y)
        normals = np.array([normal_x, normal_y]).T

        # Compute the first intersection for each normal vector in both directions
        all_intersections, distances = self.compute_intersections_and_distances(x_new, y_new, normal_x, normal_y, xt,
                                                                                yt,
                                                                                1)
        opposite_intersections, opposite_distances = self.compute_intersections_and_distances(x_new, y_new, normal_x,
                                                                                              normal_y, xt, yt, -1)

        '''intersection_matrix = np.concatenate(
            (np.expand_dims(np.array(all_intersections), 0), np.expand_dims(np.array(opposite_intersections), 0)),
            0)'''
        distances_matrix = np.concatenate(
            (np.expand_dims(np.array(distances), 0), np.expand_dims(np.array(opposite_distances), 0)), 0)

        # Normalize distances
        max_distance = max(np.max(distances), np.max(opposite_distances))
        if max_distance > 0:
            distances_matrix = distances_matrix / max_distance

        return distances_matrix, midline_intersection_points, normals, max_distance, all_intersections, opposite_intersections

    def get_grid(self, shape_vector, num_vertical_splits=3):
        distances_matrix, midline_intersection_points, normals, max_distance, _, _ = shape_vector
        cells = self.create_coordinates(midline_intersection_points, normals, max_distance, num_vertical_splits)

        return cells

    def calculate_data_structures(self, images, features):
        contours = dict()
        curvatures = dict()
        midlines = dict()
        endpoints_s = dict()
        grids = dict()
        data_structures = {
            'contour': contours,
            'curvature': curvatures,
            'midline': midlines,
            'endpoints': endpoints_s,
            'grid': grids
        }

        datapoints_to_plot = []
        plot_features = self.config['tasks']['feature_extraction']['to_plot'] # TODO curvature without contour not possible
        save_plots = self.config['tasks']['feature_extraction']['save_plots']
        if plot_features and save_plots > 0:
            seed = self.config['seed']
            random.seed(seed)
            datapoints_to_plot = random.sample(list(images.keys()), min(save_plots, len(images.keys())))

        for name, image in tqdm(images.items()):  # TODO: calculate only what is desired
            contour = self.get_contour(image)
            contour_x, contour_y = contour

            #self.plot(image, show=True)

            if contour_x is None:
                ValueError("no contour was calculated")
                continue
            contours[name] = (contour_x, contour_y)

            curvature = self.calculate_curvature(contour_x, contour_y)
            if curvature is None:
                ValueError("no curvature was calculated")
                continue
            curvatures[name] = curvature

            midline = self.get_midline(contour_x, contour_y)
            if midline is None:
                ValueError("no midline was calculated")
                continue
            midlines[name] = midline

            endpoints = self.find_endpoints(contour_x, contour_y, curvature, midline)
            if endpoints is None:
                ValueError("no endpoints were calculated")
                continue
            endpoints_s[name] = endpoints

            endpoint_coords = ((contour_x[endpoints[0]], contour_y[endpoints[0]]),
                               (contour_x[endpoints[1]], contour_y[endpoints[1]]))

            extended_midline, new_points, total_length = self.extend_midline(midline, endpoint_coords)

            shape_vector_results = self.get_shape_vector(contour, extended_midline)
            shape_vector = shape_vector_results[0]

            px = self.config['tasks']['feature_extraction']['image_size']
            grid = self.get_grid(shape_vector_results)
            #midline_intersection_points, normals, max_distance
            if grid is None:
                ValueError("no grid was calculated")
                continue
            else:
                grids[name] = grid
            if name in datapoints_to_plot:
                self.plot(image, contour=(contour_x, contour_y), curvature=curvature, midline=midline, name=name,
                          endpoints=endpoints, extended_midline=extended_midline, shape=shape_vector_results,
                          save=True, show=self.config['tasks']['feature_extraction']['show_plots'])

        return data_structures

    def save_data_structures(self, structures_to_save, data_structures):
        for structure_name in structures_to_save:
            structure = data_structures[structure_name]
            folder_name = structure_name
            self.file_handler.save_numpy_data(folder_name, structure_name, structure)

    def run(self, images, save_raw_features=[]):
        # 1. first calculate all the needed data structures
        data_structures = self.calculate_data_structures(images, save_raw_features)

        # 1.2. save this data, where needed and make plots available
        self.save_data_structures(save_raw_features, data_structures)



        # 2. derive relevant features
        # 2. from these datastructures derive wanted features

    def extend_midline(self, midline, endpoints):

        start_point = np.array(endpoints[0])
        end_point = np.array(endpoints[-1])

        start_midline = np.array(midline[0])
        end_midline = np.array(midline[-1])

        option_1 = np.linalg.norm(start_point - start_midline) + np.linalg.norm(end_point - end_midline) # names alligned
        option_2 = np.linalg.norm(start_point - end_midline) + np.linalg.norm(end_point - start_midline) # names misalligned

        if option_1 <= option_2: #
            extended_coords_ml = np.insert(midline, 0, start_point, axis=0)
            extended_coords_ml = np.append(extended_coords_ml, [end_point], axis=0)
        else:
            extended_coords_ml = np.insert(midline, 0, end_point, axis=0)
            extended_coords_ml = np.append(extended_coords_ml, [start_point], axis=0)

        # Smooth the extended midline
        window_length = 11  # Must be an odd number, try different values
        polyorder = 3  # Try different values
        smoothed_x_ml = savgol_filter(extended_coords_ml[:, 1], window_length, polyorder)
        smoothed_y_ml = savgol_filter(extended_coords_ml[:, 0], window_length, polyorder)

        # Fit a function to the smoothed coordinates using UnivariateSpline
        spline_x_ml = UnivariateSpline(np.arange(extended_coords_ml.shape[0]), smoothed_x_ml, s=3)
        spline_y_ml = UnivariateSpline(np.arange(extended_coords_ml.shape[0]), smoothed_y_ml, s=3)
        new_points_x_ml = spline_x_ml(np.linspace(0, len(extended_coords_ml) - 1, 1000))
        new_points_y_ml = spline_y_ml(np.linspace(0, len(extended_coords_ml) - 1, 1000))

        total_length = self.arc_length(spline_x_ml, spline_y_ml, 0, len(extended_coords_ml) - 1)

        return (smoothed_x_ml, smoothed_y_ml), (new_points_x_ml, new_points_y_ml), total_length






