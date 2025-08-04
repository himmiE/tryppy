import math
import random
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy.integrate import quad, IntegrationWarning
import skimage
from scipy.signal import find_peaks
import warnings
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fsolve
from shapely import affinity, Point
from shapely.geometry import Polygon, LineString
from skimage.draw import polygon
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from rasterio.features import rasterize
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



    def plot(self, crop, mask, contour=None, curvature=None, endpoints=None, midline=None,
             extended_midline=None, shape=None, grid=None, show=True, save=False, name=""):
        to_plot = self.config['tasks']['feature_extraction']['to_plot']
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        title =""
        shifting = 0
        min_row = 0
        current_field = 0
        if "crop" in to_plot and crop is not None:
            ax[current_field].imshow(crop[:, :, 0], cmap='gray', origin='upper')
            ax[current_field].set_title('Phase')
            ax[current_field].set_aspect('equal')
            current_field += 1


        if "mask" in to_plot and mask is not None:
            ax[current_field].imshow(mask, cmap='gray', origin='upper')
            ax[current_field].set_title('Mask')


            ax[current_field].set_aspect('equal')
            current_field += 1
        shifting = self.get_offset(mask)
        if current_field < 2:
            if 'contour' in to_plot and contour is not None:

                #align with mask
                contour_x = contour[0] + shifting[0]
                contour_y = contour[1] + shifting[1]

                if 'curvature' in to_plot and curvature is None:
                    title = "Contour"
                    ax[current_field].plot(contour_x, contour_y, color='cyan', linewidth=2, label='Contour')
                else:
                    title = "Curvature"
                    ax[current_field].plot(contour_x, contour_y, color='cyan', linewidth=2)
                    points = np.array([contour_x, contour_y]).T
                    segments = [points[i:i + 2] for i in range(len(points) - 1)]
                    lc = LineCollection(segments, cmap='viridis', linewidth=3, array=curvature,
                                        norm=plt.Normalize(vmin=np.min(curvature), vmax=np.max(curvature)), label='Curvature')
                    ax[current_field].add_collection(lc)

            if 'endpoints' in to_plot and endpoints and contour:
                title = title + ", enpoints"
                start_idx, end_idx = endpoints
                x_start, y_start = contour_x[start_idx], contour_y[start_idx]
                x_end, y_end = contour_x[end_idx], contour_y[end_idx]
                ax[current_field].scatter([x_start, x_end], [y_start, y_end], color='red', marker='o', s=30, label='Endpoints')

            if 'midline' in to_plot and midline is not None:
                title = title + ", midline"
                if extended_midline:
                    extended_x = extended_midline[1] + shifting[0]
                    extended_y = extended_midline[0] + shifting[1]
                    ax[current_field].plot(extended_x, extended_y, color='orange', linewidth=3, label='Extended-Midline')

                x, y = zip(*midline)
                x = np.array(x) + shifting[0]
                y = np.array(y) + shifting[1]
                ax[current_field].plot(x, y, color='red', linewidth=1, label='Midline')

            if 'shape' in to_plot and shape is not None:
                title = title + ", shape"
                distances_matrix, midline_intersection_points, normals, max_distance, all_intersections, opposite_intersections = shape
                # Plot der Normalen mit skalierter Länge je nach Abstand
                for i in range(len(normals)):
                    midline_x = midline_intersection_points[:, 1] + shifting[0]
                    midline_y = midline_intersection_points[:, 0] + shifting[1]
                    x0, y0 = midline_x[i], midline_y[i]
                    nx, ny = normals[i]

                    # Holen der Distanz aus distances_matrix (positive Richtung)
                    distance = distances_matrix[0, i]

                    # Normalenlänge skalieren
                    # Skalierung: max_distance als max. Länge der Normalen (optional anpassen)
                    normal_length = distance * max_distance  # Hier mit max_distance multiplizieren, um die Länge zu justieren

                    # Normale zeichnen, skalierte Länge
                    ax[current_field].plot([x0, x0 + nx * normal_length], [y0, y0 + ny * normal_length], 'b--', alpha=0.5)

                # Plot der Schnittpunkte in der positiven Richtung (grün)
                all_intersections_x = [p[0] + shifting[0] for p in all_intersections if not np.isnan(p[0])]
                all_intersections_y = [p[1] + shifting[1] for p in all_intersections if not np.isnan(p[1])]
                ax[current_field].scatter(all_intersections_x, all_intersections_y, color='g', label="Schnittpunkte (positiv)", zorder=5)

                # Plot der Schnittpunkte in der negativen Richtung (orange)
                opposite_intersections_x = [p[0] + shifting[0] for p in opposite_intersections if not np.isnan(p[0])]
                opposite_intersections_y = [p[1] + shifting[1] for p in opposite_intersections if not np.isnan(p[1])]
                ax[current_field].scatter(opposite_intersections_x, opposite_intersections_y, color='orange',
                            label="Schnittpunkte (negativ)", zorder=5)

            if 'grid' in to_plot and grid is not None:
                title = title + ", grid"

                # Vierecke plotten
                self.plot_quadrilaterals(ax[current_field], grid, shifting)

        ax[current_field].set_title(title)
        ax[current_field].set_aspect('equal')
        plt.tight_layout()

        if any([contour is not None, endpoints is not None, midline is not None]):
            ax[current_field].legend(loc='lower right', fontsize='small')

        if save:
            self.file_handler.save_plot("plots", name, plt)

        if show:
            plt.show(block=True)
        else:
            plt.close(fig)
        return plt

    def plot_quadrilaterals(self, ax, quadrilaterals, shifting):
        dx, dy = shifting
        for quad in quadrilaterals:
            shifted = np.array(quad) + np.array([dx, dy])
            patch = matplotlib.patches.Polygon(shifted, closed=True, facecolor='cyan', edgecolor='black', alpha=0.3)
            ax.add_patch(patch)

    def sort_corners_clockwise(self, pts):
        pts = np.array(pts)
        centroid = np.mean(pts, axis=0)  # Schwerpunkt
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        sorted_indices = np.argsort(angles)
        return pts[sorted_indices]

    def get_contour(self, image):
        image = skimage.morphology.area_closing(image, 10)
        contours = skimage.measure.find_contours(image, 0.8)

        if not contours:
            #print("No contours found, using 0.5 as area threshold")
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

        contour = best_contour
        coeffs = spatial_efd.CalculateEFD(contour[:, 0], contour[:, 1], harmonics=20)
        xt, yt = spatial_efd.inverse_transform(coeffs, harmonic=20, n_coords=10000)

        shifted_xt = xt - np.min(xt)
        shifted_yt = yt - np.min(yt)

        poly = Polygon(best_contour[:, ::-1])
        area = poly.area
        perimeter = poly.length

        return (shifted_yt, shifted_xt), area, perimeter

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

    def find_endpoints(self, contour_x, contour_y, curvature, midline):
        head, tail = self.find_minima_with_midline(contour_x, contour_y, curvature, midline)
        return head, tail

    def extend_line_to_contour(self, start_point, direction, contour_x, contour_y):
        contour_points = list(zip(contour_x, contour_y))
        poly = Polygon(contour_points)
        min_x, min_y, max_x, max_y = poly.bounds
        length = math.hypot(max_x - min_x, max_y - min_y)/10

        # Shapely erwartet (x, y)
        line = LineString([
            (start_point[0], start_point[1]),
            (start_point[0] + direction[0], start_point[1] + direction[1])
        ])
        long_line = affinity.scale(line, xfact=length, yfact=length, origin=tuple(start_point))

        intersection = long_line.intersection(poly.boundary)

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

    def find_minima_with_midline(self, xt, yt, curvature, midline, threshold=+20):
        midline = np.array(midline)
        start_point = midline[0]
        end_point = midline[-1]

        start_vec = start_point - midline[5]
        end_vec = end_point - midline[-6]

        extended_start = self.extend_line_to_contour(start_point, start_vec, xt, yt)
        extended_end = self.extend_line_to_contour(end_point, end_vec, xt, yt)

        minima_indices, _ = find_peaks(curvature)
        minima_values = curvature[minima_indices]

        # Find minima below first threshold
        filtered_indices = minima_indices[minima_values >= threshold]
        filtered_values = minima_values[minima_values >= threshold]

        minima_coords = np.stack([yt[filtered_indices], xt[filtered_indices]], axis=1)

        dists_to_start = np.linalg.norm(minima_coords - extended_start, axis=1)
        dists_to_end = np.linalg.norm(minima_coords - extended_end, axis=1)

        alpha = -0.02  # weight of the curvature in relation to the distance

        score_start = dists_to_start + alpha * filtered_values
        score_end = dists_to_end + alpha * filtered_values

        best_start_idx = filtered_indices[np.argmin(score_start)]  # Index des besten Startpunkts
        best_end_idx = filtered_indices[np.argmin(score_end)]  # Index des besten Endpunkts

        return best_start_idx, best_end_idx

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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)
            length = quad(integrand, a, b)[0]
        return length

    '''def find_point_on_boundary(self, boundary, point, distance):
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

        return closest_point  # Return the closest point if no exact match is found'''

    # Generic function to find the first intersection based on direction
    def find_first_intersection(self, normal_x, normal_y, x0, y0, boundary_x, boundary_y, direction, ray_length=1000):
        dx = direction * normal_x
        dy = direction * normal_y
        start = (x0, y0)
        end = (x0 + dx * ray_length, y0 + dy * ray_length)
        ray = LineString([start, end])
        boundary = LineString(zip(boundary_x, boundary_y))
        inter = ray.intersection(boundary)

        if inter.is_empty:
            return None

        if inter.geom_type == 'MultiPoint':
            points = list(inter.geoms)
            points.sort(key=lambda pt: Point(start).distance(pt))
            closest = points[0]
            return closest.x, closest.y

        elif inter.geom_type == 'Point':
            return inter.x, inter.y
        return None

    # Function to compute intersections and distances
    def compute_intersections_and_distances(self, x_new, y_new, normal_x, normal_y, boundary_x, boundary_y, direction):
        intersections = []
        distances = []

        for i in range(len(x_new)):
            dx = direction * normal_x[i]
            dy = direction * normal_y[i]
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
            curr_point = midline_point[::-1]
            pos_coords = [curr_point]
            neg_coords = []

            for j in range(num_vertical_splits):
                next_point_pos = curr_point + normal * vertical_distance
                pos_coords.append(next_point_pos)
                curr_point = next_point_pos

            curr_point = midline_point[::-1]
            for j in range(num_vertical_splits):
                next_point_neg = curr_point - normal * vertical_distance
                neg_coords.append(next_point_neg)
                curr_point = next_point_neg

            vertical_coordinates.append(neg_coords + pos_coords)
        vc = vertical_coordinates
        cells = [self.sort_corners_clockwise([vc[i][j], vc[i + 1][j], vc[i + 1][j + 1], vc[i][j + 1]])
                 for j in range(num_vertical_splits * 2) for i in range(len(vc) - 1)]
        return cells

    def get_shape_vector(self, contour, extended_midline):
        #2xn featurevector with distance from boundaries on tangents of midline to the midline
        num_points = self.config['tasks']['feature_extraction']['num_points']

        smoothed_y_ml, smoothed_x_ml = extended_midline
        xt, yt = contour

        # Fit a function to the smoothed coordinates using UnivariateSpline
        spline_x_ml = UnivariateSpline(np.arange(smoothed_x_ml.shape[0]), smoothed_x_ml, s=5)
        spline_y_ml = UnivariateSpline(np.arange(smoothed_y_ml.shape[0]), smoothed_y_ml, s=5)

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
        neighborhood_size = 5

        normals_x = []
        normals_y = []

        tangents_x = []
        tangents_y = []

        for i in range(len(t_new)):
            # Make sure to handle boundaries correctly
            t_neighborhood = t_new[max(0, i - neighborhood_size):min(len(t_new), i + neighborhood_size + 1)]

            # Compute tangents in the neighborhood
            dx_neighborhood = spline_x_ml.derivative()(t_neighborhood)
            dy_neighborhood = spline_y_ml.derivative()(t_neighborhood)

            # Average the tangents
            avg_dx = np.mean(dx_neighborhood)
            avg_dy = np.mean(dy_neighborhood)

            tangents_x.append(avg_dx)
            tangents_y.append(avg_dy)

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
        normals = np.array([normal_x, normal_y]).T

        # Compute the first intersection for each normal vector in both directions
        all_intersections, distances = self.compute_intersections_and_distances(x_new, y_new, normal_x, normal_y, xt,
                                                                                yt, 1)
        opposite_intersections, opposite_distances = self.compute_intersections_and_distances(x_new, y_new, normal_x,
                                                                                              normal_y, xt, yt, -1)

        distances_matrix = np.concatenate(
            (np.expand_dims(np.array(distances), 0), np.expand_dims(np.array(opposite_distances), 0)), 0)

        # Normalize distances
        max_distance = max(np.max(distances), np.max(opposite_distances))
        if max_distance > 0:
            distances_matrix = distances_matrix / max_distance

        return distances_matrix, midline_intersection_points, normals, max_distance, all_intersections, opposite_intersections

    def get_grid(self, shape_vector, image_crop, shifting, num_vertical_splits=3):
        distances_matrix, midline_intersection_points, normals, max_distance, _, _ = shape_vector
        cells = self.create_coordinates(midline_intersection_points, normals, max_distance, num_vertical_splits)
        intensities = self.batch_weighted_intensities(image_crop[:, :, 2], cells, shifting)

        return cells, intensities

    def calculate_data_structures(self, crops, masks, features):
        contours = dict()
        curvatures = dict()
        midlines = dict()
        endpoints_s = dict()
        shapes = dict()
        grids = dict()
        data_structures = {
            'contour': contours,
            'curvature': curvatures,
            'midline': midlines,
            'endpoints': endpoints_s,
            'shape': shapes,
            'grid': grids
        }

        if not masks:
            masks = self.file_handler.get_input_files(input_folder_name="mask", extension=".npy")

        plot_features = self.config['tasks']['feature_extraction']['to_plot']
        save_plots = self.config['tasks']['feature_extraction']['save_plots']
        if plot_features and save_plots > 0:
            seed = self.config['seed']
            random.seed(seed)

        features = []
        for orig_id, mask_dict in masks.items():
            contours[orig_id] = {}
            curvatures[orig_id] = {}
            midlines[orig_id] = {}
            endpoints_s[orig_id] = {}
            shapes[orig_id] = {}
            grids[orig_id] = {}

            for crop_id, mask in mask_dict.items():
                temp_features = {}
                shifting = self.get_offset(mask)

                crop = crops[orig_id][crop_id]

                contour, area, perimeter = self.get_contour(mask)
                temp_features['crop_id'] = crop_id
                temp_features['orig_image'] = orig_id
                temp_features['area'] = area
                temp_features['perimeter'] = perimeter
                contour_x, contour_y = contour

                if contour_x is None:
                    ValueError("no contour was calculated")
                    continue
                contours[orig_id][crop_id] = (contour_x, contour_y)

                curvature = self.calculate_curvature(contour_x, contour_y)
                if curvature is None:
                    ValueError("no curvature was calculated")
                    continue
                curvatures[orig_id][crop_id] = curvature

                midline = self.get_midline(contour_x, contour_y)
                if midline is None:
                    ValueError("no midline was calculated")
                    continue
                midlines[orig_id][crop_id] = midline


                endpoints = self.find_endpoints(contour_x, contour_y, curvature, midline)
                if endpoints is None:
                    ValueError("no endpoints were calculated")
                    continue
                endpoints_s[orig_id][crop_id] = endpoints

                endpoint_coords = ((contour_x[endpoints[0]], contour_y[endpoints[0]]),
                                   (contour_x[endpoints[1]], contour_y[endpoints[1]]))

                extended_midline, new_points, total_length = self.extend_midline(midline, endpoint_coords)
                temp_features['length'] = total_length

                shape_vector_results = self.get_shape_vector(contour, extended_midline)
                shape_vector = shape_vector_results[0]
                shapes[orig_id][crop_id] = shape_vector
                temp_features['shape'] = shape_vector

                cells, intensities = self.get_grid(shape_vector_results, crop, shifting)
                temp_features['intensities'] = intensities

                features.append(temp_features)

                if cells is None:
                    ValueError("no grid was calculated")
                    continue
                else:
                    grids[orig_id][crop_id] = intensities
                self.plot(None, mask, contour=(contour_x, contour_y), curvature=curvature, midline=midline, name=crop_id,
                          endpoints=endpoints, extended_midline=extended_midline, shape=shape_vector_results,
                          grid=cells,
                          save=True, show=self.config['tasks']['feature_extraction']['show_plots'])

        return data_structures, features

    def save_data_structures(self, structures_to_save, data_structures):
        for structure_name in structures_to_save:
            structure = data_structures[structure_name]
            folder_name = structure_name
            self.file_handler.save_numpy_data(folder_name, structure_name, structure)

    def run(self, crops, masks, save_raw_features=[]):
        # 1. first calculate all the needed data structures and make plots available
        data_structures, features = self.calculate_data_structures(crops, masks, save_raw_features)

        # 1.2. save this data, where needed
        self.save_data_structures(save_raw_features, data_structures)

        # 2. derive relevant features
        df = self.make_df(features)
        self.file_handler.save_df(df)

        return df

    def extend_midline(self, midline, endpoints):

        start_point = np.array(endpoints[0])
        end_point = np.array(endpoints[-1])

        start_midline = np.array(midline[0])
        end_midline = np.array(midline[-1])

        option_1 = np.linalg.norm(start_point - start_midline) + np.linalg.norm(end_point - end_midline)
        option_2 = np.linalg.norm(start_point - end_midline) + np.linalg.norm(end_point - start_midline)

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

    def batch_weighted_intensities(self, image, polygons, shifting, shape=(320, 320)):
        """
        image: 2D ndarray (grayscale)
        polygons: List of polygons (each as list of (x, y))
        shape: Shape of the image (height, width), if different from image.shape
        """
        image = image.astype(np.float32) / 65535.0
        shiftes_polygons = []
        for quad in polygons:
            shiftes_polygons.append(np.array(quad) + np.array([shifting[0], shifting[1]]))

        intensities = []
        for poly_coords in shiftes_polygons:
            poly = Polygon(poly_coords)
            mask = rasterize(
                [(poly, 1)],
                out_shape=shape,
                fill=0,
                dtype='float32',
                all_touched=True  # oder False für präziser an den Pixelgrenzen
            )
            masked_values = image[mask.astype(bool)]
            intensities.append(masked_values.mean())

        return np.array(intensities)

    def get_offset(self, mask):
        labeled_mask = skimage.measure.label(mask)
        props = regionprops(labeled_mask)
        min_row, min_col, max_row, max_col = props[0].bbox
        return min_col, min_row

    def make_df(self, features):
        # Liste für vorbereitete Daten
        flattened_data = []

        for entry in features:
            row = {
                'crop_id': entry['crop_id'],
                'orig_image': entry['orig_image'],
                'area': entry['area'],
                'perimeter': entry['perimeter'],
                'length': entry['length'],
            }

            # 2D array flatten mit eindeutigen Namen
            shape_array = entry['shape'].flatten()
            half = len(shape_array) / 2
            for i, val in enumerate(shape_array):
                if i/half < 1:
                    row[f"distance_{'left'}_{i}"] = val
                else:
                    row[f"distance_{'right'}_{i}"] = val

            # 1D array für Intensitäten
            for i, val in enumerate(entry['intensities']):
                row[f"intensity_{i}"] = val

            flattened_data.append(row)

        # DataFrame erzeugen
        df = pd.DataFrame(flattened_data)
        return df
