import os
import numpy as np
import skimage
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from spatial_efd import spatial_efd # spatial-efd for installation

import plotly.graph_objects as go

class FeatureVisualizer:
    def __init__(self):
        pass

    def make_plot(self, xt, yt, curvature, endpoints, grid):
        axs = None

        if curvature:
            # Spline Plot with curvature on spline line
            fig, axs = plt.subplots(1, 2, figsize=(15, 6))
            sc = axs[0].scatter(xt, yt, c=curvature, cmap='viridis', label='Curvature')
            fig.colorbar(sc, ax=axs[0], label='Curvature')
            axs[0].legend()
            axs[0].set_xlabel('X-axis')
            axs[0].set_ylabel('Y-axis')
            axs[0].set_title('Spline with Curvature')
            axs[0].axis("equal")

            # Coordinates vs Curvature Plot
            axs[1].plot(np.arange(len(curvature)), curvature, label='Curvature')
            axs[1].set_xlabel('Coordinate Index')
            axs[1].set_ylabel('Curvature')
            axs[1].set_title('Curvature vs Coordinates')
            axs[1].legend()

        if endpoints:
            if axs:
                ax = axs[0]
            else:
                fig, ax = plt.subplots(figsize=(15, 6))
            ax.scatter(xt[endpoints[0]], yt[endpoints[0]], color="red")
            ax.scatter(xt[endpoints[1]], yt[endpoints[1]], color="red")

        if grid:
            if axs:
                ax = axs[0]
            else:
                fig, ax = plt.subplots()

            poly_collection = PolyCollection(cells, facecolors='cyan', edgecolors='r', alpha=.25)
            ax.add_collection(poly_collection)

            # Plot each cell boundary
            for cell in cells:
                polygon = plt.Polygon(cell, closed=True, edgecolor='black', fill=None)
                ax.add_patch(polygon)

            # Plot horizontal lines
            for pos_coords, neg_coords in zip(vertical_coordinates_pos, vertical_coordinates_neg):
                ax.plot([pt[0] for pt in pos_coords], [pt[1] for pt in pos_coords], 'b-')
                ax.plot([pt[0] for pt in neg_coords], [pt[1] for pt in neg_coords], 'b-')

            # Plot vertical lines
            for i in range(len(vertical_coordinates_pos[0])):
                vertical_line_pos = [coords[i] for coords in vertical_coordinates_pos]
                vertical_line_neg = [coords[i] for coords in vertical_coordinates_neg]
                ax.plot([pt[0] for pt in vertical_line_pos], [pt[1] for pt in vertical_line_pos], 'b-')
                ax.plot([pt[0] for pt in vertical_line_neg], [pt[1] for pt in vertical_line_neg], 'b-')

            ax.plot(xt, yt, "r-")

            ax.autoscale()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()

        plt.tight_layout()
        #plot_filename = os.path.join(self.output_folder, f"plot_{filename.split('.p')[0]}.png")
        plt.show()
        # plt.savefig(plot_filename)
        plt.close()



    def plot_spine(self):
        # ToDo
        pass

    def plot_extended_midline(self):
        # ToDo
        pass

    def plot_normals(self):
        # ToDo
        pass

    def plot_grid(self):
        # ToDo
        pass
