import os

import numpy as np
import skimage
from matplotlib import pyplot as plt
from spatial_efd import spatial_efd



class FeatureExtractionVisualizer:
    def __init__(self, kn_folder_path):
        self.kn_folder_path = kn_folder_path

    def plot_curvature(self, curvature):
        #image_path = os.path.join(self.kn_folder_path, filename)

        # Load image and get contour
        image = skimage.io.imread(image_path)
        image = skimage.morphology.area_closing(image, 10)

        contour = skimage.measure.find_contours(image, 0.8)[0]

        coeffs = spatial_efd.CalculateEFD(contour[:, 0], contour[:, 1], harmonics=20)
        norm_coeff, rotation = spatial_efd.normalize_efd(coeffs, size_invariant=True)
        xt, yt = spatial_efd.inverse_transform(coeffs, harmonic=20, n_coords=10000)

        fig, axs = plt.subplots(1, 2, figsize=(15, 6))

        # Spline Plot with curvature on spline line
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

        plt.tight_layout()
        plot_filename = os.path.join(output_folder_path, f"plot_{filename.split('.p')[0]}.png")
        plt.show()
        # plt.savefig(plot_filename)
        plt.close()

    def plot_endpoints(self):
        # ToDo
        pass

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
