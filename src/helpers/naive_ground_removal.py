import math
import cv2
import numpy as np
import cupy as cp

class NaiveGroundRemoval:
    def __init__(self, min_x, max_x, min_y, max_y, cell_size, tolerance, filter_type, filter_size, filter_iterations):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.cell_size = cell_size
        self.tolerance = tolerance
        self.filter = filter_type
        self.filter_size = filter_size
        self.filter_iterations = filter_iterations

        self.width = int(math.ceil((self.max_x - self.min_x) / self.cell_size))
        self.height = int(math.ceil((self.max_y - self.min_y) / self.cell_size))
        self.cols = cp.empty((self.width, self.height), dtype=cp.float32)

    def remove_ground(self, pointcloud):

        # convert x and y coordinates into indexes
        xi = ((pointcloud[:, 0] - self.min_x) / self.cell_size).astype(np.int32)
        yi = ((pointcloud[:, 1] - self.min_y) / self.cell_size).astype(np.int32)
        zi = pointcloud[:, 2]

        # write minimum height for each cell to cols
        # thanks to sorting in descending order,
        # the minimum value will overwrite previous values
        self.cols[...] = cp.nan
        idx = cp.argsort(-zi)
        self.cols[xi[idx], yi[idx]] = zi[idx]

        cols_cpu = cp.asnumpy(self.cols)

        # bring cell minimum lower, if all cells around it are lower
        for _ in range(self.filter_iterations):
            if self.filter == 'median':
                cols_filtered = cv2.medianBlur(cols_cpu, self.filter_size)
                np.fmin(cols_cpu, cols_filtered, out=cols_cpu)
            elif self.filter == 'average':
                mask = np.isnan(cols_cpu)
                cols_cpu[mask] = 0
                cols_filtered = cv2.blur(cols_cpu, (self.filter_size, self.filter_size), cv2.BORDER_REPLICATE) / \
                        cv2.blur((~mask).astype(np.float32), (self.filter_size, self.filter_size), cv2.BORDER_REPLICATE)
                np.fmin(cols_cpu, cols_filtered, out=cols_cpu)
            elif self.filter == 'minimum':
                mask = np.isnan(cols_cpu)
                cols_cpu[mask] = np.inf
                cols_filtered = cv2.erode(cols_cpu, np.ones((self.filter_size, self.filter_size)), cv2.BORDER_REPLICATE)
                np.fmin(cols_cpu, cols_filtered, out=cols_cpu)
            elif self.filter != 'none':
                assert False, "Unknown filter value: " + self.filter

        self.cols = cp.asarray(cols_cpu)

        # filter out closest points to minimum point up to some tolerance
        ground_mask = (zi <= (self.cols[xi, yi] + self.tolerance))

        # return non-ground points
        return pointcloud[~ground_mask]

