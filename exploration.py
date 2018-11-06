from scipy.spatial.distance import directed_hausdorff
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


class Explore():
    def __init__(self, constraints, n_results, inflation=0,
                 output_file='output_file.txt'):
        """Construct a Explore object to generate output file

        :param constraints: object to apply constraints (Constraint class)
        :param n_results: number of output combinations (int)
        :param inflation: increases combinations to be explored for diversity
                          (float)
        :param output_file: file to output all data
        """
        self.constraints = constraints
        self.n_results = n_results
        self.inflation = inflation
        self.n = (1+self.inflation)*self.n_results
        self.filename = output_file

    def random(self, bounds=None, crowding_distance=None):
        """Populate output array with randomly generated individuals that
        satisfy constraints

        :param bounds: lower and upper bounds for each input (list, dim (n_dim, 2))
        :param crowding_distance: minimum distance between explored
                                  configurations allowable. If '0', there is
                                  no constraint (float)
        """
        np.random.seed()
        self.output = np.zeros((self.n, self.constraints.n_dim))
        counter = 0
        while counter != self.n:
            x_try = np.random.rand(self.constraints.n_dim)
            x_try = self._scale2bounds(x_try, bounds)
            crowding_constraint = self._crowding_constraint(x_try,
                                                            crowding_distance,
                                                            counter)
            if self.constraints.apply(x_try) and crowding_constraint:
                self.output[counter, :] = x_try
                counter += 1

    def _scale2bounds(self, x, bounds):
        """Scales bounds from [0,1) to any desired bounds

        :param x: vector to scale (list, dim n_dim)
        :param bounds: lower and upper bounds for each input (list, dim (n_dim, 2))
        """
        if bounds is not None:
            for i in range(self.constraints.n_dim):
                try:
                    x[:, i] = bounds[i][0] + x[:, i]*(bounds[i][1] -
                                                      bounds[i][0])
                except(TypeError):
                    x[i] = bounds[i][0] + x[i]*(bounds[i][1] - bounds[i][0])
        return x

    def _crowding_constraint(self, x, crowding_distance, first_flag=0):
        """Determines if explored node is too close or not to all previously
           explored nodes
           :param x: node to explore (list, dim n_dim)
           :param crowding_distance: minimum distance between explored
                                     configurations allowable. (float)
           :param first_flag: flag to determine if first explored combination.
                              (int)
        """
        self.crowding_distance = crowding_distance
        if crowding_distance is None or first_flag != 0:
            return True
        else:
            norms = np.linalg.norm(np.array(self.output) - np.array(x), axis=1)
            distance = min(norms)

            if type(crowding_distance) == str:
                crowding_distance = directed_hausdorff(np.array(self.output),
                                                       np.array([x]))[0]
            return distance >= crowding_distance

    def deflate(self):
        """Remove the (inflation*n_results) less diverse individuals"""
        data = []
        self.output = np.array(self.output)
        for i in range(self.n):
            norms = np.linalg.norm(self.output - np.array([self.output[i, :]]),
                                   axis=1)
            distance = sorted(norms)[1]
            data.append((distance, np.where(norms == distance)))

        sorted_data = sorted(data)
        counter = 0
        while counter != self.inflation*self.n_results:
            self.output[sorted_data[counter][1], :] = 0
            counter += 1
        self.output = self.output[~np.all(self.output == 0, axis=1)]

    def write2file(self):
        """Write all explored configurations to text file"""
        np.savetxt(self.filename, self.output, delimiter=' ')

    def plotting(self, index=(0, 1), grid_size=(100, 100),
                 plot_training=False, plot_grid=False):
        """ Plot distribution of sampled domain using Kernel Density
        estimation (KDE)

        :param index: dimensions of x to plot (e.g. index=(0,1) plots x[0] vs
                      x[1]) (list, dim 2)
        :param grid_size: x and y explored for plotting (list, dim 2)
        :param plot_training: plot explored nodes used to train KDE (bool)
        :param plot_grid: plot grid used to generate surface (bool)
        """

        x = self.output[:, index[0]]
        y = self.output[:, index[1]]
        xmin, xmax = min(x), max(x)
        ymin, ymax = min(y), max(y)

        # Determine probability density function
        self.kernel = gaussian_kde(self.output[:, index].T)
        X, Y = np.mgrid[xmin:xmax:complex(grid_size[0]),
                        ymin:ymax:complex(grid_size[1])]
        positions = np.vstack([X.ravel(), Y.ravel()])

        P = np.reshape(self.kernel(positions).T, X.shape)

        plt.figure()
        plt.contourf(X, Y, P)
        if plot_training:
            plt.scatter(x, y)
        if plot_grid:
            plt.scatter(X, Y)
        plt.colorbar(norm=mcolors.NoNorm)
        plt.xlabel('Input %i' % index[0])
        plt.ylabel('Input %i' % index[1])
        plt.title('Crowding distance = %f' % self.crowding_distance)
        plt.show()
