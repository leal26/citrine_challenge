from scipy.spatial.distance import directed_hausdorff
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import random


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

    def select(self, algorithm='random', bounds=None,
               crowding_distance=None, mutation_period=10):
        """Populate output array with randomly generated individuals that
        satisfy constraints

        :param algorithm:
                  - 'random': all configurations are randomly generated
                  - 'mutation': all configurations are mutated from previous
                                feasible combinations
                  - 'hybrid': generates configurations randomly and also mutates
                              (how often is defined by mutation_period)
        :param bounds: lower and upper bounds for each input (list, dim (n_dim, 2))
        :param crowding_distance: minimum distance between explored
                                  configurations allowable. If '0', there is
                                  no constraint (float)
        :param mutation_period: determines how often a configuration is
                                mutated (slight modifications from previous)
        """
        np.random.seed()

        # definiting internal variables
        self.crowding_distance = crowding_distance
        if self.inflation == 0:
            deflate_period = self.n_results
        else:
            deflate_period = self.n_results*2
        self.output = [self.constraints.example]
        self.all = [self.constraints.example]

        # deifining counters foe algorithms
        counter = 1
        inflation_counter = 0
        mutate = 1
        while inflation_counter != (self.inflation+1):
            if algorithm == 'random' or (algorithm == 'hybrid' and
                                         mutate % mutation_period):
                x_try = np.random.rand(self.constraints.n_dim)
                x_try = self._scale2bounds(x_try, bounds)
            elif algorithm == 'mutation' or algorithm == 'hybrid':
                x_try = self._mutate()

            if self.constraints.apply(x_try):
                crowding_constraint = self._crowding_constraint(x_try,
                                                                counter)
                if crowding_constraint:
                    self.output.append(x_try)
                    self.all.append(x_try)
                    counter += 1

            if not len(self.output) % deflate_period:
                self.deflate()
                inflation_counter += 1
            mutate += 1

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
                except(TypeError, IndexError):
                    x[i] = bounds[i][0] + x[i]*(bounds[i][1] - bounds[i][0])
        return x

    def _mutate(self):
        """Slightly modifies an already existing confioguration
        """
        def _fix(x, i):
            if x[i] < 0:
                x[i] = 0
            elif x[i] > 1:
                x[i] = 1
            return(x)
        bandwith = 0.1  # 10*self.crowding_distance

        j = np.random.randint(0, len(self.output))

        x = np.array(self.output[j])
        all = list(range(self.constraints.n_dim))
        if self.constraints.n_dim > 2:
            n = np.random.randint(1, self.constraints.n_dim/2)
        else:
            n = 1
        i = random.sample(all, k=n)
        for ii in sorted(i, reverse=True):
            del(all[ii])
        j = random.sample(all, k=n)
        for k in range(n):
            ii = i[k]
            jj = j[k]
            delta = np.random.uniform(self.crowding_distance, bandwith)
            x[ii] += delta
            x[jj] -= delta
            x = _fix(x, ii)
            x = _fix(x, jj)
        return(x)

    def _crowding_constraint(self, x, first_flag=0):
        """Determines if explored node is too close or not to all previously
           explored nodes
           :param x: node to explore (list, dim n_dim)
           :param first_flag: flag to determine if first explored combination.
                              (int)
        """

        if self.crowding_distance is None or first_flag == 0:
            return True
        else:
            norms = self._norm(x, domain='unequal')
            distance = min(norms)

            return distance >= self.crowding_distance

    def deflate(self):
        """Remove the (inflation*n_results) less diverse individuals"""
        data = []
        output_array = np.array(self.output)

        for i in range(len(self.output)):
            norms = self._norm(self.output[i], domain='unequal',
                               compared_to=self.output)
            distance = sorted(norms)[1]
            data.append((distance, np.where(norms == distance)[0][0]))

        sorted_data = np.array(sorted(data, reverse=True))
        # counter = 0

        to_keep = list(sorted_data[:self.n_results, 1].astype(int))
        self.output = list(output_array[to_keep])

    def _norm(self, x, domain='equal', compared_to=None):
        """Calculate norm

        :param x: combination vector to evaluate
        :param domain: if one design variable is orders of magnitude greater,
                       use 'unequal'. Otherwise, 'equal' will be sufficient and
                       faster"""
        if compared_to is None:
            all_array = np.array(self.all)
        else:
            all_array = np.array(compared_to)
        x_array = np.array(x)
        if domain == 'unequal':
            for i in range(self.constraints.n_dim):
                max_i = np.max(all_array[:, i])
                min_i = np.min(all_array[:, i])
                if (max_i - min_i) != 0:
                    all_array[:, i] = (all_array[:, i]-min_i)/(max_i-min_i)
                    x_array[i] = (x_array[i]-min_i)/(max_i-min_i)
        norms = np.linalg.norm(all_array - x_array, axis=1)
        return norms

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
        output_array = np.array(self.output)
        x = output_array[:, index[0]]
        y = output_array[:, index[1]]
        xmin, xmax = min(x), max(x)
        ymin, ymax = min(y), max(y)

        # Determine probability density function
        self.kernel = gaussian_kde(output_array[:, index].T)
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
        # plt.colorbar(norm=mcolors.NoNorm)
        plt.xlabel('Input %i' % index[0])
        plt.ylabel('Input %i' % index[1])
        plt.title('Crowding distance = %f' % self.crowding_distance)
        plt.show()
