from scipy.stats import gaussian_kde
from scipy.spatial.distance import directed_hausdorff
import numpy as np


class Explore():
    def __init__(self, constraints, n_results):
        self.constraints = constraints
        self.n_results = n_results

    def _scale2bounds(self, x, bounds):
        if bounds is not None:
            for i in range(self.constraints.n_dim):
                try:
                    x[:, i] = bounds[i][0] + x[:, i]*(bounds[i][1] -
                                                      bounds[i][0])
                except(TypeError):
                    x[i] = bounds[i][0] + x[i]*(bounds[i][1] - bounds[i][0])
        return x

    def _crowding_constraint(self, x, crowding_distance, counter):
        self.crowding_distance = crowding_distance
        if crowding_distance is None or counter != 0:
            return True
        else:
            norms = np.linalg.norm(np.array(self.output) - np.array(x), axis=1)
            distance = min(norms)

            if type(crowding_distance) == str:
                crowding_distance = directed_hausdorff(np.array(self.output),
                                                       np.array([x]))[0]
            return distance >= crowding_distance

    def random(self, bounds=None, crowding_distance=None):
        self.output = np.zeros((self.n_results, self.constraints.n_dim))
        counter = 0
        while counter != self.n_results:
            x_try = np.random.rand(self.constraints.n_dim)
            x_try = self._scale2bounds(x_try, bounds)
            crowding_constraint = self._crowding_constraint(x_try,
                                                            crowding_distance,
                                                            counter)
            if self.constraints.apply(x_try) and crowding_constraint:
                self.output[counter, :] = x_try
                counter += 1

    def post_filter(self, n_target):
        data = []
        self.output = np.array(self.output)
        for i in range(self.n_results):
            norms = np.linalg.norm(self.output - np.array([self.output[i, :]]),
                                   axis=1)
            distance = sorted(norms)[1]
            data.append((distance, np.where(norms == distance)))

        sorted_data = sorted(data)
        counter = 0
        while counter != self.n_results - n_target:
            self.output[sorted_data[counter][1], :] = 0
            counter += 1
        self.output = self.output[~np.all(self.output == 0, axis=1)]

    def write2file(self, filename='output.txt'):
        np.savetxt(filename, self.output, delimiter='\t')

    def plotting(self, index=(0, 1), grid_size=(100, 100),
                 plot_training=False, plot_grid=False):
        import matplotlib.pyplot as plt
        x = self.output[:, index[0]]
        y = self.output[:, index[1]]
        xmin, xmax = min(x), max(x)
        ymin, ymax = min(y), max(y)

        self.kernel = gaussian_kde(self.output[:, index].T)
        X, Y = np.mgrid[xmin:xmax:complex(grid_size[0]),
                        ymin:ymax:complex(grid_size[1])]
        positions = np.vstack([X.ravel(), Y.ravel()])
        P = np.reshape(self.kernel(positions).T, X.shape)

        # P = self.kernel(A.flatten(), V.flatten())

        # plt.figure()
        plt.contourf(X, Y, P)
        if plot_training:
            plt.scatter(x, y)
        if plot_grid:
            plt.scatter(X, Y)
        plt.colorbar()
        plt.xlabel('Input %i' % index[0])
        plt.ylabel('Input %i' % index[1])
        plt.title(self.crowding_distance)
        # plt.show()
