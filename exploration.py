from scipy.optimize import differential_evolution
import numpy as np


class Explore():
    def __init__(self, constraints, n_results):
        self.constraints = constraints
        self.n_results = n_results

    def random_selection(self, bounds=None):
        self.output = []
        counter = 0
        while counter != self.n_results:
            x_try = np.random.rand(self.constraints.n_dim)
            if bounds is not None:
                for i in range(self.constraints.n_dim):
                    x_try[i] = bounds[i][0] + x_try[i]*(bounds[i][1] -
                                                        bounds[i][0])
            if self.constraints.apply(x_try):
                self.output.append(x_try)
                counter += 1

    def genetic_exploration(self, bounds=None):
        if bounds is None:
            bounds = self.constraints.n_dim*[[0, 10], ]
        print(bounds)
        return differential_evolution(self._func, bounds,
                                      popsize=self.n_results)

    def write2file(self, filename='output.txt'):
        np.savetxt(filename, self.output, delimiter='\t')
