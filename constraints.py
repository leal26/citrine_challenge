from scipy.optimize import differential_evolution

class Constraint():
    """Constraints loaded from a file."""

    def __init__(self, fname):
        """
        Construct a Constraint object from a constraints file

        :param fname: Name of the file to read the Constraint from (string)
        """
        with open(fname, "r") as f:
            lines = f.readlines()
        # Parse the dimension from the first line
        self.n_dim = int(lines[0])
        # Parse the example from the second line
        self.example = [float(x) for x in lines[1].split(" ")[0:self.n_dim]]

        # Run through the rest of the lines and compile the constraints
        self.exprs = []
        for i in range(2, len(lines)):
            # support comments in the first line
            if lines[i][0] == "#":
                continue
            self.exprs.append(compile(lines[i], "<string>", "eval"))
        return

    def get_example(self):
        """Get the example feasible vector"""
        return self.example

    def get_ndim(self):
        """Get the dimension of the space on which the constraints are defined"""
        return self.n_dim

    def apply(self, x):
        """
        Apply the constraints to a vector, returning True only if all are satisfied

        :param x: list or array on which to evaluate the constraints
        """
        for expr in self.exprs:
            if not eval(expr):
                return False
        return True   

class Explore():
    def __init__(self, constraints, n_results):
        self.constraints = constraints
        self.n_results = n_results
    
    def _func(self, x):
        test = self.constraints.apply(x)
        if test:
            return -1
        else:
            return 0
        
    def genetic_exploration(self, bounds = None):
        if bounds is None:
            bounds = self.constraints.n_dim*[[0, 10],]
        print(bounds)
        return differential_evolution(self._func, bounds, popsize=self.n_results)
if __name__ == '__main__':
    c = Constraint(r'.\formulation.txt')
    e = Explore(c, 10)
    print(e.genetic_exploration())