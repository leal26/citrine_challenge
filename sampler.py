import time
import sys

from constraints import Constraint
from exploration import Explore

# read inputs
inputs = sys.argv[-3:]
input_file = inputs[0]
output_file = inputs[1]
n_results = int(inputs[2])

# Algorithm parameters
inflation = 4
crowding_distance = 0.0005
mutation_period = 4
# Defining main objects for exploration
c = Constraint(input_file)

e = Explore(c, n_results, inflation, output_file)

# Run and return run time
start = time.time()
e.select(crowding_distance=crowding_distance, algorithm='hybrid',
         mutation_period=mutation_period)
# e.deflate()
e.write2file()
end = time.time()
print('Time', end - start)
# Plotting sampling distribution just for visualization
e.plotting(index=(0, 1), plot_training=True)
