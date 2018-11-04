import matplotlib.pyplot as plt
import time
from constraints import Constraint
from exploration import Explore


c = Constraint(r'.\formulation.txt')
e = Explore(c, 1000)

# start = time.time()
# # e.random()
# e.random(crowding_distance=0.1)
# # e.write2file()
# end = time.time()
# print(end - start)
#
# plt.figure()
# x, y = e.output.T
# plt.scatter(x, y)
#
# e.random(crowding_distance='Hausdorff')
#
# x, y = e.output.T
# plt.scatter(x, y, c='r')
# plt.show()

e.random(crowding_distance=0.1)
e.plotting(index=(2, 3))

e.random(crowding_distance='Hausdorff')
e.plotting(index=(2, 3))
plt.show()
