import matplotlib.pyplot as plt
import time
from constraints import Constraint
from exploration import Explore


c = Constraint(r'.\mixture.txt')
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

plt.figure()
e.random(crowding_distance='Hausdorff')
e.plotting(index=(0, 1))
plt.figure()
e.post_filter(50)
e.plotting(index=(0, 1))
plt.show()
