import time
from constraints import Constraint
from exploration import Explore

c = Constraint(r'.\formulation.txt')
e = Explore(c, 1000)

start = time.time()
e.random_selection()
e.write2file()
end = time.time()
print(end - start)
