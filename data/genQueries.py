# USAGE: python genQueries <Data File>
import sys
import random
import numpy as np

dim = 12
filename = str(dim) + 'D-qI0'

rs = [.5]

lines = []
for r in rs:
    queries = np.zeros([1000, 2*dim])
    for d in range(dim):
        queries[:,d] = np.random.uniform(low=0, high=1-r, size=(1000))
        queries[:,d+dim] = queries[:,d] + r
    for q in queries:
        qStr = ['r']
        qStr += ["{0:.3f}".format(i) for i in q]
        qStr.append(str(r))
        lines.append(' '.join(qStr) + '\n')

random.shuffle(lines)
with open(filename, 'w') as f:
    f.writelines(lines)
