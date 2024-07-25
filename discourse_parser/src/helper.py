import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors

width_dist = 10
depth_dist = 10
levels = 5

paras = []
para_cache = ''

with open('../data/sample/article', 'r') as fin3:
    for line in fin3:
        if line.strip():
            para_cache += line.strip() + ' '
        else:
            paras.append(para_cache.strip())
            para_cache = ''
    if para_cache:
        paras.append(para_cache)
        
print(len(paras))