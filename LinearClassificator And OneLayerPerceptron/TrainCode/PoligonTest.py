import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

ax = plt.gca()
xmin, xmax = ax.get_xbound()
ymin, ymax = ax.get_ybound()

verts = [(xmin, ymin), # left, bottom
         (1., 0.), # left, top
         (1., 1.), # right, top
         (xmax, ymax), # right, bottom
         (0., 0.),] # ignored

codes = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.LINETO,
         Path.CLOSEPOLY,]

path = Path(verts, codes)
fig = plt.figure()
ax = fig.add_subplot(111)
patch = patches.PathPatch(path, facecolor='#90EE90', lw=2)
ax.add_patch(patch) 
ax.axis('equal')
plt.show()