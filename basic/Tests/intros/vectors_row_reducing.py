from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()
ax = plt.axes(projection='3d')
origin = (0,0,0)
#
col='k'
vec1 = [2, -1, 0]
vec2 = [-1, 2,-3]
vec3 = [0, -1, 4]

xx, yy = np.meshgrid(range(10), range(10))
norm12 = np.cross(vec1, vec2)
z = -(norm12[0]*xx + norm12[1]*yy)/norm12[2]
ax.plot_surface(xx,yy,z, alpha = 0.5)
ax.quiver(*origin,*vec1, color=col)
ax.quiver(*origin,*vec2, color=col)
ax.quiver(*origin,*vec3, color=col)
ax.scatter(0,-1,4, color='r')
ax.scatter(1,1,-3,color ='b')
#

col = 'b'
ax.quiver(*origin,0,-1,0, color=col)
ax.quiver(*origin,3,2,-3, color=col)
ax.quiver(*origin,-2,-1,4, color=col)
#
col = 'g'
ax.quiver(*origin,0,-1,0, color=col)
ax.quiver(*origin,0, 2,-3, color=col)
ax.quiver(*origin,2,-1,4, color=col)
#
col = 'r'
ax.quiver(*origin,0,-1,0, color=col)
ax.quiver(*origin,0,2,-3, color=col)
ax.quiver(*origin,1,0,4, color=col)
col = 'c'
ax.quiver(*origin,0,-1,0, color=col)
ax.quiver(*origin,0,2,-3, color=col)
ax.quiver(*origin,1,0,0, color=col)


##
scale = 4
ax.set_xlim([-scale,scale])
ax.set_ylim([-scale,scale])
ax.set_zlim([-scale,scale])
plt.show()