import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

#
colors = [(),(),(),()]

N = 256
blu_vals = np.ones((N, 4))
blu_vals[:, 0] = np.linspace(39/256,1, N)
blu_vals[:, 1] = np.linspace(102/256,1, N)
blu_vals[:, 2] = np.linspace(199/256,1, N)
bluecmp = ListedColormap(blu_vals)
bluecmp.set_bad(color='w')

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return (1-mix)*c1 + mix*c2 #mpl.colors.to_hex()#

north='#1f77b4' #blue "#50a2d5"
east = "#4bb900" #"#76bb4b" #green
south= '#ffe200' # yellow
west = "#eb3920"# red
n=90
fade_1 = []
fade_2 = []
fade_3 = []
fade_4 = []
for i in range(n):
    fade_1.append(colorFader(north,east,i/n))
    fade_2.append(colorFader(east,south,i/n))
    fade_3.append(colorFader(south,west,i/n))
    fade_4.append(colorFader(west,north,i/n))


fade = ListedColormap(np.vstack(fade_1 + fade_2 + fade_3 + fade_4))



##### generate data grid like in above
N=256
x = np.linspace(-2,2,N)
y = np.linspace(-2,2,N)
z = np.zeros((len(y),len(x))) # make cartesian grid
for ii in range(len(y)):
    z[ii] = np.arctan2(y[ii],x) # simple angular function

fig = plt.figure()
ax = plt.gca()
pmesh = ax.pcolormesh(x, y, z/np.pi,
    cmap = fade, vmin=-1, vmax=1)
plt.axis([x.min(), x.max(), y.min(), y.max()])
cbar = fig.colorbar(pmesh)
cbar.ax.set_ylabel('Phase [pi]')
plt.show()