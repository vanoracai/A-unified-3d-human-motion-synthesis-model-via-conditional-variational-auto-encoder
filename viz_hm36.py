"""
    Functions to visualize human poses
    adapted from https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/viz.py
"""
import matplotlib.pyplot as plt
import numpy as np
# import h5py
import os
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial.transform import Rotation as R


class Ax3DPose(object):
    """
    num_obj means only show one skeleton or show two skeletons together to get comparisions
    """
    def __init__(self, fig, num_obj=1, titles=None):
        ## REMOVE 15 from I and J
        self.I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])
        self.J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        self.LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
        self.num_obj = num_obj

        vals = np.zeros((17, 3))

        ### 1 means right 0 means left
        # Make connection matrix
        self.plots = []
        self.ax_3d = []
        self.scatt =[[] for _ in range(num_obj)]
        self.scatt_flag = False
        for i_obj in range(num_obj):
            ax = fig.add_subplot(1, num_obj, 1+i_obj, projection='3d')
            self.subplots = []
            for i in np.arange( len(self.I) ):
                x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
                z = -np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
                y = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )

                self.subplots.append(ax.plot(x, y, z, lw=2))
  
            self.ax_3d.append(ax)

            if titles is not None:
                self.ax_3d[i_obj].set_title(titles[i_obj])
            self.ax_3d[i_obj].set_xticks([])
            self.ax_3d[i_obj].set_yticks([])
            self.ax_3d[i_obj].set_zticks([])
            self.ax_3d[i_obj].get_xaxis().set_ticklabels([])
            self.ax_3d[i_obj].get_yaxis().set_ticklabels([])
            self.ax_3d[i_obj].set_zticklabels([])
            self.ax_3d[i_obj].set_aspect('equal')
            # Get rid of the panes (actually, make them white)
            white = (1.0, 1.0, 1.0, 0.0)
            self.ax_3d[i_obj].w_xaxis.set_pane_color(white)
            self.ax_3d[i_obj].w_yaxis.set_pane_color(white)
            self.ax_3d[i_obj].w_zaxis.set_pane_color(white)
            # Keep z pane

            # Get rid of the lines in 3d
            self.ax_3d[i_obj].w_xaxis.line.set_color(white)
            self.ax_3d[i_obj].w_yaxis.line.set_color(white)
            self.ax_3d[i_obj].w_zaxis.line.set_color(white)

            self.ax_3d[i_obj].set_aspect('equal')
            self.ax_3d[i_obj].view_init(azim=-10., elev=-32.)
            r_y = 1000
            r_x = 1000
            r_z = 1000
            xroot, yroot, zroot = 0, 0, 0
            self.ax_3d[i_obj].set_xlim3d([-r_x + xroot, r_x + xroot])
            self.ax_3d[i_obj].set_zlim3d([-r_z + zroot, r_z + zroot])
            self.ax_3d[i_obj].set_ylim3d([-r_y + yroot, r_y + yroot])

            self.plots.append(self.subplots)

    def update(self, channels, mask=None, lcolor="#3498db", rcolor="#e74c3c"):


        if len(lcolor) ==1:
            lcolor = lcolor * self.num_obj
            rcolor = rcolor * self.num_obj
        vals = [channel.reshape([-1,3]) for channel in channels]

        for i_obj in range(self.num_obj):
            if_draw = mask is None or i_obj != 0 or (mask==1).sum()==0 
            if mask is not None and (mask==1).sum() == 0 and i_obj==0:
                vals[i_obj][:,:] = 0
            for i in np.arange( len(self.I) ):
                if if_draw or( mask is not None and mask[self.I[i]][0]>0 and mask[self.J[i]][0]>0) :

                    x = np.array( [vals[i_obj][self.I[i], 0], vals[i_obj][self.J[i], 0]] )
                    z = -np.array( [vals[i_obj][self.I[i], 1], vals[i_obj][self.J[i], 1]] )
                    y = np.array( [vals[i_obj][self.I[i], 2], vals[i_obj][self.J[i], 2]] )
                    self.plots[i_obj][i][0].set_xdata(x)
                    self.plots[i_obj][i][0].set_ydata(y)
                    self.plots[i_obj][i][0].set_3d_properties(z)
                    self.plots[i_obj][i][0].set_color(lcolor[i_obj] if self.LR[i] else rcolor[i_obj])

 

class Ax2DPose(object):
    def __init__(self, ax):

        vals = np.zeros((17, 3))
        self.ax = ax
        self.I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])
        self.J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        self.LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)


        ### 1 means right 0 means left
        # Make connection matrix
        self.plots = []
        for i in np.arange( len(self.I) ):
          x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
          y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
          self.plots.append(self.ax.plot(x, y, lw=2))
        self.scatt_flag =  False
  

        self.ax.set_aspect('equal')

        self.ax.set_ylabel("z")
        self.ax.set_xlabel("x")
        self.ax.set_axis_off()


    def update(self,  channels, mask=None, lcolor="#3498db", rcolor="#e74c3c", mask_color=None):

        vals = np.reshape( channels, (-1, 3) )
        r = R.from_euler('y', 35, degrees=True)
        vals = r.apply(vals)

        # Make connection matrix
        if_draw = mask is None or (mask == 1).sum() == 0
        if mask is not None and (mask == 1).sum() == 0:
            vals[:,:] = 0
        for i in np.arange( len(self.I) ):
            if if_draw or (mask is not None and mask[self.I[i]][0] > 0 and mask[self.J[i]][0] > 0):
                x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
                y = -np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
                self.plots[i][0].set_xdata(x)
                self.plots[i][0].set_ydata(y)
                self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)


    
        r_x = 600
        r_y = 1000

        xroot, yroot = vals[0,0], vals[0,1]
        self.ax.set_xlim([-r_x+xroot, r_x+xroot])
        self.ax.set_ylim([-r_y+yroot, r_y+yroot])



