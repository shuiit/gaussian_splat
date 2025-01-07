import plotly



import plotly.graph_objects as go
import plotly.io as pio
# import plotly.express as px
# import matplotlib.cm
import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
from plotly.subplots import make_subplots

pio.renderers.default='browser'

# import matplotlib.pyplot as plt
import numpy as np


def plot_branch(branch,color,ax = False):
    if ax == False:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    
    points = np.vstack(branch)
    ax.scatter3D(points[:,0], points[:,1], points[:,2], 
                marker='o', s=20, c=color)  # Adjust marker size and color as desired
    ax.plot3D(points[:,0],points[:,1],points[:,2],c = color)
    return ax




def scatter3d(fig,data,legend, mode = 'markers',line_dict = {},marker_dict = {}):

    marker_dict = marker_dict if 'markers' in mode else {}
    
    # Include line dict if mode includes 'lines'
    line_params = line_dict if 'lines' in mode else {}
    
      
    fig.add_trace(go.Scatter3d(
        x=data[:, 0],
        y=data[:, 1],
        z=data[:, 2],
        mode=mode,
        marker=marker_dict,
        line = line_params,
        name = legend
    ))
    
    # Update layout to set aspectmode to 'cube'
    fig.update_layout(scene=dict(
        aspectmode='data'  # Ensures x, y, z axes have the same scale
    ))
    return fig

def plot_cones(fig, points, normals,skip = 10,sizeref = 1000,opacity = 0.5):
    skip = 10
    fig.add_trace(go.Cone(
    x=points[::skip,0],
    y=points[::skip,1],
    z=points[::skip,2],
    u=normals[::skip,0],
    v=normals[::skip,1],
    w=normals[::skip,2],
    opacity= opacity,
    sizemode="absolute",
    showscale = False,
    sizeref=sizeref))
    fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.8),aspectmode = 'data',
                             camera_eye=dict(x=1.2, y=1.2, z=0.6)))



def plot_skeleton_and_skin(skin_points,skin,skeleton,skip_skin_points = 10, idx_weight = 3, normals = False, **kwargs):
    marker_dict_skin = {'size':4,'color':skin.weight[::skip_skin_points,idx_weight],  # Set color to distances
                    'colorscale':'Viridis','opacity':0.3}
    marker_dict_skeleton = {'size': 10}
    line_dict_skeleton = {'width': 10}

    fig = go.Figure()
    if isinstance(normals, np.ndarray):
        plot_cones(fig, skin_points,normals, skip = skip_skin_points,**kwargs)
    else:
        scatter3d(fig,skin_points[::skip_skin_points,:],'skin',marker_dict = marker_dict_skin)
    
    skeleton.plot_skeleton(fig,marker_dict_skeleton,line_dict_skeleton)
    fig.show()