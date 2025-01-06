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
