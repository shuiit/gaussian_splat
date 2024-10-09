import plotly



import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import matplotlib.cm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plotly.subplots import make_subplots


pio.renderers.default='browser'

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 08:25:02 2023

@author: Roni
"""


def scatter3d(fig,data,color,size):
    fig.add_trace(go.Scatter3d(
        x=data[:,0],
        y=data[:,1],
        z=data[:,2],
        mode='markers',
        marker=dict(
            color=color,  # Set marker color to green
            size=size  # Set marker size to small
        )
        ))
    # Update layout to set aspectmode to 'cube'
    fig.update_layout(scene=dict(
        aspectmode='data'  # Ensures x, y, z axes have the same scale
    ))
    return fig

def plot_hull(real_hull,size = 3):
    colors = ['green','red','blue']

    fig = go.Figure()
    [scatter3d(fig,data,color,size) for data,color in zip(real_hull.values(),colors)]

    fig.show()
