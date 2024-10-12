#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 22:31:30 2024

@author: acxyle
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ----------------------------------------------------------------------------------------------------------------------
# components
def color_to_hex(color_name):
    """
        assume the input is legal hex color name
    """
    if color_name.startswith('#') and len(color_name) == 7:
        return color_name
    else:
        return mcolors.to_hex(mcolors.CSS4_COLORS[color_name])


def lighten_color(hex_color, increase_value=80):

    increase_value = min(255, increase_value)
    
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:], 16)
    r, g, b = min(r + increase_value, 255), min(g + increase_value, 255), min(b + increase_value, 255)

    return f"#{r:02x}{g:02x}{b:02x}"


def darken_color(hex_color, decrease_value=80):
    
    decrease_value = max(0, decrease_value)
    
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:], 16)  
    r, g, b = max(r - decrease_value, 0), max(g - decrease_value, 0), max(b - decrease_value, 0)
    
    return f"#{r:02x}{g:02x}{b:02x}"

# ----------------------------------------------------------------------------------------------------------------------
# functions
def plot_pie_chart(fig, ax, values, labels, title=None, colors=None, explode=None, **kwargs):
    """
        ...
    """

    if explode is None:
        explode = np.zeros(len(values))
    
    ax.pie(values, colors = colors, labels=labels, autopct='%1.1f%%', pctdistance=0.85, explode=explode)
    
    centre_circle = plt.Circle((0,0),0.70, fc='white')

    fig.gca().add_artist(centre_circle)    

    ax.axis('equal')  
    
    for i, label in enumerate(labels):
        if values[i] < 10:  # Adjust threshold for small slices
            x, y = ax.patches[i].theta1, ax.patches[i].r
            ax.annotate(label + " - " + str(values[i]) + "%", xy=(x, y), xytext=(x+1.2, y+0.5),
                         arrowprops=dict(facecolor='black', shrink=0.05), fontsize=20)
    
    ax.set_title(f'{title}', x=0.45, y=0.45)