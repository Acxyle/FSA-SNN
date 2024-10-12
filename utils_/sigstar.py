#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 22:01:26 2024

@author: acxyle

    local python re-write for matlab code: https://github.com/raacampbell/sigstar/tree/master
"""

import numpy as np


__all__ = ['sigstar']

# ------ sigstar functions
def sigstar(groups, stats, ax, nosort=False):
    """
        ...
    """
    if stats is None or len(stats) == 0:
        stats = [0.05] * len(groups)
        
    xlocs = np.array(groups)
    
    """
        Optionally sort sig bars from shortest to longest so we plot the shorter ones first
        in the loop below. Usually this will result in the neatest plot. If we waned to 
        optimise the order the sig bars are plotted to produce the neatest plot, then this 
        is where we'd do it. Not really worth the effort, though, as few plots are complicated
        enough to need this and the user can define the order very easily at the command line. 
    """
    if not nosort:
        sort_inds = np.argsort(xlocs[:, 1] - xlocs[:, 0])
        xlocs = xlocs[sort_inds]
        stats = np.array(stats)[sort_inds]
    
    """
        Add the sig bar lines and asterisks 
    """
    H = []
    for i in range(len(groups)):
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        y_offset = y_range * 0.05
        
        thisY = findMinY(xlocs[i], ax) + y_offset
        
        h = makeSignificanceBar(xlocs[i], thisY, stats[i], ax)
        H.append(h)
        
    """
        Now we can add the little downward ticks on the ends of each line. We are
        being extra cautious and leaving this it to the end just in case the y limits
        of the graph have changed as we add the highlights. The ticks are set as a
        proportion of the y axis range and we want them all to be the same the same
        for all bars.
    """
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    yd = y_range * 0.01
    
    for h in H:
        line = h['line']
        y = line.get_ydata()
        y[0] -= yd
        y[3] -= yd
        line.set_ydata(y)

    return H


def makeSignificanceBar(x, y, p, ax):
    """
        makeSignificanceBar produces the bar and defines how many asterisks we get for a 
        given p-value ★ *
    """
    if p<=1E-3:
        stars='***'
    elif p<=1E-2:
        stars='**'
    elif p<=0.05:
        stars='*'
    elif np.isnan(p):
        stars='N/A'
    else:
        stars='n.s.'
        
    x = np.tile(x, (2,1))
    y = np.tile(y, (4,1))
    
    # FIXME --- need to find a better way to rewrite
    line, = ax.plot(x.T.ravel(), y, 'C0', linewidth=1.5)
    line.set_label('sigstar_bar')
    
    if not np.isnan(p):
        offset=0.015
    else:
        offset=0.02
    
    starY=np.mean(y)+myRange(ax.get_ylim())*offset
    
    text = ax.text(np.mean(x), starY, stars, horizontalalignment='center', backgroundcolor='none', fontsize=14)
    text.set_label('sigstar_stars')
    
    Y = ax.get_ylim()
    if Y[1]<starY:
        ax.set_ylim([Y[0], starY+myRange(Y)*0.05])
            
    H = {'line': line, 'text': text}
    
    return H
    
def myRange(x):
    
    return np.max(x)-np.min(x)


def findMinY(x, ax):
    """
        The significance bar needs to be plotted a reasonable distance above all the data points
        found over a particular range of X values. So we need to find these data and calculat the 
        the minimum y value needed to clear all the plotted data present over this given range of 
        x values. 
       
        This version of the function is a fix from Evan Remington
    """
    
    old_xlim = ax.get_xlim()
    old_ylim = ax.get_ylim()
    
    ax.autoscale_view(tight=True)
    
    x[0] += 0.1
    x[1] += 0.1
    
    ax.set_xlim(x)
    
    yLim = ax.get_ylim()
    Y = max(yLim)
    
    #ax.relim()
    #ax.autoscale_view()
    #ax.set_aspect('auto')
    
    ax.set_xlim(old_xlim)
    ax.set_ylim(old_ylim)
    
    return Y     # [question] should return ax


def generate_threshold(input, alpha=1, delta=2):
    """
        return np.mean(input)+delta*np.std(input), delta(default)=2
    """
    return alpha*np.mean(input)+delta*np.std(input)