### This file contains several useful plotting functions

# importing useful modules
import numpy as np
from numpy.linalg import *
import pandas as pd
import scipy as sp
from scipy.spatial.distance import pdist
from scipy.spatial import distance
from scipy.stats import rankdata
from scipy import stats
import scipy.cluster.hierarchy as sch
import math

# matplotlib stuff
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams
import matplotlib.cm as cm
import matplotlib.colors as col
from matplotlib import ticker
import pylab as py
from matplotlib.ticker import MaxNLocator

# scientific and computing stuff
import uniprot
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from itertools import combinations, permutations, product

# colorbrewer2 Dark2 qualitative color table
import brewer2mpl
dark2_cmap = brewer2mpl.get_map('Dark2', 'Qualitative', 7)
dark2_colors = dark2_cmap.mpl_colors

rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 1000
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'white'
rcParams['patch.facecolor'] = dark2_colors[0]
rcParams['font.family'] = 'Helvetica'

# suppress annoying "deprecation warnings"
import warnings
warnings.filterwarnings("ignore")


##########################################################################################


# procedure to minimize chartjunk by stripping out unnecessary plot borders and axis ticks    
# the top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
def remove_border(axes=None, top=False, right=False, left=True, bottom=True):

    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    
    # turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    
    # now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()
        
        
# procedure to take away tick marks, but leave plot frame
def yes_border_no_ticks(axes=None, border=True):
    
    ax = axes or plt.gca()
    
    # removing frame and ticks
    remove_border(ax, left=False, right=False, top=False, bottom=False)
    
    if border:
        # putting frame back
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        

##########################################################################################


# making procedure to set some preferred plot parameters
# 'fig' is the figure name
# 'ax' is an axes handle
# if numbers are in xlabels, make sure they are integers and not strings
# 'ylim' is a list with a minimum value and a maximum value
# if no ylim desired, insert empty list
def set_plot_params(fig=None, ax=None, xlabels=[], xlim=[], ylim=[], 
                    tick_size=17, label_size=25, legend_size=15, 
                    legend_loc='best', all_axis_lines=False):

    if fig == None:
        fig = plt.gcf()
    if ax == None:
        ax = plt.gca()
    
    # adjusting thickness of axis spines
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['top'].set_linewidth(2.0)
    
    # adjusting thickness of ticks
    ax.tick_params('both', width=2, which='major')
    
    # this positions the xtick labels so that the last letter is beneath the tick
    #fig.autofmt_xdate()
    
    if len(xlabels) != 0:
        
        ##
        if not isinstance(xlabels[0], str):
            ax.set_xticks(xlabels)   # placing the xticks
        else:
            ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels([str(i) for i in xlabels], rotation=0)
    
    ax.tick_params(axis='both', which='major', direction='out', 
                   labelsize=tick_size)     # adjusting tick label size
    
    ax.xaxis.label.set_fontsize(label_size)                             # adjusting x-fontsize
    ax.yaxis.label.set_fontsize(label_size)                             # adjusting y-fontsize
    ax.title.set_fontsize(label_size)
    
    # if a y-limit list is inputted, setting limits for y-axis
    if len(ylim) != 0:
        ax.set_ylim(ylim)
        
    if len(xlim) != 0:
        ax.set_xlim(xlim)
    
    # adding legend and prettifying figure
    if isinstance(legend_loc, str):
        plt.legend(loc=legend_loc, prop={'size':legend_size}, 
                   fancybox=True, framealpha=0.5, numpoints=1, frameon=0)
    else:
        plt.legend(prop={'size':legend_size}, bbox_to_anchor=legend_loc,
                   fancybox=True, framealpha=0.5, numpoints=1, frameon=0)    
  
    if all_axis_lines:
        remove_border(axes=ax, top=1, right=1)
    else:
        remove_border(axes=ax)
        
    
    
##########################################################################################


# procedure to turn a white plot into a black plot
# anything black on a white background becomes white on a black background
def black_plot_param(figure=None, axes=None, legend=False, loc='best'):
    
    fig = figure or plt.gcf()
    ax = axes or plt.gca()
    
    # resetting axis colors
    ax.set_axis_bgcolor('black')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    
    ax.tick_params(axis='both', colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    fig.patch.set_facecolor('black')
    
    if legend:
        # setting legend colors
        legend = ax.legend(loc=loc, prop={'size':18}, numpoints=1)
        frame = legend.get_frame()
        frame.set_facecolor('black')
        frame.set_edgecolor('white')
        
        for text in legend.get_texts():
            text.set_color('white')
    
    
##########################################################################################


# function to make box plot
def make_box_plot(df, quant_cols, box_color='k', xticks=[], yticks=[], ylim=[], xlabel=[], 
                  ylabel=[], title=[], fig=[], ax=[], xtick_angle=0, 
                  tick_size=14, label_size=14, all_axis_lines=True):
        
    data_array = np.array(df[quant_cols])
    data_list = [(data_array[:, c][~np.isnan(data_array[:, c])]).tolist() for c in np.arange(np.shape(data_array)[1])]
    
    fig_input = True
    if (isinstance(fig, list) & isinstance(ax, list)):
        fig_input = False
        fig, ax = plt.subplots(figsize=(5, 2.5))
    
    # patch_artist must be enabled to change patch colors
    bp = plt.boxplot(data_list, notch=0, sym='o', vert=1, whis=1.0, 
                     patch_artist=True)
    
    # get data attributes
    medians = []
    for i, line in enumerate(bp['medians']):
        y = line.get_xydata()[1][1] # top of median line
        medians.append(y)
        
    # getting inner quartile ranges
    IQRs = []
    for col in range(np.shape(data_array)[1]):
        
        y = data_array[:, col]
        upper_quartile = np.percentile(y[~np.isnan(y)], 75)
        lower_quartile = np.percentile(y[~np.isnan(y)], 25)
        IQR = upper_quartile-lower_quartile
        IQRs.append(IQR)
    
    # getting values at upper and lower caps
    low_caps = []
    for low_cap in bp['caps'][::2]:
        y = low_cap.get_xydata()[1][1] # y-value of cap
        low_caps.append(y)
        
    high_caps = []
    for high_cap in bp['caps'][1::2]:
        y = high_cap.get_xydata()[1][1] # y-value of cap
        high_caps.append(y)
    
    # make box_color a list equal in size to the number of boxes (if user hasn't done this already)
    if len(box_color) == 1:
        box_color = [box_color]*np.shape(data_array)[1]
    
    for idx, clr in enumerate(box_color):
        
        plt.setp(bp['boxes'][idx], color=clr, linewidth=1.5)
        plt.setp(bp['whiskers'][idx*2], color=clr, linewidth=1.5, linestyle='-')
        plt.setp(bp['whiskers'][idx*2+1], color=clr, linewidth=1.5, linestyle='-')
        plt.setp(bp['medians'][idx], color=clr, linewidth=1.5)
        plt.setp(bp['caps'][idx*2], color=clr, linewidth=1.5)
        plt.setp(bp['caps'][idx*2+1], color=clr, linewidth=1.5)
    
    # set internal box color to white
    for patch in bp['boxes']:
        patch.set(facecolor='w')
            
    # make fliers invisible
    w_list = ['none' for c in np.arange(np.shape(data_array)[1])]
    for f, fc in zip(bp['fliers'], w_list):
        f.set_markerfacecolor(fc)
        f.set_markeredgecolor('none')
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    if len(yticks) > 0:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)
    
    if xtick_angle == 45:
        ha = 'right'
    else:
        ha = 'center'
    
    if len(xticks) != 0:
        ax.set_xticklabels(xticks, rotation=xtick_angle, ha=ha)
    else:
        ax.set_xticklabels(quant_cols, rotation=xtick_angle, ha=ha)
        
    if len(title) > 0:
        ax.set_title('  '+title, loc='left')
    set_plot_params(ylim=ylim, tick_size=tick_size, label_size=label_size)
    
    if all_axis_lines:
    	yes_border_no_ticks()
        
    if not fig_input:
        plt.show()
        
    # output medians, IQRs, and values at 5th and 95th percentile
    return medians, IQRs, low_caps, high_caps
    
    
##########################################################################################


def make_dot_plot(df, quant_cols, sem_quant_cols=[], ylim=[], xticks=[], yticks=[], xlabel=[], 
                  ylabel=[], title=[], dot_color='b', fig=[], ax=[], dot_alpha=0.5, xtick_angle=0, 
                  tick_size=20, label_size=20, annot_med=True):
    
    
    # if we want to add error bars
    errorbar = False
    if len(sem_quant_cols) > 0:
        data_sem_array = np.array(df[sem_quant_cols])
        data_sem_list = [(data_sem_array[:, c][~np.isnan(data_sem_array[:, c])]).tolist() for c in np.arange(np.shape(data_sem_array)[1])]
        errorbar = True
        
    data_array = np.array(df[quant_cols])
    data_list = [(data_array[:, c][~np.isnan(data_array[:, c])]).tolist() for c in np.arange(np.shape(data_array)[1])]
    
    fig_input = True
    if (isinstance(fig, list) & isinstance(ax, list)):
        fig_input = False
        fig, ax = plt.subplots(figsize=(5, 2.5))
    
    # patch_artist must be enabled to change patch colors
    bp = plt.boxplot(data_list, notch=0, sym='o', vert=1, whis=1.0, 
                     patch_artist=True)
    
    
    medians = []
    for i, line in enumerate(bp['medians']):
        y = line.get_xydata()[1][1] # top of median line
        medians.append(y)
    
    line_color = 'k'
        
    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color=line_color, linewidth=1.5)
            
    plt.setp(bp['whiskers'], color=line_color, linewidth=1.5, linestyle='-')

    for patch in bp['boxes']:
        patch.set(facecolor='w')
            
    w_list = ['none' for c in np.arange(np.shape(data_array)[1])]
    for f, fc in zip(bp['fliers'], w_list):
        f.set_markerfacecolor(fc)
        f.set_markeredgecolor('none')

    IQRs = []
    # re-plotting actual data
    for col in range(np.shape(data_array)[1]):
        
        y = data_array[:, col]
        
        upper_quartile = np.percentile(y[~np.isnan(y)], 75)
        lower_quartile = np.percentile(y[~np.isnan(y)], 25)
        IQR = upper_quartile-lower_quartile
        IQRs.append(IQR)
        
        if annot_med:
            print quant_cols[col]
            print 'Median (IQR): %s (%s)' % (medians[col], IQR)
            print
        
        # this randomizes the x-value
        # still hovers in correct column though
        x = np.random.normal(col+1, 0.04, size=len(y))
        
        # zorder=1 puts plot in back; zorder=10 puts plot in front
        
        if len(dot_color) == np.shape(data_array)[1]:
            color = dot_color[col]
        else:
            color = dot_color[col]
        
        if len(y) > 50:
            markersize = 5
        else:
            markersize = 15
            
        if not errorbar:
            ax.plot(x, y, '.', color=color, zorder=10, alpha=dot_alpha, 
                    markeredgewidth=0.0, markersize=markersize)
        else:
            y_err = data_sem_array[:, col]
            
            ax.plot(x, y, '.', color=color, zorder=10, alpha=dot_alpha, 
                    markeredgewidth=0.0, markersize=markersize)
            ax.errorbar(x, y, yerr=y_err, fmt=' ', color=color, zorder=10, 
                        alpha=dot_alpha/5.0, markeredgewidth=1.0, linewidth=0.5)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    if len(yticks) > 0:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)
    
    if xtick_angle == 45:
        ha = 'right'
    else:
        ha = 'center'
    
    if len(xticks) != 0:
        ax.set_xticklabels(xticks, rotation=xtick_angle, ha=ha)
    else:
        ax.set_xticklabels(quant_cols, rotation=xtick_angle, ha=ha)
        
    if len(title) > 0:
        ax.set_title('  '+title, loc='left')
    set_plot_params(ylim=ylim, tick_size=tick_size, label_size=label_size)
        
    if not fig_input:
        plt.show()
        
    # output medians
    return medians, IQRs
    
    
##########################################################################################





