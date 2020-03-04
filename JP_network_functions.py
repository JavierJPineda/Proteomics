## This file contains functions useful for network/connection analysis (e.g. 2-D correlations)

# importing useful modules
import numpy as np
from numpy.linalg import *
import networkx as nx
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
import matplotlib.colors as cl
from matplotlib import ticker
import pylab as py

# scientific and computing stuff
import uniprot
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from itertools import combinations, permutations, product
import random
import re
import string
import glob
import os
import json
import ast

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

# import all of my useful functions
from JP_plotting_functions import *

##########################################################################################


# making function to make networkx graph for protein combinations
# 'corr_df' is the dataframe with all of the combinations
# 'desc_df' is the dataframe with family descriptions, etc
# 'anticorr' determines whether to treat anti-correlation separately
def network_graph(corr_df, desc_df, method='pearson', anticorr=True, 
                  only_coreg=False):
    
    ## filter combos section 
    if only_coreg == True:
        # only consider pairwise positive correlations
        # where there is a common motif between the peptides
        corr_df = corr_df[~(corr_df.common_motif.isnull())].reset_index(drop=True)
        corr_df = corr_df[corr_df[method] > 0].reset_index(drop=True)
    
    # only taking statistically significant combos
    if method.lower() == 'pearson':
        corr_df = corr_df[~(corr_df.p_pval >= 0.05)]
        
        # treating anti-correlation and correlation equally for phospho
        if anticorr == False:
            corr_df['pearson'] = corr_df.pearson.apply(lambda x: abs(x))
    else:
        corr_df = corr_df[~(corr_df.s_pval >= 0.05)]
        
        if anticorr == False:
            corr_df['spearman'] = corr_df.spearman.apply(lambda x: abs(x))  
        
    # list of genes/sites
    site_list = list(set(corr_df.site1.tolist() + corr_df.site2.tolist()))
    
    # getting family information
    
    ##
    
    if 'Gene_Site' in [str(i) for i in desc_df.columns]:
        desc_df = desc_df[desc_df.Gene_Site.isin(site_list)]
        desc_df = assign_fam_color(desc_df, hm_color='rkc')
    else:
        desc_df = desc_df[desc_df.Gene_Symbol.isin(site_list)]
        desc_df = assign_fam_color(desc_df, hm_color='rkc')
  
    
    G = nx.Graph()                           # initiating the graph
    
    ## node section
    for site in site_list:                   # adding nodes
        
        ##
        
        if 'Gene_Site' in [str(i) for i in desc_df.columns]:
            color = desc_df.fam_colors[desc_df.Gene_Site == 
                                            site].values[0]
        else:
            
            color = desc_df.fam_colors[desc_df.Gene_Symbol == 
                                            site].tolist()
            if len(color) == 0:
                color = (0.9, 0.9, 0.9)
            else:
                color = color[0]
        
        G.add_node(site, color=color)
        
    # adding new index column
    corr_df['idx'] = np.arange(len(corr_df))
    
    # adding edges
    for i in corr_df.idx:
        site1 = corr_df.site1[corr_df.idx == i].tolist()[0]
        site2 = corr_df.site2[corr_df.idx == i].tolist()[0]
        
        if method.lower() == 'pearson':
            meth_val = corr_df.pearson[corr_df.idx == i].tolist()[0]
        else:
            meth_val = corr_df.spearman[corr_df.idx == i].tolist()[0]
        
        weight = 0.5*meth_val # arbitrary scaling for asthetics
        
        ##
        if only_coreg == True:
            coreg_weight = corr_df.common_motif_num[corr_df.idx == i].tolist()[0]
        
            weight = weight * coreg_weight
        
        G.add_edge(site1, site2, weight=weight, difference=1./weight)
    
    return G


# written for analysis of phospho data
# 'anticorr' determine whether to treat anticorrelation different
# 'anticorr=False' means: take absolute values of correlations
def plot_network_graph(corr_df, desc_df, method='pearson', 
                       anticorr=True, include_site=True, 
                       iterations=20, hm_color='rkc', 
                       only_coreg=False, legend=True):
    
    # getting the network
    network = network_graph(corr_df, desc_df,
                           method=method, anticorr=anticorr, 
                           only_coreg=only_coreg)

    # this makes sure draw_spring results are the same at each call
    np.random.seed(1) 
    
    # getting node colors
    color = [network.node[site]['color'] for site in network.nodes()]

    # determine position of each node using a spring layout
    pos = nx.spring_layout(network, iterations=iterations)
    
    fig = plt.figure(figsize=(12, 12))
    
    # plot the edges
    nx.draw_networkx_edges(network, pos, alpha = 0.1, 
                           edge_color='k', width=2.0)
    
    # plot the nodes
    nodes = nx.draw_networkx_nodes(network, pos, node_color=color, 
                                   node_size=500, linewidths=0.5)
    
    # set node outline color to black
    nodes.set_edgecolor('k')
    
    # draw the labels
    labels_dict = {}
    for gene_site in network.nodes():
        
        if '_' in gene_site:
            gene = gene_site[:gene_site.index('_')]
        else:
            gene = gene_site
        
        labels_dict[gene_site] = gene
    
    if include_site == True:
        labels_dict = None
        
    labels = nx.draw_networkx_labels(network, pos, labels_dict, 
                                     alpha=5, font_size=9)
    
    # coordinate information is meaningless here, so let's remove it
    plt.xticks([])
    plt.yticks([])
    remove_border(left=False, bottom=False)
    
    ### adding color legend for protein family (AXES = ax6) ###
    if legend == True:
    
        temp_df = pd.DataFrame(network.nodes(), columns=['Gene_Symbol'])
        temp_df['Site_Position'] = temp_df.Gene_Symbol.apply(lambda x: 
                                                        x[x.index('_')+1:])
        temp_df['Gene_Symbol'] = temp_df.Gene_Symbol.apply(lambda x: 
                                                           x[:x.index('_')])
        
        # not using desc_df anymore
        desc_df = desc_df.drop_duplicates('Gene_Symbol')
        
        # getting family information back
        temp_df['Family'] = 'null'
        for row in range(len(temp_df)):
            gene = temp_df.Gene_Symbol[temp_df.index == row].tolist()[0]
            family = desc_df.Family[desc_df.Gene_Symbol == gene].tolist()[0]
            temp_df.Family[temp_df.index == row] = family
        
        # making small dataframe of unique families and colors
        unq_fam_df = pd.DataFrame(sorted(temp_df.Family.unique()), 
                                  columns=['Family'])
        unq_fam_df = assign_fam_color(unq_fam_df, hm_color=hm_color)
        unq_fam_colors = list(unq_fam_df.fam_colors)
        unq_fam_labels = list(unq_fam_df.Family)
        
        # numbers to plot to get nice bar graph
        colorx = np.arange(len(unq_fam_df))
        colory = np.ones(len(unq_fam_df))
        
        # setting axis for family legend
        pos1 = plt.gca().get_position()
        pos2 = [pos1.x0+pos1.width/6., pos1.y0 + pos1.height*3./4, 
                pos1.width/25., pos1.height/4.]
        ax = fig.add_axes(pos2)
    
        # plotting horizontal bar graph (i.e. vertical plot)
        ax.barh(colorx, colory, color=unq_fam_colors, 
                 height=1.0, edgecolor='none')
        ax.set_xticks([])
        ax.set_yticks(np.arange(len(colorx))+0.5)
        ax.set_yticklabels(unq_fam_labels)
        yes_border_no_ticks(ax)
    
    plt.show()
                                       

##########################################################################################
