## This file contains several useful statistical functions

# importing necessary modules
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
import matplotlib.colors as cl
from matplotlib import ticker
import pylab as py

# scientific and computing stuff
import uniprot
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from itertools import combinations, permutations, product
from sklearn import svm
import random
import re

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

# import my useful functions
from JP_plotting_functions import *


##########################################################################################


# making function to compute quantile normalization on multiple arrays
# 'array' columns are what will be qnormalized
# NOTE: input array must have floating values
def quantile_norm(data_array):
    
    data_array = np.array(data_array)     # convert input array to numpy array
    rows, cols = np.shape(data_array)     # get dimensions of input array
    
    # each col in rank_array will corresponding 
    # to the ranks of the col in data_array
    rank_array = np.zeros((rows, cols))
    for i in range(cols):
        
        # these two lines get the ranks of the values in column 'i'
        order_i = data_array[:, i].argsort()
        ranks_i = order_i.argsort()
        rank_array[:, i] = ranks_i
    
    rank_array = rank_array.astype(int)
        
    # sorting columns of the input array to get ordered row averages
    data_sorted = np.zeros((rows, cols))
    for i in range(cols):  
        
        data_sorted[:, i] = np.sort(data_array[:, i])
    rank_means = np.mean(data_sorted, axis=1)
    
    # generating qnormalized array
    qnorm_array = data_array
    for i in range(cols):
        
        idx = rank_array[:, i]
        qnorm_array[:, i] = rank_means[idx]
        
    return qnorm_array


##########################################################################################


# making function to calculate correlations between 2 or more arrays
# 'arrays' is a list of arrays
# 'arrays' can also be a numpy array with columns to be compared
def multi_corr(arrays, corr_type='spearman', matrix=True):
    
    # if a numpy array is inputted, get transpose, then convert to list
    if isinstance(arrays, np.ndarray):
        arrays = arrays.T.tolist()
    
    # number of arrays
    num_arrays = len(arrays)
    
    # gives a list of tuples of combinations
    array_combos = list(combinations(np.arange(num_arrays), 2))
    
    # intializing correlation and p-value arrays
    corr_list = np.zeros((1, len(array_combos)))
    pval_list = np.zeros((1, len(array_combos)))
    
    # getting correlation for each combination
    for i in range(len(array_combos)):
        
        array_combo_i = array_combos[i]     # combo 'i'
        
        # getting spearman or pearson of the relevant array combination
        if corr_type.lower() == 'spearman':
            r, p = spearmanr(arrays[array_combo_i[0]], arrays[array_combo_i[1]])
            
        elif corr_type.lower() == 'pearson':
            r, p = pearsonr(arrays[array_combo_i[0]], arrays[array_combo_i[1]])
        
        # recording correlation coefficient and p-value
        corr_list[:, i] = r
        pval_list[:, i] = p
    
    # building corr matrix
    if matrix == True:
        corr_matrix = np.eye(len(arrays))
        pval_matrix = np.eye(len(arrays))
        
        for i in range(len(array_combos)):
            
            array_combo_i = array_combos[i]
            
            # placing correlation coefficients in proper positions
            corr_matrix[array_combo_i[0], array_combo_i[1]] = corr_list[:, i]
            corr_matrix[array_combo_i[1], array_combo_i[0]] = corr_list[:, i]
        
            pval_matrix[array_combo_i[0], array_combo_i[1]] = pval_list[:, i]
            pval_matrix[array_combo_i[1], array_combo_i[0]] = pval_list[:, i]
        
        # outputting symmetrical correlation matrix
        return corr_matrix, pval_matrix
        
    else:
        # outputting list of correlation coefficients
        return corr_list[0], pval_list[0]


##########################################################################################


# making function to compute spearman distance
def spdist(a, b, ranked=False):
    
    if isinstance(a, list):
        a = np.array(a)
        
    if isinstance(b, list):
        b = np.array(b)
    
    if not ranked:
        rank_a = rankdata(a)*1.0          # ranking data
        rank_b = rankdata(b)*1.0
    
    rank_a = rank_a - np.mean(rank_a)     # subtracting means
    rank_b = rank_b - np.mean(rank_b)
    
    dot_a = np.dot(rank_a, rank_a.T)      # getting self and inter dot products
    dot_b = np.dot(rank_b, rank_b.T)
    dot_ab = np.dot(rank_a, rank_b.T)
    
    # computing spearman distance
    dist = 1. - dot_ab/np.sqrt(dot_a)/np.sqrt(dot_b)
    
    return dist


# here a and b are both 2d arrays
# corresponding rows of a and b are the array combinations
def spdist2d(a, b):

    assert np.shape(a) == np.shape(b)
    
    rows, cols = np.shape(a)
    
    # 100 iterations each with a 50000-row dataframe takes ~30 min to run
    eff_num = 50000
    
    if len(a) > eff_num:
        
        num_sub = rows*1.0 / eff_num
        num_sub_c = np.ceil(num_sub)

        spdist_array = np.zeros(rows)
        
        print 'Will need to compute spdist over %i iteration(s).\n' % num_sub_c

        start = time.time()
        
        # iterating through every 50,000 rows of a and b
        for i in range(int(num_sub_c)):
            
            print 'Iteration %i' % (i+1)
    
            # if it's the last iteration
            if i == num_sub_f - 1:
                sub_a = a[eff_num*i:, :]
                sub_b = b[eff_num*i:, :]
            
            else:
                sub_a = a[eff_num*i:eff_num*(i+1), :]
                sub_b = b[eff_num*i:eff_num*(i+1), :]
    
            sub_rows, sub_cols = np.shape(sub_a)
    
            # these indices are the same for every iteration
            sub_df = pd.DataFrame(np.arange(sub_rows), columns=['idx'])
            
            sub_df['a'] = sub_df.idx.apply(lambda x: sub_a[x, :])
            sub_df['b'] = sub_df.idx.apply(lambda x: sub_b[x, :])
                                                
            sub_df['spdist'] = sub_df[['a', 'b']].apply(lambda x: spdist(x[0], x[1]), axis=1)
            
            if i == num_sub_f - 1:
                spdist_array[eff_num*i:] = np.array(sub_df.spdist)
                
            else:
                spdist_array[eff_num*i:eff_num*(i+1)] = np.array(sub_df.spdist)
    
            if i == 0:
                end = time.time()
                
                loop_time = (end-start)/60.     # minutes
                total_time = str(loop_time*num_sub_c)
                print 'Estimated time to complete job: %s minutes' % total_time
    
    
    # if the input arrays have less than 50,000 rows each
    else:
        df = pd.DataFrame(np.arange(rows), columns=['idx'])
        df['a'] = df.idx.apply(lambda x: a[x, :])
        df['b'] = df.idx.apply(lambda x: b[x, :])
        
        df['spdist'] = df[['a', 'b']].apply(lambda x: spdist(x[0], x[1]), axis=1)
        spdist_array = np.array(df.spdist)
        
    return spdist_array
        
    
# making function that will take an input array,
# generate combinations of the columns and compute spdist for each combination
# order will be preserved in the output
def combos_spdist(df_array):

    df_array = df_array.T     # taking transpose
    
    # num_events (number of rows); num_items (number of columns)
    num_events, num_items = np.shape(df_array)
    
    # generating list of combination tuples
    idx_combos = list(combinations(np.arange(num_items), 2))
    
    # putting combinations in column of dataframe
    combos_df = pd.DataFrame(np.arange(len(idx_combos)), columns=['idx'])
    combos_df['combos'] = idx_combos
    
    # creating separate columns for the arrays corresponding to the combinations
    combos_df['a'] = combos_df.combos.apply(lambda x: df_array[:, x[0]].tolist())
    combos_df['b'] = combos_df.combos.apply(lambda x: df_array[:, x[1]].tolist())
    
    # getting final input arrays
    a = np.array(sum(list(combos_df.a), [])).reshape((len(idx_combos), num_events))
    b = np.array(sum(list(combos_df.b), [])).reshape((len(idx_combos), num_events))
    
    # computing spdist for all combinations
    spdist_array = spdist2d(a, b)
    
    return spdist_array


##########################################################################################


# making PCA function
# 'quant_cols' will be a list of the columns with the desired values
def PCA_func(df, quant_cols, standardize=False):

    data = np.array(df[quant_cols])
    M, N = np.shape(data)   # M = num of rows     (i.e. num of dimensions/proteins)
                            # N = num of columns  (i.e. num of "trials"/samples)
    # getting mean of quant col vals across a row
    avg = np.mean(data, axis=1)
    
    # np.tile() makes a tile of the input matrix (like MATLAB's repmat())
    data = data - np.tile(avg, [len(quant_cols), 1]).T
    
    # divide by stdev
    if standardize:
        std = np.std(data, axis=1)
        data = data / np.tile(std, [len(quant_cols), 1]).T
    
    # getting covariance matrix (i.e. proteins X proteins)
    covar = 1./(N-1.) * np.dot(data, data.T)
    
    # getting eigenvectors and eigenvalues
    # eigvals is a vector of eigenvalues that correspond to the respective 
    # cols in eigvecs (each column in eigvecs corresponds to an eigenvector)
    eigvals, eigvecs = np.linalg.eig(covar)
    
    # NOTE: the order of eigenvalues in 'eigvals' is the same order as the 
    # proteins in the input dataframe
    # sort the eigenvalues (i.e. variance), but keep the original index
    eigvals = pd.Series(eigvals)
    
    # getting indices of original positions in unordered eigvals
    # will correspond to the index positions of proteins in the input dataframe
    # this list of indices corresponds to variances in decreasing order
    eigvals_sort_idx = np.argsort(eigvals)[::-1].values
    
    eigvals_sort = np.sort(eigvals)[::-1]    # actual sorted eigenvalues
    
    # sorting eigenvector cols in 'eigvecs' matrix to reflect decreasing variance
    # i.e. in the new, sorted 'eigvecs' variance is decreasing top-to-bottom
    eigvecs_sort = eigvecs[:, eigvals_sort_idx] 
    
    # projecting the original data set onto the eigenspace
    signals = np.dot(eigvecs_sort.T, data)
    
    return eigvals_sort, eigvecs_sort, signals, covar


# making function to do PCA and output a dataframe
# this is useful if PCA is the end goal
# 'num_axes' is the desired number of components to include in the dataframe
# may need to be adapted to differently formatted data
def get_PCA_df(df, quant_cols, loading_col='Gene_Symbol', num_axes=3, standardize=True):
    
    # first, remove rows that have NaNs
    df = df.ix[~df[quant_cols].apply(lambda x: np.any(np.isnan(x)), axis=1), :]
    
    print 'Number of initial loadings: %s' % str(len(df))

    eigvals_sort, eigvecs_sort, signals, cov_mat = PCA_func(df, quant_cols, standardize=standardize)
    
    # getting protein loadings of principal axes
    # NOTE: for eigenvector v=[a, b], corresponding equation is C = a*x_1 + b*x_2
    PCA_df = pd.DataFrame()
    
    cols = ['Loading', 'Loading_Val', 
            'Component', 'Eigenvalue', 'Var_Explained', 'Projected_Data']
    
    for i in range(num_axes):
        
        proj_data = signals[i].real

        # getting the i_th principal component 
        compi = eigvecs_sort[:, i]
        
        # percent of variance explained by component i
        var_explained = abs(eigvals_sort[i] / sum(eigvals_sort))
        
        compi_df = pd.DataFrame(list(df[loading_col]), columns=['Loading'])
        compi_df['Loading_Val'] = compi.real/sum(compi.real)
        compi_df['Component'] = int(i+1)
        
        # including eigenvalues; 
        # should meet eigval > 1 criterion
        compi_df['Eigenvalue'] = eigvals_sort[i].real
        
        # including percent variance derived from eigenvalues; 
        # should meet var% > 5% criterion
        compi_df['Var_Explained'] = var_explained
        
        # including projected data
        compi_df['Projected_Data'] = 0.
        compi_df['Projected_Data'] = compi_df.Projected_Data.apply(lambda x: proj_data)
        compi_df['abs_loading_val'] = abs(compi)     # temporary for sorting purposes
        
        compi_df = compi_df.sort('abs_loading_val', ascending=False).reset_index(drop=True)
        compi_df = compi_df.drop('abs_loading_val', 1)
        
        PCA_df = pd.concat([PCA_df, compi_df], ignore_index=True)
    
    PCA_df = PCA_df[cols]
        
    return PCA_df
    
    
# function that plots one component of PCA
def plot_PCA_1d(pca_df, comp=1, proj_labels=[], xticks=[]):
    
    ## plot PCA projection
    
    var_perc = int(np.around(pca_df.Var_Explained[pca_df.Component == comp].values[0], 2)*100)
    
    proj = pca_df.Projected_Data[pca_df.Component == comp].values[0]
    
    if len(proj_labels) > 0:
        labels = proj_labels
    else:
        labels = ['' for i in range(len(proj1))]
        
    colors = [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0, 0), (1.0, 0, 0), (1, 0.5, 0), (0, 0, 1), (0, 0.5, 1)]
        
    plt.figure(figsize=(5, 0.5))
    for i in range(len(proj)):
        plt.plot(proj[i], [1], 'o', color=colors[i], markersize=10, markeredgewidth=1.0, label=labels[i])
    
    
    plt.xlabel('PC'+str(comp)+' ('+str(var_perc)+'%)')
    
    if len(xticks) > 0:
        plt.xlim([xticks[0], xticks[-1]])
        plt.gca().set_xticks(xticks)
        
    plt.gca().set_yticks([])
    set_plot_params(tick_size=20, label_size=20)
    yes_border_no_ticks()
    
    plt.legend(fancybox=True, framealpha=0.5, numpoints=1, bbox_to_anchor=(1.6, 1.01), frameon=0)
    plt.show()


# making procedure to plot projected data from PCA
# num_comps is the number of principal components to include
# makes sure to include the x-axis labels for the plot
def plot_PCA_1(PCA_df, quant_labels, num_comps=2, nload=15, xticks=[], 
             legend=False, bg=[]):
    
    max_load_num = len(PCA_df.Loading.unique())
    
    if nload > max_load_num:
        print '''
        There are fewer than %i loadings\n
        Displaying all loadings (%i)
        ''' % (nprot_load, max_load_num)
        
        nload = max_load_num
    
    if legend:
        # setting colors and creating figure for loadings plot
        # blue, red, green, magenta, black]  
        
        # this will be [dark blue, light blue, dark red, light red, etc...]
        colors = [(0, 0, 0.6), (0, 0, 1), (0.6, 0, 0), (1, 0, 0), (0, 0.6, 0),
                  (0, 1, 0), (0.6, 0, 0.6), (1, 0, 1), (0, 0, 0), (0.6, 0.6, 0.6)]
            
        if bg == 'black':
            colors = colors[:-2] + [(0.6, 0.6, 0.6), (1, 1, 1)]
            
        colors = colors[:len(quant_labels)]
    
    # figsize=(width, height)
    fig, axes = plt.subplots(figsize=(8, 8), nrows=1, ncols=num_comps)
    pos = np.arange(nload)+0.5  # want the bar graph to center on the y-axis
    
    # getting projected data
    pca_data = np.zeros(shape=(len(PCA_df.Projected_Data[0]), num_comps))
    for i in range(num_comps):
        
        # turning component values into series
        compi_series = PCA_df.Projected_Data[PCA_df.Component == i+1].reset_index(drop=True)
        xi = np.array(compi_series[0]).T    # these are the projected data values
        pca_data[:, i] = xi                 # inserting projected data vector as column
        
        # plotting loadings
        loading_vals = list(PCA_df.Loading_Val[PCA_df.Component == i+1].head(nload).values)
        loading_labels = list(PCA_df.Loading[PCA_df.Component == i+1].head(nload))

        # setting color of reference line
        if bg == 'black':
            ref_col = 'w-'
        else:
            ref_col = 'k-'
            
        axes[i].barh(pos, loading_vals, color='b', align='center', edgecolor='none')
        
        xmin, xmax, ymin, ymax = axes[i].axis()     # getting axis limits
        
        # adding x=0 line for reference
        x_ref = np.zeros((ymax+1)); y_ref = np.arange(ymax+1)
        axes[i].plot(x_ref, y_ref, ref_col, linewidth=1.0)
        
        # setting x-axis limits
        
        # only want 5 x-ticks
        if (xmin < 0) & (xmax > 0):        
            # there are positive and negative values
            x_max_abs = max([abs(xmin), abs(xmax)])
            
            axes[i].set_xlim([-x_max_abs, x_max_abs])
            axes[i].set_xticks([-x_max_abs, -x_max_abs/2., 0, x_max_abs/2., x_max_abs])
            axes[i].set_xticklabels([-x_max_abs, -x_max_abs/2., 0, x_max_abs/2., x_max_abs])
        
        elif (xmin == 0) & (xmax > 0):
            # there are only positive values
            axes[i].set_xticks([0, xmax/4., xmax/2., xmax*3./4, xmax])
            axes[i].set_xticklabels([0, xmax/4., xmax/2., xmax*3./4, xmax])
            
        elif (xmin < 0) & (xmax == 0):
            # there are only negative values
            axes[i].set_xticks([xmin, xmin*3./4, xmin/2., xmin/4, 0])
            axes[i].set_xticklabels([xmin, xmin*3./4, xmin/2., xmin/4, 0])
        
        
        axes[i].set_yticks(np.arange(nload)+0.5)
        axes[i].set_yticklabels(loading_labels)
        axes[i].set_xlabel('Loading Value')
        axes[i].xaxis.label.set_fontsize(17)
        axes[i].invert_yaxis()     # making sure longer bars are on top
        
        set_plot_params()
        remove_border()
        yes_border_no_ticks(axes[i])
        
        
        var_exp = str(round(PCA_df.Var_Explained[PCA_df.Component == i+1].unique()[0], 3)*100)
            
        # will also include eigenvalue in output plot title
        eigval = str(round(PCA_df.Eigenvalue[PCA_df.Component == i+1].unique()[0], 1))
            
        loadings_title = '''
        Principal Component %i:
        Variance Explained = %s%%
        Eigenvalue = %s\n
        ''' % (i+1, var_exp, eigval)

        if bg != 'black':
            axes[i].set_title(loadings_title, loc='left')
            axes[i].title.set_fontsize(17)
            
        else:
            axes[i].set_title(loadings_title, loc='left', color='white')
            axes[i].title.set_fontsize(17)
            black_plot_param(fig, axes[i])
            
        # for some reason, when I do the above, it leaves the top and bottom spines
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

        
    fig.tight_layout()   # making sure plot labels are not overlapping
    plt.show()

    # plotting projected data
    for j in range(num_comps):
        x = np.arange(len(quant_labels))+1
        y = pca_data[:, j]      # 1st principal component axis    
        plt.figure(j)
        
        for k in range(len(x)):
            
            if legend:
                plt.plot(x[k], y[k], 'o', color=colors[k], markersize=10, label=quant_labels[k])
            else:
                plt.plot(x[k], y[k], 'bo', markersize=10)
        
        plt.xlim([-0.5, len(quant_labels)+0.5])
        plt.ylim([min(y)-0.15*abs(min(y)), max(y)+0.15*abs(max(y))])
        
        #plt.xlabel('Samples', fontsize=17)
        if len(xticks) > 0:
            plt.xticks(x, xticks)
        else:
            plt.xticks(x, quant_labels, rotation=90)  
            
        plt.ylabel('Principal Component %s' % str(j+1), fontsize=17)
        
        var_exp = PCA_df.Var_Explained[PCA_df.Component == j+1].unique()

        var_expj = str(round(var_exp[0], 3)*100)
        plt.title('Variance Explained: %s%%\n\n' % var_expj) 
        plt.legend(loc='best', numpoints=1, prop={'size':12})
        remove_border()
        
        if bg == 'black':
            black_plot_param(legend=True)
        plt.show()
        
        
def plot_PCA_2(pca_df, comps=[1, 2], proj_labels=[]):
    
    ## plot PCA projection
    
    var_perc1 = int(np.around(pca_df.Var_Explained[pca_df.Component == comps[0]].values[0], 2)*100)
    var_perc2 = int(np.around(pca_df.Var_Explained[pca_df.Component == comps[1]].values[0], 2)*100)
    
    proj1 = pca_df.Projected_Data[pca_df.Component == comps[0]].values[0]
    proj2 = pca_df.Projected_Data[pca_df.Component == comps[1]].values[0]
    
    if len(proj_labels) > 0:
        labels = proj_labels
    else:
        labels = ['' for i in range(len(proj1))]
        
    colors = [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0, 0), (1.0, 0, 0), (1, 0.5, 0), (0, 0, 1), (0, 0.5, 1)]
        
    plt.figure(figsize=(5, 5))
    for i in range(len(proj1)):
        plt.plot(proj1[i], proj2[i], 'o', color=colors[i], markersize=10, markeredgewidth=1.0, label=labels[i])
    
    
    plt.xlabel('PC'+str(comps[0])+' ('+str(var_perc1)+'%)')
    plt.ylabel('PC'+str(comps[1])+' ('+str(var_perc2)+'%)')
        
    set_plot_params(tick_size=20, label_size=20)
    yes_border_no_ticks()
    
    plt.legend(fancybox=True, framealpha=0.5, numpoints=1, bbox_to_anchor=(1.6, 1.01), frameon=0)
    plt.show()
        

##########################################################################################


# making core fuzzy c-means clustering function    
def fcmc_core(data_array, clust_num, exp_m=2., 
              remove_empty=True, print_jm=False):
    
    # "rows" corresponds to data variables
    # "cols" corresponds to features of the variabes
    data_vars, feat_cols = np.shape(data_array)
    
    max_iter = 100                         # maximum number of iterations
    min_imp = 1.0e-05                      # minimum amount of improvement
    
    # initializing while-loop
    # when no more empty clusters, loop will terminate
    
    ##
    empty_clust = 100
    while empty_clust > 0:
    
    ##
        # generating random initial partition matrix
        u_array = np.random.uniform(size=(clust_num, data_vars))
        
        # making columns sum to unity
        ucol_sums = np.sum(u_array, axis=0)                     
        u_array = u_array / np.tile(ucol_sums, (clust_num, 1))

        # re-setting number of clusters according to updated u_array
        clust_num = np.shape(u_array)[0]
    
        # iterating over procedure until convergence
        jm_funcj = np.zeros(max_iter)
        for j in range(max_iter):
                
            weights = np.array(u_array) ** exp_m         # raising each element to exponent "m"
            
            # getting the ith cluster center
            # this will produce a "clust_num" x "feat_cols" sized array
            clust_cent = np.dot(weights,  # getting the ith cluster center
                      data_array) / np.array(np.matrix(np.ones(feat_cols)[:, 
                      np.newaxis])*np.matrix(np.sum(weights.T, axis=0))).T
            
    
            # getting squared distance matrix (using euclidean norm, i.e. identity matrix)
            sq_dist = np.zeros((clust_num, data_vars))
            for k in range(clust_num):
                  
                sq_dist[k, :] = np.sqrt(np.sum((data_array - np.array(np.matrix(np.ones(data_vars)[:, 
                                np.newaxis])*np.matrix(clust_cent[k, :])))**2, axis=1)[:, 
                                np.newaxis]).reshape((1, data_vars))
         
            # getting value of least-squared errors functional
            jm_funcj[j] = np.sum((sq_dist**2.) * weights)
            
            
            # if user wants to see results displayed
            if print_jm:
                if j<9:
                    j_str = ' '+str(j+1)
                else:
                    j_str = str(j+1)
                print 'Iteration %s: Functional = %f' % (j_str, jm_funcj[j])
            
            # making a new partition matrix based on the residual distances
            dist_m = sq_dist**(-2./(exp_m-1.))  # distance residuals raised to exponent "m"
            u_array = dist_m / np.array(np.matrix(np.ones(clust_num)[:, 
                    np.newaxis])*np.matrix(np.sum(dist_m, axis=0)))
    
            # checking for termination condition; minimum improvement
            if j > 1:
                if np.abs(jm_funcj[j] - jm_funcj[j-1]) < min_imp:
                    
                    jm_output = jm_funcj[j]
                    break
                    
                # last iteration
                elif j == max_iter-1:
                    jm_output = jm_funcj[j]
                    
        # if user wants to keep empty clusters, exiting loop
        if remove_empty == False:
            empty_clust = 0
            
        else:
            # NOTE: if no membership scores for a cluster are above 0.5,
            # the cluster is considered "empty"
            empty_cutoff = 0.5
            
            nonempty_clust = np.where(u_array > 0.5)[0]
            nonempty_clust = len(list(set(nonempty_clust)))
            
            # resetting cluster num to num of non-empty clusters + 1
            # i.e. N_new = N_old - (N_empty - 1) to be safe
            # unless there is only one empty cluster,
            # then N_new = N_old - N_empty
            empty_clust = clust_num - nonempty_clust
            
            # resetting cluster number
            if empty_clust > 1:
                clust_num = nonempty_clust + 1
            elif empty_clust == 1:
                clust_num = nonempty_clust
            else:
                pass   
                
    return u_array, jm_output
    
    
# making function to implement fuzzy c-means clustering
def fuzzy_cmc(data_array, clust_num=20, exp_m=2., 
              remove_empty=False, print_jm=False):
    
    # getting u_array and functional associated with that cluster number
    u_array, jm_output = fcmc_core(data_array, clust_num, exp_m, 
                                   remove_empty, print_jm)
        
    clust_assign = np.argmax(u_array, axis=0)    # top cluster for each protein
    clust_score = np.max(u_array, axis=0)        # protein cluster scores
    
    return u_array, clust_assign, clust_score


def get_fcmc_data(data_array, u_array, clust_assign, clust_score, 
                  plot_thresh=0.5, log_data=True, prot_data=True, keys=[], 
                  key_col='Gene_Symbol', data_type='time', color_code=True, bg='white'):
    
    '''
    For "data_type" input, user can input a list of lists with x-axis label and xtick labels.
    For example, ['Tissue', ['B', 'K', 'L', 'M', 'S']]
    '''
    
    ### organizing data section ###
    
    # making dataframe for partition matrix
    u_df = pd.DataFrame(u_array.T)
    u_df['Cluster'] = clust_assign     # this column must come right after the values
    
    # making dataframe of data array
    data_df = pd.DataFrame(data_array)
    data_df['Cluster'] = u_df.Cluster
    data_df['Cscore'] = clust_score
        
    clust_num = len(u_df.Cluster.unique())     # number of clusters
    cell_states = np.shape(data_array)[1]      # number of cell_states
    
    # adding keys
    data_df[key_col] = keys
    
    
    ### presetting figure and axes section ###
    
    # setting figure size according to number of clusters
    # there must be more than 4 clusters for the axes specification to work
    if clust_num < 4:
        print 'Inputted cluster number must be 4 or greater.'
        print 'Using 4 as the cluster number.'
        clust_num = 4
        fig_size = (5, 5)
    
    elif clust_num == 4:
        fig_size = (5, 5)
    
    elif (clust_num > 4) & (clust_num <= 8):
        fig_size = (10, 5)
    
    elif clust_num == 9:
        fig_size = (8, 7.5)
    
    elif (clust_num > 8) & (clust_num <= 12):
        fig_size = (10, 7.5)
    
    elif (clust_num > 12) & (clust_num <= 16):
        fig_size = (10, 10)
    
    elif (clust_num > 16) & (clust_num <= 20):
        fig_size = (10, 12)
        
    elif clust_num > 20:
        fig_size = (10, 13)

    
    # if clust_num is 4, 9, or 16
    if (np.sqrt(clust_num) == int(np.sqrt(clust_num))) & (clust_num < 25):
        
        ncols = int(np.sqrt(clust_num))
        nrows = ncols
        
        # setting figure and axes
        # n_axes is a 2D nested tuple
        # for figsize, 1st number affects wideness, 2nd number affects tallness
        
        fig, axes_tup = plt.subplots(figsize=fig_size, nrows=nrows, ncols=ncols)
        
        # turning the nested axes tuple into one list
        axes = sum([list(i) for i in list(axes_tup)], [])
        
    # if clust_num is a multiple of 4, make rows of 4 plots each
    elif clust_num/4. == int(clust_num/4.):
        
        ncols = 4
        nrows = int(clust_num / ncols)
    
        fig, axes_tup = plt.subplots(figsize=fig_size, nrows=nrows, ncols=ncols)
        
        # turning the nested axes tuple into one list
        axes = sum([list(i) for i in list(axes_tup)], [])
    
    # if clust_num is not a multiple of 4, round up to the next multiple of 4
    else:
        
        ncols = 4
        nrows = int(np.ceil(clust_num/4.))
    
        fig, axes_tup = plt.subplots(figsize=fig_size, nrows=nrows, ncols=ncols)
            
        
        # turning the nested axes tuple into one list
        axes = sum([list(i) for i in list(axes_tup)], [])
        
        # make unnecessary axes invisible
        unn_axes = ncols*nrows - clust_num     # number of unnecessary axes
        
        for i in range(unn_axes):
            
            ax = axes[i+clust_num]
            remove_border(ax, left=False, right=False, top=False, bottom=False)
            ax.set_xticks([])
            ax.set_yticks([])
     
    
    # getting proteins whose cscores are above the plot threshold
    plot_prots = data_df[data_df.Cscore > plot_thresh]
    plot_prots_array = np.array(plot_prots.iloc[:, :cell_states])
    
    # getting maximum range in the plotting subset
    min_array = np.min(plot_prots_array, axis=1)    
    max_array = np.max(plot_prots_array, axis=1)
    
    min_val = np.min(min_array)
    max_val = np.max(max_array)
    max_val_round = 0.5*np.ceil(max_val*2)

    # for plots
    rot = 0  # rotation for xticklabels
    if isinstance(data_type, str):
        xlabel = 'Time Point'
        
        xticklabels = [str(i+1) for i in range(cell_states)]
    else:
        xlabel = data_type[0]
        xticklabels = data_type[1]

        for l in xticklabels:
            if len(l) > 2:
                rot = 90
        
     
    for c in sorted(u_df.Cluster.unique()):
        
        # taking cluster subset
        sub_u_df = u_df[u_df.Cluster == c]
        sub_data_df = data_df[data_df.Cluster == c]
    
            
        # plotting clusters
        #axes[c].set_title('Cluster '+str(c+1)+'\n')
        remove_border(axes[c])
        
        # making a yellow, green, cyan, blue, magenta, red color map        
        cmap = cl.LinearSegmentedColormap.from_list(name='y_g_c_b_m_r', 
                                                 colors =[(1.0, 1.0, 0), 
                                                          (0, 0.95, 0), 
                                                          (0, 0.95, 0.95),
                                                          (0, 0.05, 1.0),
                                                          (0.9, 0.0, 0.9),
                                                          (1, 0, 0)], N=101)
        
        
        
        # plot lower cluster scores first
        sub_data_df = sub_data_df.sort_values('Cscore').reset_index(drop=True)
        
        cscore_max = sub_data_df.Cscore.max()  # record max cscore
        
        for prot in range(len(sub_data_df)):
            x = np.arange(cell_states)
            y = np.array(sub_data_df.iloc[prot, :cell_states])
            
            y = y.tolist()
            
            # getting cluster score in order to assign color
            cscore_val = sub_data_df.Cscore[sub_data_df.index == prot].tolist()[0]

            if cscore_val < plot_thresh:
                continue
            
            
            ## fixing color assignment for different plot_thresholds
            # reciprical of (1 - plot_threshold)
            recip_thresh = 1.0 / (1.0 - plot_thresh*1.0) 
            
            if color_code:
                axes[c].plot(x, y, color=cmap(cscore_val*recip_thresh - recip_thresh + 1.0), linewidth=1.0)
            else:
                if cscore_val == cscore_max:
                    axes[c].plot(x, y, color=(0, 0, 0), linewidth=2.0)
                else:
                    axes[c].plot(x, y, color=(0.6, 0.6, 0.6), linewidth=1.0)
            
        xmin, xmax, ymin, ymax = axes[c].axis()          # getting axes limits
        
        axes[c].set_yticks([-max_val_round, -(max_val_round)/2., 
                            0, (max_val_round)/2., max_val_round])
        axes[c].set_ylim([min_val, max_val_round])
        axes[c].set_xticks(np.arange(cell_states))

        if len(xticklabels) <= 11:
            axes[c].set_xticklabels(xticklabels, rotation=rot)
        else:
            axes[c].set_xticklabels([])
        axes[c].set_xlabel(xlabel)
        axes[c].xaxis.label.set_fontsize(12)
        axes[c].tick_params(axis='y', which='major', labelsize=9)
        
        if log_data:
            axes[c].set_ylabel('Log'+r'$_{2}$'+'(FC)')
        else:
            axes[c].set_ylabel('Fold Change')
        
        
        axes[c].yaxis.label.set_fontsize(12)
        axes[c].set_title('Cluster '+str(c+1))
        axes[c].title.set_fontsize(12)


    # adding color legend and histogram of cluster distribution
    fig_b, axes_b = plt.subplots(nrows=1, ncols=2)
    
    ax1b, blank = list(axes_b)
    
    # making blank axis invisible
    remove_border(blank, left=False, right=False, top=False, bottom=False)
    blank.set_xticks([])
    blank.set_yticks([])

    
    # not including proteins with cluster membership lower than 0.2 in histogram
    trunc_data_df = data_df[data_df.Cscore >= 0.2].reset_index(drop=True)
    
    # plotting histogram
    counts, bins, patches = ax1b.hist(trunc_data_df.Cluster, 
                                      bins=clust_num, color='k', edgecolor='none')
    
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]  # centering bins
    
    ax1b.spines['left'].set_linewidth(3.0)
    ax1b.spines['bottom'].set_linewidth(3.0)
    ax1b.tick_params('both', width=2, which='major')
    ax1b.set_xticks(bin_centers)
    ax1b.set_xticklabels(np.arange(clust_num)+1)
    ax1b.set_xlim([-0.25, ax1b.axis()[1]+0.25])
    ax1b.set_xlabel('Cluster')
    
    if prot_data:
        ax1b.set_ylabel('Proteins')
    else:
        ax1b.set_ylabel('Peptides')
   
    ax1b.tick_params(axis='both', which='major', labelsize=15)
    ax1b.xaxis.label.set_fontsize(25)
    ax1b.yaxis.label.set_fontsize(25)
    ax1b.set_title('Distribution')
    ax1b.title.set_fontsize(25)
    remove_border(ax1b)

    ### plotting color bar section ###
    
    if color_code:
        pos1b = ax1b.get_position()
        cbar_pos = [pos1b.x0+pos1b.width*1.5, pos1b.y0+pos1b.height/2.,
                 pos1b.width/10., pos1b.height/2.*0.95]

        ax2b = fig_b.add_axes(cbar_pos)


        # plotting vertical bar graph
        fill = np.tile(np.linspace(0, 1), (1, 1))
        ax2b.pcolor(fill.T, cmap=cmap, vmin=0, vmax=1)


        ax2b.set_xticks([])

        # setting yticks (0-to-1)
        xmin, xmax, ymin, ymax = ax2b.axis()
        yrange = ymax-ymin 

        yticks = [ymin, ymin+yrange/5., ymin+yrange*2./5, 
                  ymin+yrange*3./5, ymin+yrange*4./5, ymax]

        ax2b.set_yticks(yticks)

        color_step = (1.0 - plot_thresh) / 5.
        color_scale = np.around(np.arange(plot_thresh, 1.0001, color_step), 2).astype(str)

        ax2b.set_yticklabels(color_scale)
        ax2b.set_title('Cluster Score\n')
        yes_border_no_ticks(ax2b)
        ax2b.tick_params(axis='both', which='major', labelsize=10)
    
    if bg == 'black':
        for i in axes:
            black_plot_param(fig, i)
            
        black_plot_param(fig_b, ax1b)
        
        if color_code:
            black_plot_param(fig_b, ax2b)
        
        # changing histogram colors
        black_plot_param(fig_b, blank)
        for patch in patches:            
            patch.set_facecolor((0, 0.75, 0))   # changing face color for each bin
            patch.set_edgecolor('white')
            
    fig.tight_layout()
    fig_b.tight_layout()
    
    
    # building output dataframe
    if len(key_col) > 0:
        output_df = data_df[[key_col, 'Cluster', 'Cscore']]
        output_df.Cluster = np.array(output_df.Cluster.tolist())+1
        output_df = output_df.sort_values(['Cluster', 'Cscore'], 
                                   ascending=[1, 0]).reset_index(drop=True)
    
        # outputting cluster assignments
        return output_df
    
    else:
        return
    
    
##########################################################################################


# making function to sort a dataframe by absolute values of a column
def sort_abs(df, col, ascend=False):
    
    df_copy = df.copy()
    new_col = col+'_new'
    df_copy[new_col] = df[col].abs()
    df_copy = df_copy.sort([new_col], ascending=ascend)
    df_copy = df_copy.drop(new_col, axis=1)
    df_copy = df_copy.reset_index(drop=True)
    
    return df_copy
    
    
##########################################################################################


# function that takes a 1-D column np array in question
# and determines whether it is statistically similar 
# to a np array of column 1-D data vectors (i.e. >= 2 vectors), 
# NOT including the vector in question
# NOTE: higher-order Euclidean distance is used as the metric
def vec_stat_sig(vector, array, num_shuffles=1000):
    
    if isinstance(vector, list):
        vector = np.array(vector)
    if isinstance(array, list):
        array = np.array(array)
    
    
    ### determine vector and array distances
    
    # computing median and stadard error; 1-D vectors now
    array_med = np.median(array, axis=1)
    
    # getting distance between vector in question and median
    vec_med_dist = distance.euclidean(vector, array_med) 
    
    
    ### null hypothesis testing
    
    # shuffling vector "n" times one time in a dataframe
    # as long as this function is only used iteratively,
    # there shouldn't be a RAM problem
    
    # tile vector and create dataframe
    # it's a bit tricky to get an array into a single well...
    vector_tile = np.tile(vector, (num_shuffles, 1))
    vector_df = pd.DataFrame([str(i) for i in list(vector_tile)], 
                         columns=['shuffles'])
    
    # modify string, convert back to list of floats, 
    # then shuffle
    vector_df['shuffles'] = vector_df.shuffles.apply(lambda x: 
                        [float(i) for i in x.replace('[', 
                        '').replace(']','').split()])
    vector_df['shuffles'] = vector_df.shuffles.apply(lambda x: 
                        shuffle_array(np.array(x)))
    
    # add median vector
    med_tile = np.tile(array_med, (num_shuffles, 1))
    vector_df['med'] = [str(i) for i in list(med_tile)]
    vector_df['med'] = vector_df.med.apply(lambda x: 
                        np.array([float(i) for i in x.replace('[', 
                        '').replace(']','').split()]))
    
    # compute null-hypothesis distances
    vector_df['med_null'] = vector_df[['shuffles', 
                        'med']].apply(lambda x: 
                        distance.euclidean(x[0], x[1]), axis=1)
        
    # determine whether null hypothesis falls within bound region
    # i.e. "1": null hypothesis closer to median trend than curve in question, else "0"
    
    # getting all distances
    all_dist_array = vector_df.med_null.values 
    
    # computing pval                         
    pval = np.sum(all_dist_array < vec_med_dist)*1.0/num_shuffles

    return round(vec_med_dist, 5), round(pval, 5)

                   
##########################################################################################


## factor analysis components

# this function will take the specified orthogonal vectors from a 
# for given vectors x and y, the function will rotate y to a new y vector
# the new y vector will be slightly correlated with x
# a rough ballpark for correlation is taken as an input
# 'cor' is a number between 0 and 1.0, depending on the amount of correlation desired
def build_corr_vec(L_mat, cols, corr=0, corr_type='spearman', test=False):
    
    # the most accurate range for comparing angle and correlation
    # between correlation of 0.1 and 0.4
    
    two_vecs = L_mat[:, cols]                   # the relevant vectors
    
    # the input vectors need not be orthogonal
    # they may be already slightly correlated; taking this into account
    
    if corr_type.lower() == 'spearman':
        r_current, p_current = spearmanr(two_vecs[:, 0], two_vecs[:, 1])
    else:
        r_current, p_current = pearsonr(two_vecs[:, 0], two_vecs[:, 1])
    
    if p_current >= 0.05:
        r_current = 0
    
    corr = corr - r_current
    
    
    # maximum possible rotation here will be pi/2 (i.e. 90 degrees)
    
    ### correlation = -0.94185*angle + 0.00894 for pearson
    ### correlation = -0.91774*angle - 0.01555 for spearman
    
    if corr == 0:
        angle = 0.
    elif (corr == 1) | (test == True):
        angle = -math.pi/2.*corr
    else:
        if corr_type.lower() == 'pearson':
            angle = (corr-0.00894)/(-0.94185)   # rotating clockwise
        else:
            angle = (corr+0.01555)/(-0.91774)
    
    # building the rotation matrix
    rotate_mat = np.array([[np.cos(angle), -np.sin(angle)], 
                           [np.sin(angle), np.cos(angle)]])
    
    new_vecs = np.dot(two_vecs, rotate_mat)     # rotating the matrix
    L_mat[:, cols[1]] = new_vecs[:, 1]          # replacing the second vector
    
    return L_mat
    

#####


# making function to see if elements of interest in one array are
# at the top of a different array that is ordered by descending values
# 'array2D' is the factor loadings matrix (in nested list form)
# 'array1D' is a list of indices where the proteins occur in the MS dataframe
# 'top' is the number of indices to look through at the top of the ordered array2D
# 'include_tail' is an optional argument to examine the last entries of the array as well
def compare_list(array2D, array1D, top=10, include_tail=False):
    
    array2D = np.array(array2D)
    
    row_num, col_num = np.shape(array2D)
    
    comps_present = np.zeros((1, col_num))
    # look in each column
    for i in range(col_num):
        array2D_coli = array2D[:, i]
        
        # get indices of the ordered descending array
        order_idx = np.argsort(array2D_coli)[::-1]
        
        # now look for array1D elements (which are indices) in 'top' of order_idx
        top_of_idx = order_idx[:top]
        
        # number of components from inputted array1D that are
        # at the top of the indices of the ordered descending array2D
        num_top_comps = sum(pd.Series(top_of_idx).isin(array1D))
        
        if include_tail:
            bot_of_idx = order_idx[::-1][:top]
        
            # making sure bottom indices are not the same as top indices
            # this is only a problem with very small datasets
            bot_of_idx = np.setdiff1d(bot_of_idx, top_of_idx)
            num_bot_comps = sum(pd.Series(bot_of_idx).isin(pd.Series(array1D)))
            comps_present[:, i] = num_top_comps + num_bot_comps
            
        else:
            comps_present[:, i] = num_top_comps

    # maximum number of desired components in a column
    max_comps_present = int(np.max(comps_present))

    return max_comps_present


#####


# making function to take a matrix, get the length of each row, and
# divide each element of that row by that length
# this will be used to scale factor coefficients by communalities
# the function can also take a matrix of summed communalities and return the original matrix
def scale_commun(array2D, sum_mat=[]):
    
    if not isinstance(array2D, np.ndarray):
        array2D = np.array(array2D)
    
    # if we want to scale by communaliies
    if len(sum_mat) == 0:
        rows, cols = np.shape(array2D)
        
        # making matrix of square roots of communalities
        array2D_sq = array2D ** 2
        row_sums = np.sqrt(np.sum(array2D_sq, axis=1))
        sum_mat = np.array(np.tile(row_sums, [cols, 1])).T
        
        # scaling the array
        scaled_array2D = array2D / sum_mat
    
        # returning the scaled array AND the summed communality matrix
        return scaled_array2D, sum_mat
    
    # if we want to get the original matrx
    else:
        if not isinstance(sum_mat, np.ndarray):
            sum_mat = np.array(sum_mat)
        
        descaled_array2D = array2D * sum_mat
        
        # only returning the descaled array
        return descaled_array2D
    
    
#####


# this function will rotate two components, keeping them orthogonal in the process
# in the output matrix, every column will be orthogonal to each other
# 'L_mat' is the input matrix of factors (columns) by protein variables (rows)
# 'theta' is the radian angle used to rotate
def rotate_matrix(L_mat, cols, theta=0):
    
    # L_cols corresponds to the number of factors
    # this will be the dimension of the rotation matrix
    L_rows, L_cols = np.shape(L_mat)
    
    ### first rotate the indicated columns
    matrix = np.eye(L_cols)    # width and height are equal to number of factors
    
    cos_theta = round(np.cos(theta), 3)
    sin_theta = round(np.sin(theta), 3)
    negsin_theta = round(-np.sin(theta), 3)
    
    col_1 = cols[0]
    col_2 = cols[1]
    
    # filling in appropriate positions of matrix
    matrix[:, cols] = matrix[:, cols] * cos_theta
    matrix[col_1+1, col_1] = sin_theta
    matrix[col_2-1, col_2] = negsin_theta
    
    # rotate the indicated columns
    rot_mat = np.dot(L_mat, matrix)     # has same dimensions as 'L_mat'
        
    return rot_mat

    
#####


# making procedure to plot factor combos
def factor_plot(L, combo_i):
    x = L[:, combo_i[0]]
    y = L[:, combo_i[1]]
    
    linexy = np.polyfit(x, y, 1); funcxy = np.poly1d(linexy)
    trendline = 'Linear Fit: y = %sx + %s' % (str(round(linexy[0],2)), str(round(linexy[1],2)))

    plt.figure()
    plt.plot(x, y, 'b.', markersize=15)
    plt.plot(x, funcxy(x), 'k--', label=trendline)
    plt.xlim([min(x)-0.5, max(x)+0.5])
    plt.ylim([min(y)-0.5, max(y)+0.5])
    fac1 = combo_i[0]+1
    fac2 = combo_i[1]+1
    plt.xlabel('Factor %i' % fac1)
    plt.ylabel('Factor %i' % fac2)
    plt.legend(loc='best')
    remove_border()
    plt.show()
    
    
#####


# making function to perform explorative factor analysis
def EFA_func(MS_df, quant_cols, loading_col='GS_STY_KEY', corr=0.0, pathway=[], supervise=False):
    
    
    # first, remove rows that have NaNs
    MS_df = MS_df.ix[~MS_df[quant_cols].apply(lambda x: np.any(np.isnan(x)), axis=1), :].reset_index(drop=True)
    
    ### getting statistics needed later ###
    
    # getting mean and stdev of quant col vals across a row
    data = np.array(MS_df[quant_cols].values)
    avg = np.mean(data, axis=1)
    std = np.std(data, axis=1)
        
    # ensuring that data is zero-averaged
    std_data = (data - np.tile(avg, [len(quant_cols), 1]).T)
     
    # if units are not consistent between the different quant cols
    # toggle this line to standardize data --> (X - X_mean) / X_std
    #std_data = std_data / np.tile(std, [len(quant_cols), 1]).T
    
    # turning standardized data into matrix for later
    std_data = np.matrix(std_data)
    
    # number of protein variables
    prot_num = np.shape(std_data)[0]
    
    
    ### PCA section: using PC method for EFA to determine number of factors ###
    
    # running PCA to get principal component and eigenvalues
    # not standardizing since all data has same unit
    eigvals_sort, eigvecs_sort, signals, cov_mat = PCA_func(MS_df, quant_cols, False)
    eigvals_sort = eigvals_sort.real
    
    tot_eigvals = len(eigvals_sort)     # number of eigenvalues
    
    # plotting eigenvalues to see trend
    # will choose number of eigenvalues above where there's a dramatic change in slope
    # will only plot up to 20 eigenvalues
    
    x = np.arange(tot_eigvals+1)[1:]
    y = np.array(eigvals_sort)
    ones = np.ones((1, tot_eigvals))[0]
    
    if len(x)>20:
        x = x[:20]
        y = y[:20]
        ones = ones[:20]
    
    # plotting eigenvalues
    plt.figure()
    plt.plot(x, y, 'b.', markersize=15)
    plt.plot(x, y, 'b-')
    plt.plot(x, ones, 'k--', label='Threshold (Eigenvalue=1)')
    plt.xlim([0, len(x)+2])
    plt.title('Eigenvalues\n')
    plt.legend(loc='best')
    remove_border()
    plt.show()
    
    print '''
    Choose the number of eigenvalues that corresponds to 
    the bottom of the cliff (before the curve flattens). 
    Note: any selected eigenvalues should be greater than 1.
    '''
    
    # number eigenvalues chosen will be the number of factors used
    factor_num = int(raw_input('How many eigenvalues to use for factor analysis?: ')); print
    
    # taking 'factor_num' worth of the available eigenvectors
    # before something went wrong because I treated rows 
    # as eigenvectors by mistake
    prin_comps = eigvecs_sort[:, :factor_num]
    
    
    ### NOTE: I AM ONLY TAKING THE REAL PARTS OF THE LOADINGS
    prin_comps = prin_comps.real     # each row is an eigenvector
                                     # each column is a protein

    
    
    
    # here, I am multiplying a given eigenvector column 
    # by the square root of the respective eigenvalue
    L = prin_comps

    print 'Number of proteins: %i\n' % len(L[:, 0])
    prot_num = len(L[:, 0])
    
    print '''Eigenvectors need to have unit length. \nL dot product before normalization: %f
    ''' % float(np.dot(L[:, 0].T, L[:, 0]))
    
    # turning eigenvectors into unit vectors
    for i in range(factor_num):
        L[:, i] = L[:, i] / math.sqrt(np.dot(L[:, i].T, L[:, i]))
    
    print '''After normalization: %f''' % float(np.dot(L[:, 0].T, L[:, 0]))

    
    ### building factor loadings matrix 'L' ###
    
    for i in range(factor_num):
        L[:, i] = L[:, i] * math.sqrt(eigvals_sort[i] / max(eigvals_sort))
        # now L is a matrix of estimated factor loadings
    
    # L x L' matrix; this is an approximation of the covariance
    LxLprime = np.dot(L, L.T)
    
    # this is the 'specific covariance'
    spec_var = cov_mat - LxLprime


    ### examining best rotation angle ###

    # rotating matrix to 'focus' the factors
    # getting combinations of factor column positions
    fac_combos = list(combinations(np.arange(factor_num), 2))
    
    # making dataframe with 100 angles
    # also getting indices where the inputted pathway components occur in the MS_df
    if len(pathway) > 0:

        pathway_df = MS_df[loading_col].isin(pathway)
        pathway_idx = list(pathway_df.index[pathway_df == True]) # the pathway indices
        
    n_ang = 100
    
    
    angles = np.arange(n_ang)+1                     # this goes from 1 to 100
    angles = angles[:-1][::-1]                     # this goes from 100 to 2
    angles = math.pi*2/angles                      # this goes from 100/(2*pi) to 2*pi
    
    # finalized angle range to test
    angles = np.array([0] + list(angles))          # now this goes from 0 to 2*pi/2
    
    
    ### automated angle choice ###
    
    # going through one combination of factors at a time; then rotate
    for i in range(len(fac_combos)):
        
        rot_df = pd.DataFrame(angles, columns=['angle'])
        
        combo_i = fac_combos[i]
        
        
        ### making factors correlated to each other according to the user input ###
        
        L = build_corr_vec(L, combo_i, corr, 'pearson')
       
        # if the user just wants the program to pick the best angle
        if not supervise:
        
            # can't put 2D arrays in a dataframe cell -->
            # need to turn the matrix output (a numpy array) into nested lists
            # need to scale the factor coefficients by the square root of the communalities
            L_scaled, comm_mat = scale_commun(L)
            
            # each element in this column is a rotated version of L
            rot_df['L_star'] = rot_df.angle.apply(lambda x: 
                            rotate_matrix(L, combo_i, x).tolist())
            
            # making temporary column with matrix of summed communalities
            rot_df['Commun'] = rot_df.L_star.apply(lambda x: 
                            scale_commun(np.array(x))[1].tolist())
            
            # scaling L_star matrices by communalities
            rot_df['L_star'] = rot_df.L_star.apply(lambda x: 
                            scale_commun(np.array(x))[0].tolist())
            
            # implementing 'Varimax Criterion' to get best rotation angle for combo_i
            rot_df['varimax'] = rot_df.L_star.apply(lambda x: 
                        1./prot_num*np.sum(np.sum(np.array(x)**4, 
                        axis=0) - np.sum(np.array(x)**2, axis=0)**2/prot_num))
            
            # de-scaling the rotated matrices
            rot_df['L_star'] = rot_df[['L_star', 'Commun']].apply(lambda x: 
                            scale_commun(x[0], x[1]).tolist(), axis=1)

            # searching for pathway components
            # if pathway inputted, search for pathway among factor loadings
            if len(pathway) > 0:
                rot_df['pathway_comps'] = rot_df.L_star.apply(lambda x:
                        compare_list(x, pathway_idx))
                        
                # max number of pathway components in top 10 factor loadings
                max_comps = rot_df.pathway_comps.max()
                    
                # number of rotations with the maximum number of inputted pathway components
                num_max = len(rot_df.pathway_comps[rot_df.pathway_comps == max_comps])
                    
                print 'Number of rotations that have %i of the indicated pathway components: %i\n' % (max_comps, num_max)
            
                rot_df = rot_df[rot_df.pathway_comps == max_comps].reset_index(drop=True)
            
            
            best_angles = rot_df.angle[rot_df.varimax == rot_df.varimax.max()].reset_index(drop=True)
            
            # this angle maximizes the variance of the squared loadings for a given factor
            best_angle = best_angles[0]     # if more than one 'best' angle, take the smallest
            
            # converting angle to degrees for display
            best_angle_deg = best_angle * 360./(2*math.pi)
            
            print 'Best angle for rotating column %i and %i: %s degrees\n' % (combo_i[0]+1, 
                                                    combo_i[1]+1, str(round(best_angle_deg, 3)))
        
            # now rotating combo_i --> changing the respective columns in L
            # other columns that are not in combo_i are left unchanged
            L = np.array(rot_df.L_star[rot_df.angle == best_angle].values[0])
            
            # plotting factors after rotation
            factor_plot(L, combo_i)
        
        
    ### if user wants to choose the angle themselves ###
        
        else:
            
            commit = 'no'
            while commit.lower() != 'yes':
            
                # plotting factors before rotation
                factor_plot(L, combo_i)
                
                print '''
                Provide an angle: if you wish to indicate an angle in degrees,
                please include a space after the angle as well as the word 'degrees.'
                '''
                
                response = raw_input('Angle to use?: '); print
                response = response.split()
                
                best_angle = float(response[0])
                
                if len(response) > 1:
                
                # if user gives angle in degrees, convert to radians
                    if response[1].lower() == 'degrees':
                        best_angle = best_angle * math.pi*2/360.
    
                # making the appropriate rotation matrix
                L_trial = rotate_matrix(L, combo_i, best_angle)
                
                # showing rotated plots
                factor_plot(L_trial, combo_i)

                # if user likes the result, commit with 'yes'
                commit = raw_input('Commit to angle?: '); print
            
            # if user commits, make L_trial the new factor loadings matrix
            L = L_trial
        
        
    ### section specifying factor scores and protein loadings ###
        
    # now that we have our rotated matrix, we need to get factor scores
    L = np.matrix(L)
    
    # here, columns are states and rows are factors
    fscore_matrix = (L.T * L).I * L.T * std_data

    # normalize factor scores to absolute value of 1
    # getting sum of factor loadings for each factor; 
    # i.e. getting sum across each row, then taking transpose
    fscore_sums = np.sum(abs(fscore_matrix), axis=1).T     
    fscore_sums = np.tile(fscore_sums, [len(quant_cols), 1]).T
    
    # dividing each factor loading by the appropriate factor sum
    fscore_matrix = np.array(fscore_matrix) / fscore_sums
    fscore_matrix = fscore_matrix.astype(float).round(3)     # round to 3 decimal places
    
    # making columns the states and rows the factors, in order
    # making dataframe for states in terms of factor loadings
    fscore_states = pd.DataFrame(quant_cols, columns=['States'])
    
    # making column titles for factors
    cols = ['Factor_'+str(i+1) for i in range(factor_num)]
    temp_df = pd.DataFrame(fscore_matrix.T, columns=cols)
    
    # joining states; setting states as the index
    fscore_states = fscore_states.join(temp_df)
    
    # normalizing factor loadings (for protein variables)
    # getting sum of each column
    prot_loading_sums = np.sum(abs(L), axis=0)    
    prot_loading_sums = np.array(np.tile(prot_loading_sums, [prot_num, 1]))    
    
    # dividing each factor loading by the appropriate factor sum
    prot_loading_matrix = np.array(L) / prot_loading_sums
    prot_loading_matrix = prot_loading_matrix.astype(float).round(3)     # round to 3 decimal places
    
    # making dataframe for protein loadings of each factor
    fscore_loadings = pd.DataFrame(prot_loading_matrix, columns=cols)
    fscore_loadings['Loadings'] = MS_df[loading_col].values
    

    ### re-ordering factors
    
    # renumbering factors according to most variance in loadings
    factor_vars = []
    for i in range(factor_num):
        factor_i_var = fscore_loadings['Factor_'+str(i+1)].var()
        factor_vars.append(factor_i_var)
        
    # now 'Factor_1' will correspond to the factor with the most variance, etc
    new_order = np.argsort(factor_vars)[::-1]
    
    old_factors = []
    for i in range(factor_num):
        
        factor = 'Factor_'+str(i+1)
        
        # temporarily adding an 'a' to not overwrite columns
        # i.e. need the original columns for reference for each iteration
        fscore_loadings[factor+'a'] = fscore_loadings['Factor_'+str(new_order[i]+1)]
        
        # doing the same thing in the fscore_states dataframe
        fscore_states[factor+'a'] = fscore_states['Factor_'+str(new_order[i]+1)]
        
        old_factors.append(factor)     # making list of old factor columns
        
    # dropping old factor columns
    fscore_loadings = fscore_loadings.drop(old_factors, 1)
    fscore_states = fscore_states.drop(old_factors, 1)
        
    # taking off the extra 'a' in the factor column labels
    for i in range(factor_num):
        fscore_loadings = fscore_loadings.rename(columns={old_factors[i]+'a': old_factors[i]})
        fscore_states = fscore_states.rename(columns={old_factors[i]+'a': old_factors[i]})
    
    
    ### quality control section ###
    
    # testing to see whether factors are still orthogonal to each other
    L_test = np.array(L)
    rows, cols = np.shape(L_test)
    
    print 'Testing if factors are still orthogonal\n'
    for i in range(cols-1):
        
        dot_L = np.dot(L_test[:, 0], L_test[:, i+1])
        print 'Dot product of Factor 1 and Factor %i: %f' % (i+2, dot_L)
    
    # making sure specific variance has not changed much
    old_spec_var = spec_var
    new_spec_var = cov_mat - L*L.T
    
    # linearizing arrays and finding correlation for the first 100 rows and 100 columns
    # they should be almost perfectly correlated if everything has been done correctly
    old_spec_var = sum(old_spec_var[:100, :100].tolist(), [])
    new_spec_var = sum(new_spec_var[:100, :100].tolist(), [])
    
    r, p = pearsonr(old_spec_var, new_spec_var)
    
    print '\nCorrelation between old and new specific variance: %s' % str(round(r, 3))
    
    return fscore_states, fscore_loadings


#####


# making procedure to plot loadings from EFA
# num_comps is the number of factors to include
def plot_EFA(EFA_loadings, EFA_states, quant_labels, num_factors=2, nprot_load=15, 
             xticks=[], legend=False, bg=[]):
    
    max_load_num = len(EFA_loadings.Loadings.unique())
    
    if nprot_load > max_load_num:
        print '''
        There are fewer than %i loadings\n
        Displaying all protein loadings (%i)
        ''' % (nprot_load, max_load_num)
        
        nprot_load = max_load_num
    
    
    if legend:
        # setting colors and creating figure for loadings plot
        # blue, red, green, magenta, black]  
        
        # this will be [dark blue, light blue, dark red, light red, etc...]
        colors = [(0, 0, 0.6), (0, 0, 1), (0.6, 0, 0), (1, 0, 0), (0, 0.6, 0),
                  (0, 1, 0), (0.6, 0, 0.6), (1, 0, 1), (0, 0, 0), (0.6, 0.6, 0.6)]
            
        if bg == 'black':
            colors = colors[:-2] + [(0.6, 0.6, 0.6), (1, 1, 1)]
            
        colors = colors[:len(quant_labels)]  
        
    
    fig, axes = plt.subplots(figsize=(10,10), nrows=1, ncols=num_factors)
    
    ypos = np.arange(nprot_load)+0.5  # want the bar graph to center on the y-axis
    
    for i in range(num_factors):
        
        factor_i = 'Factor_'+str(i+1)
        
        
        ##### DO THE NEGATIVE LOADINGS MEAN ANYTHING FOR FACTOR ANALYSIS? #####
        ##### IF THEY DO, THEN I SHOULD SORT ACCORDING TO ABSOLUTE VALUE #####
        
        #EFA_df = EFA_df.sort(factor_i, ascending=False)
        EFA_loadings = sort_abs(EFA_loadings, factor_i)

        # plotting loadings
        loading_vals = list(EFA_loadings[factor_i].head(nprot_load).values)
        loading_labels = list(EFA_loadings.Loadings.head(nprot_load))
        
        axes[i].barh(ypos, loading_vals, color='b', align='center')
        
        xmin, xmax, ymin, ymax = axes[i].axis()     # getting axis limits
        
        # adding x=0 line for reference
        x_ref = np.zeros((ymax+1)); y_ref = np.arange(ymax+1)
        axes[i].plot(x_ref, y_ref, 'k-', linewidth=1.0)
        
        axes[i].set_yticks(ypos)
        axes[i].set_yticklabels(loading_labels)
        axes[i].set_xlabel('Loading Value')

        start, end = axes[i].get_xlim()
        stepsize = (end-start)/2.5     # this gets about 3 xticks
        axes[i].xaxis.set_ticks(np.arange(start, end, stepsize))
        
        axes[i].xaxis.label.set_fontsize(17)     # axis label
        axes[i].invert_yaxis()     # making sure longer bars are on top
        
        loadings_title = 'Factor %i\n' % (i+1)
        axes[i].set_title(loadings_title, loc='center')
        remove_border(axes[i])
        set_plot_params(ax=axes[i])
        

    fig.tight_layout()   # making sure plot labels are not overlapping
    plt.show()
    
    
    
     # plotting projected data
    for j in range(num_factors):
        
        efa_proj = EFA_states['Factor_'+str(j+1)].tolist()
        
        x = np.arange(len(quant_labels))+1
        y = efa_proj      # 1st principal component axis   
        
        plt.figure(j)
        
        for k in range(len(x)):
            
            if legend:
                plt.plot(x[k], y[k], 'o', color=colors[k], markersize=10, label=quant_labels[k])
            else:
                plt.plot(x[k], y[k], 'bo', markersize=10)
        
        plt.xlim([-0.5, len(quant_labels)+0.5])
        plt.ylim([min(y)-0.15*abs(min(y)), max(y)+0.15*abs(max(y))])
        
        #plt.xlabel('Samples', fontsize=17)
        if len(xticks) > 0:
            plt.xticks(x, xticks)
        else:
            plt.xticks(x, quant_labels, rotation=90)  
            
        plt.ylabel('Factor %s' % str(j+1), fontsize=17)

        plt.legend(loc='best', numpoints=1, prop={'size':12})
        remove_border()
        
        if bg == 'black':
            black_plot_param(legend=True)
        plt.show()
    
    
    return fig


##########################################################################################


# making function that can take dot products of two arrays with missing values
def nan_dot_product(array1, array2, normalize=True):
    
    # verify inputs are numpy arrays
    array1 = np.array(array1)
    array2 = np.array(array2)
    
    # get array dimensions
    rows1, cols1 = np.shape(array1)
    rows2, cols2 = np.shape(array2)
    
    # verify that cols1 = rows2
    if cols1 != rows2:
        print 'Incompatible array dimensions...'
        return
    
    # initiate output array
    output_array = np.zeros((rows1, cols2))
    
    for row in np.arange(rows1):
        for col in np.arange(cols2):
            # for a given row in array1 and a given column in array2
            # iterate through the column values of the row in array1
            # simultaneously iterate through the row values of the column in array2
            array1_row = array1[row, :]
            array2_col = array2[:, col]
            
            # take NaN-sum of element-wise products
            pos_value = np.nansum(array1_row*array2_col)
            
            if normalize:
                pos_value = pos_value / np.sum(array1_row)
            
            # record value in appropriate output position
            output_array[row, col] = pos_value
            
    return output_array   


##########################################################################################

##########################################################################################

##########################################################################################

##########################################################################################

##########################################################################################

