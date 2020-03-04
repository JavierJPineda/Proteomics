## This file contains several functions useful for mass spec analysis

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

# useful stuff for text wrangling
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
from JP_stat_functions import *
from JP_uniprot_blast_functions import *


##########################################################################################


# loading uniprot dataset
uniprot_df = pd.read_csv('../Coding_Stuff/Human_Uniprot_Info_081816.csv')

# making function to clean-up/filter MS data
# 'filter_data' is an optional argument
# if you want to "look" at the raw data, set filter_data=False
def MS_cleanup(MS_data_raw, quant_cols, species='human', filter_data=True, normalize=True, method='median', ms='lumos', 
               phospho=False, remove_ox=True, label_free=False, search_uniprot=False):
    
    # these are abbreviated forms of the TMT channel columns
    default_ch = ['_126_', '_127n_', '_127c_', '_128n_', '_128c_', 
                  '_129n_', '_129c_', '_130n_', '_130c_', '_131_', '_131C_']

    # look for necessary columns and rename them to make them compatible with all functions
    for col in MS_data_raw.columns:
        if col.lower().replace(' ','').replace('_','') == 'genesymbol':
            MS_data_raw = MS_data_raw.rename(columns={col:'Gene_Symbol'})
        elif col.lower().replace(' ','').replace('_','') == 'proteinid':
            MS_data_raw = MS_data_raw.rename(columns={col:'Protein_ID'})
        elif col.lower().replace(' ','').replace('_','') == 'peptideid':
            MS_data_raw = MS_data_raw.rename(columns={col:'Peptide_ID'})
            
    # possible that Protein Id column was renamed in CORE (sloppy programming...)
    if 'Protein_ID' not in MS_data_raw.columns:
        if 'Reference' in MS_data_raw.columns:
            MS_data_raw = MS_data_raw.rename(columns={'Reference':'Protein_ID'})
        else: 
            print '\nNo column with protein ID detected. Aborting clean-up...'
            return
    
    # for protein level data
    for col in MS_data_raw.columns:
        if col.lower().replace(' ','').replace('_','') == 'numberofpeptides':
            MS_data_raw = MS_data_raw.rename(columns={col:'Peptide_Num'})
            
    # columns necessary for peptide data (e.g. phos or non-phos)
    for col in MS_data_raw.columns:
        if col.lower().replace(' ','').replace('_','') == 'peptide':
            MS_data_raw = MS_data_raw.rename(columns={col:'Sequence'})
        elif col.lower().replace(' ','').replace('_','') == 'peptidesequence':
            MS_data_raw = MS_data_raw.rename(columns={col:'Sequence'})
        elif col.replace(' ','').replace('_','') == 'sequence':
            MS_data_raw = MS_data_raw.rename(columns={col:'Sequence'})
        elif col.replace(' ','').replace('_','') == 'peptideid':
            MS_data_raw = MS_data_raw.rename(columns={col:'Peptide_ID'})
    
    # columns necessary for phospho data
    if phospho:
        multi = False  
        for col in MS_data_raw.columns:
            
            # at first, assuming that we're working with multi-site peptides
            if col.lower().replace(' ','').replace('_','') == 'siteidstr':
                MS_data_raw = MS_data_raw.rename(columns={col:'Site_ID'})
                
                multi = True
                
            elif (col.lower().replace(' ','').replace('_','') == 
                  'siteposition') | (col.lower().replace(' ','').replace('_','') == 'siteposstr'):
                MS_data_raw = MS_data_raw.rename(columns={col:'Site_Position'})
                
            elif 'motif' in col.lower():
                MS_data_raw = MS_data_raw.rename(columns={col:'Motif'})
           
        # if we're dealing with single-site peptides
        if not multi:
            for col in MS_data_raw.columns:
                if col.lower().replace(' ','').replace('_','') == 'siteid':
                    MS_data_raw = MS_data_raw.rename(columns={col:'Site_ID'})
                    
                # getting PTM localization score
                if col.lower().replace(' ','').replace('_','') == 'maxscore':
                    MS_data_raw = MS_data_raw.rename(columns={col:'Max_Score'})
                    
        else:
            for col in MS_data_raw.columns:
                if col.lower().replace(' ','').replace('_','') == 'maxscorestr':
                    MS_data_raw = MS_data_raw.rename(columns={col:'Max_Score'})
                    
       
        # verify presence Site_Position column
        if 'Site_Position' not in MS_data_raw.columns:
            print 'Site_Position column not present. Aborting...\n'
            return
        else:
            MS_data_raw['Site_Position'] = MS_data_raw.Site_Position.apply(lambda x: str(x))
        
    # replace channel names with desired column labels
    quant_cols_pop = quant_cols+[]
    if not label_free:
        num_ch = len(quant_cols)
        for ch in default_ch:
            for col in MS_data_raw.columns:
                
                if ch.lower() in col.lower():
                    MS_data_raw = MS_data_raw.rename(columns={col:quant_cols_pop[0]})
                    
                    # verify that numbers are interpreted as floats
                    MS_data_raw[quant_cols_pop[0]] = MS_data_raw[quant_cols_pop[0]].apply(lambda x: 
                                                    float(x) if not isinstance(x, str) else 0.0)
                    quant_cols_pop.pop(0)
             
        ch_sum_present = False
        for col in MS_data_raw.columns:
            if 'sum' in col.lower():
                MS_data_raw = MS_data_raw.rename(columns={col:'Channel_Sum'})
                ch_sum_present = True
                             
        # determine sum of ion count across channels in case filtering desired later
        if not ch_sum_present:
            MS_data_raw['Channel_Sum'] = MS_data_raw[quant_cols].apply(lambda x: 
                                    np.sum(x), axis=1)                
                     
    # remove rows where all channel columns equal 0 (e.g. this happens with phospho)
    MS_data_raw = MS_data_raw.ix[~(MS_data_raw[quant_cols].apply(lambda x: 
                                        np.all(x == 0), axis=1)), :].reset_index(drop=True)
    
    # replacing all empty strings in dataframe with 'null'
    MS_data_raw = MS_data_raw.applymap(lambda x: 
              'null' if isinstance(x, basestring) and x.isspace() else x)
        
    # getting rid of decoys (they start with '#')
    MS_data_raw = MS_data_raw.ix[MS_data_raw.Protein_ID.apply(lambda x: x[0] != '#'), :]
    
    # getting rid of contaminants
    MS_data_raw = MS_data_raw.ix[MS_data_raw.Protein_ID.apply(lambda x: x[0:2] != 'UP'), :]
    
    ### throwing out serum albumin (levels can be quite high if cells weren't washed well)
    MS_data_raw = MS_data_raw.ix[MS_data_raw.Gene_Symbol.apply(lambda x: x != 'ALB'), :]
    
    # getting rid of contaminants
    MS_data_raw['Protein_ID'] = MS_data_raw.Protein_ID.apply(lambda x: 
                'null' if x[x.index('_')+1:].lower() != species else x)
    
    desc = False
    for col in MS_data_raw.columns:
        if 'desc' in col.lower():
            
            desc = True
            
            # rename column
            MS_data_raw = MS_data_raw.rename(columns={col:'Description'})
            
            # get rid of any keratin
            MS_data_raw['Protein_ID'] = MS_data_raw[['Protein_ID', 
                    'Description']].apply(lambda x: 
                    'null' if 'keratin' in x[1].lower() else x[0], axis=1)
    if not desc:
        print '\nNo description column was detected. Keratin will not be thrown out.'
    
    # removing empty strings, contaminants, and negative controls
    MS_data_raw = MS_data_raw[~(MS_data_raw.Protein_ID == 'null')]
    
    if 'Sequence' not in MS_data_raw.columns.tolist():
        print '''
        \nProteins not including common contaminants or where missing sequence ID: %i
        ''' % len(MS_data_raw)
    else:
        print '''
        \nPeptides not including common contaminants or where missing sequence ID: %i
        ''' % len(MS_data_raw)
    
    # some protein abbreviations are null
    # in this case, get whatever is in the protein ID
    MS_data_raw['Gene_Symbol'][MS_data_raw.Gene_Symbol == 
        'null'] = MS_data_raw.Protein_ID[MS_data_raw.Gene_Symbol == 
                                         'null'].apply(lambda x: 
            x[x[3:].index('|')+4:x.index('_')])
    
    # capitalizing all protein abbreviations
    MS_data_raw['Gene_Symbol'] = MS_data_raw['Gene_Symbol'].apply(lambda x: str(x).upper())
    
    # adding column with just sequence IDs
    MS_data_raw['Sequence_ID'] = MS_data_raw.Protein_ID.apply(lambda x: 
            x[x.index('|')+1:x[3:].index('|')+3])
    
    MS_data = MS_data_raw.copy()


    # for phospho data
    # remove any lines where "*" is in the sequence
    # i.e. oxidized methionines

    if 'Sequence' in MS_data.columns.tolist():
        if remove_ox:
            # taking out oxidized peptides
            MS_data = MS_data.ix[~(MS_data.Sequence.apply(lambda x: 
                                '*' in x)), :].reset_index(drop=True)
        MS_data['Sequence'] = MS_data['Sequence'].apply(lambda x: 
                              x[x.index('.')+1:-x[::-1].index('.')-1])
        
        # sort by descending channel sum in case there are duplicate peptides
        # this only really happens with non-phos data
        MS_data = MS_data.sort('Channel_Sum', ascending=False).reset_index(drop=True)

        # best key for phospho peptides = gene symbol + site position
        if phospho:
            MS_data['Peptide_Key'] = MS_data[['Gene_Symbol', 'Site_Position']].apply(lambda x: 
                                        str(x[0])+str(x[1]), axis=1)
            
        # best key for nonphos peptides = gene symbol + sequence
        else:
            MS_data['Peptide_Key'] = MS_data[['Gene_Symbol', 'Sequence']].apply(lambda x: 
                                        str(x[0])+str(x[1]), axis=1)

        # remove duplicate peptides; takes the peptide with the best signal
        MS_data = MS_data.drop_duplicates('Peptide_Key').reset_index(drop=True)

    if filter_data:
        
        if phospho:
            nonphos = len(MS_data.ix[MS_data.Sequence.apply(lambda x: '#' not in x), :])
            print 'Filtering out non-phosphorylated peptides: %i peptides\n' % nonphos
        
            # remove non-phosphorylated peptides
            MS_data = MS_data.ix[MS_data.Sequence.apply(lambda x: 
                                    '#' in x), :].reset_index(drop=True)
                                    
        # non-phos peptide data
        elif 'Sequence' in MS_data.columns.tolist():
            phos = len(MS_data.ix[MS_data.Sequence.apply(lambda x: '#' in x), :])
        
            print 'Filtering out phosphorylated peptides: %i peptides\n' % phos
        
            # remove phosphorylated peptides
            MS_data = MS_data.ix[MS_data.Sequence.apply(lambda x: 
                                    '#' not in x), :].reset_index(drop=True)
        
        
        # filtering out rows where sum is less than:
        # 200 (using lumos instrument)
        # 386 (using fusion instrument)
        # 189 (using elite instrument)
        if ms.lower() == 'lumos':
            tot_for_10 = 185.0
        elif ms.lower() == 'fusion':
            tot_for_10 = 386
        elif ms.lower() == 'elite':
            tot_for_10 = 189
            
        num_ch = len(quant_cols)
        cutoff = tot_for_10 * num_ch / 10.0
        MS_data = MS_data[MS_data.Channel_Sum >= cutoff]
            
            
    print 'Number of measurements retained: %i\n' % len(MS_data)
    
    # finding sum of all protein intensities in each channel
    # then using this array to normalize dataset
    if method.lower() == 'median':
        ch_medians = np.array(MS_data[quant_cols].apply(np.median, axis=0))
        norm_vals = ch_medians / ch_medians.min() * 1.0
        
    else:
        ch_sums = np.array(MS_data[quant_cols].apply(np.sum, axis=0))
        norm_vals = ch_sums / ch_sums.min() * 1.0
    
    if np.any(norm_vals > 2):
        print 'Ratio between certain channels is greater than 2.\n'
    
    # now normalizing by channel sums/medians (04-06-16)
    if normalize:
        print 'Normalization values:' 
        print norm_vals, '\n'
        
        for i in range(len(quant_cols)):
            MS_data[quant_cols[i]] = MS_data[quant_cols[i]] / norm_vals[i]
    
    
    ## getting family information
    
    MS_data['Family'] = MS_data[['Gene_Symbol', 'Description']].apply(lambda x: 
                                assign_prot_fam(str(x[0]), [], str(x[1])), axis=1)
    
    # in case not enough info in description column, include family info from uniprot
    uniprot_info = pd.read_csv('/Users/javier/Desktop/Mitchison_Lab/Coding_Stuff/Human_Uniprot_Info_081816.csv')
    
    # split dataset into subset with family info and subset without family info
    MS_data_family = MS_data[MS_data.Family != 'Other']
    MS_data_other = MS_data[MS_data.Family == 'Other']
    
    # merge uniprot info for 'Other' subset
    MS_data_other['Family'] = MS_data_other.Gene_Symbol.apply(lambda x: 
                                uniprot_info.Family[uniprot_info.Gene_Symbol == 
                                x].tolist()[0] if x in uniprot_info.Gene_Symbol.tolist() else np.nan)
    
    MS_data = pd.concat([MS_data_family, MS_data_other]).reset_index(drop=True)
    
    # if gene symbols are not present in the uniprot dataset, this will give a NaN float
    # in this case, return 'Need Uniprot Info'
    MS_data['Family'] = MS_data.Family.apply(lambda x: 'Need Uniprot Info' if not isinstance(x, str) else x)
    
    if search_uniprot:
        
        MS_data_done = MS_data[MS_data.Family != 'Need Uniprot Info']
        MS_data_NUI = MS_data[MS_data.Family == 'Need Uniprot Info']
        
        if len(MS_data_NUI) > 0:
        
            # record these proteins and drop any duplicate rows (e.g. if dealing with peptide data)
            NUI = MS_data[['Gene_Symbol', 'Sequence_ID']][MS_data.Family == 'Need Uniprot Info']
            NUI = NUI.drop_duplicates()   # w/o this, we will uselessly search multiple times for the same info   

            # in these cases, go to uniprot to get the information; this will take some time
            NUI['Family'] = NUI[['Gene_Symbol', 'Sequence_ID']].apply(lambda x: 
                                    assign_prot_fam(str(x[0]), str(x[1]), []), axis=1)

            # drop duplicate gene symbols at this point (we will always merge on gene symbol for family info)
            NUI = NUI.drop_duplicates('Gene_Symbol')

            # record new family information
            MS_data_NUI = MS_data_NUI.merge(NUI[['Gene_Symbol', 'Family']], on='Gene_Symbol', how='left', copy=False)

            # concatenate MS data again
            MS_data = pd.concat([MS_data_done, MS_data_NUI]).reset_index(drop=True)

            # just in case extra columns were added
            NUI = NUI[['Gene_Symbol', 'Sequence_ID', 'Family']]
            NUI['Description'] = NUI.Sequence_ID.apply(lambda x: get_uniprot_str(x))

            uniprot_info = uniprot_info[~(uniprot_info.Gene_Symbol.isin(NUI.Gene_Symbol.tolist()))]
            uniprot_info = pd.concat([uniprot_info, NUI])

            # sort uniprot info and readout
            uniprot_info = uniprot_info.sort('Gene_Symbol').reset_index(drop=True)
            uniprot_info.to_csv('/Users/javier/Desktop/Mitchison_Lab/Coding_Stuff/Human_Uniprot_Info_081816.csv', index=False)
    
        else:
            MS_data = MS_data_done
    
    # correct septins and march proteins
    MS_data['Gene_Symbol'] = MS_data.Gene_Symbol.apply(lambda x: 
                'SEPT'+str(x)[:str(x).index('-')] if str(x)[-4:] == '-SEP' else x)
    
    MS_data['Gene_Symbol'] = MS_data.Gene_Symbol.apply(lambda x: 
                'MARCH'+str(x)[:str(x).index('-')] if str(x)[-4:].upper() == '-MAR' else str(x).upper())
    
    
    # for some reason, multiple family columns (i.e. 'Family', 'Family_x', 'Family_y')
    if 'Family_y' in MS_data.columns:
        MS_data['Family'] = MS_data[['Family', 'Family_y']].apply(lambda x: 
                    str(x[1]) if not isinstance(x[0], str) else str(x[0]), axis=1)
        MS_data = MS_data.drop('Family_x', 1).drop('Family_y', 1)
        
    # only keep necessary columns
    keep_cols = ['Gene_Symbol', 'Sequence_ID', 'Protein_ID', 'Site_ID', 'Peptide_Key', 'Description', 'Family', 
                 'Motif', 'Sequence', 'Site_Position', 'Max_Score', 'Channel_Sum'] + quant_cols
    
    for col in MS_data.columns:
        if col not in keep_cols:
            MS_data = MS_data.drop(col, 1)
    
    # return dataset sorted by gene symbol
    return MS_data.sort('Gene_Symbol').reset_index(drop=True)

    
##########################################################################################


# making function to normalize and log2 transform dataset values (i.e. to get fold change)
def get_fc_df(ms_df, quant_cols, method='mean', log=True, include_inf=False):
    
    ms_df_copy = ms_df.copy()
    
    if method.lower() == 'median':
        norm_array = np.array(ms_df_copy[quant_cols].apply(np.median, axis=1))
        
    # normalize by the first column
    elif method.lower() == 'first':
        norm_array = np.array(ms_df_copy[quant_cols[0]])
    
    elif method.lower() == 'sum':
        norm_array = np.array(ms_df_copy[quant_cols].apply(np.sum, axis=1))
        
    elif method.lower() == 'min':
        norm_array = np.array(ms_df_copy[quant_cols].apply(np.min, axis=1))
        
    # or else, just use mean normalization
    else:
        norm_array = np.array(ms_df_copy[quant_cols].apply(np.mean, axis=1))
        
    # getting row minimums
    if not log:
        mins_array = np.array(ms_df_copy[quant_cols].apply(np.min, axis=1))
        
    # performing the normalization
    for col in quant_cols:
        ms_df_copy[col] = ms_df_copy[col] / norm_array
        
        # log2 tranformation
        if log:
            ms_df_copy[col] = ms_df_copy[col].apply(lambda x: 
                                                    np.log2(x))

    # some rows will have -inf in them 
    # if those rows are not desired
    if not include_inf:
        row_num_fin = ms_df_copy[quant_cols].applymap(np.isinf).apply(np.sum, 
                                                                      axis=1).apply(lambda x: x == 0)
        ms_df_copy = ms_df_copy.ix[row_num_fin, :]
    
        
    # resetting index
    ms_df_copy = ms_df_copy.reset_index(drop=True)
        
    return ms_df_copy


##########################################################################################


# making procedure to visualize data points for given proteins
# takes in a dataframe subset and a list of protein keys
# may need to be adapted for different data formats
def see_trend(subset_df, prot_ids, quant_cols, err_cols=[],
              xvals=[], ylim=[], key_col='Gene_Symbol', 
              delog=True, log_data=True, bg=[], savefig=[]):
    
    if isinstance(prot_ids, str):
        prot_ids = [prot_ids]
    
    # this will be [dark blue, light blue, dark red, light red, etc...]
    colors = 2*[(0, 0, 0.6), (0, 0, 1), (0.6, 0, 0), (1, 0, 0), (0, 0.6, 0), 
              (0, 1, 0), (0.6, 0, 0.6), (1, 0, 1), (0.6, 0.6, 0.6), (0, 0, 0)]
    
    # setting x-axis values and tick labels
    if len(xvals) == 0:
        x = np.arange(len(quant_cols))
        xlabels = quant_cols
    else:
        if isinstance(xvals[0], str):
            x = np.arange(len(quant_cols))
        else:
            x = xvals
        xlabels = xvals
    
    
    # set figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # make array of y-values if overlaying
    y_list = []    
    y_err_list = []
    for i in range(len(prot_ids)):
        
        y = subset_df[quant_cols][subset_df[key_col] == 
                                prot_ids[i]].values
        
        if len(err_cols) > 0:
            y_err = subset_df[err_cols][subset_df[key_col] == 
                                prot_ids[i]].values
        
        # if the desired protein is not in the given dataset, move on
        if len(y) == 0:
            continue
        else:
            y = y[0]
            
            if len(err_cols) > 0:
                y_err = y_err[0]

            
        # de-log
        if delog == True:
            y = np.array(y)
            y = 2.0**y
            
            # get fold change
            y_min = np.min(y)
            y = y/y_min
            y = list(y)
        
        # add to list of y values
        y_list.extend(y)
        
        if len(err_cols) > 0:
            y_err_list.extend(y_err)
        
        # if site id was used, output the protein name
        prot_label = subset_df.Gene_Symbol[subset_df[key_col] == 
                                    prot_ids[i]].tolist()[0]
        
        if len(err_cols) > 0:
            # plotting dots and dashes
            plt.fill_between(x, y+y_err, y-y_err,
                                 color=colors[i], 
                                 alpha=0.5)
            plt.plot(x, y, '-', color=colors[i], linewidth=3, label=prot_label)
        else:
            plt.plot(x, y, '-', color=colors[i], linewidth=3, label=prot_label)
            plt.plot(x, y, 'o', color=colors[i], linewidth=3, label=prot_label)
            
        
        if not log_data:
            plt.ylabel('Fold Change')
        else:
            plt.ylabel('Log$_2$(FC)')
        
        # on last iteration, set axis parameters
        if i == len(prot_ids)-1:
            
            if len(err_cols) > 0:
                ylist_min = min(np.array(y_list)-np.array(y_err_list)); x_min = min(x)
                ylist_max = max(np.array(y_list)+np.array(y_err_list)); x_max = max(x)
            else:
                ylist_min = min(np.array(y_list)); x_min = min(x)
                ylist_max = max(np.array(y_list)); x_max = max(x)
            y_range = ylist_max - ylist_min; x_range = x_max - x_min
            
            plt.xlim([x_min, x_max+x_range/20.])
            
            if len(ylim) != 0:
                plt.ylim(ylim)
            else:
                plt.ylim([ylist_min-y_range/10., ylist_max+y_range/10.])
            
            # if the length of first element is 1, assuming time
            if len(str(xlabels[0])) == 1:
                xaxis = 'Time (min)'
                plt.xlabel(xaxis)
        
            set_plot_params(fig, ax, xlabels, [], [])
            # not doing "fig.tight_layout()" if overlaying
            # it makes it difficult to visualize the different protein levels
            plt.gca().tick_params(axis='both', which='major', 
                                  labelsize=25)
            
            #plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45)
           
        if bg == 'black':
            black_plot_param(fig, ax, True)
            
    if len(savefig) > 0:
        fig.savefig(savefig, dpi=int(300))
        

##########################################################################################


# function to get certain rows based on input keywords
# MS_df is the input mass spec dataset
# keywords argument is a list of keywords
def get_MS_subset(MS_df, keywords, quant_cols):
    
    if isinstance(keywords, str):
        keywords = [keywords]
    
    # making copy of dataframe
    MS_df_copy = MS_df.copy()

    # initializing empty dataframe to concatenate to
    MS_df_subset = pd.DataFrame()
    
    # iterating through keywords
    for i in range(len(keywords)):
        
        # as long as the keywords are specific enough this should be ok
        keyword_i = keywords[i].lower()     
        
        # making a subset dataframe
        MS_df_subset_i = MS_df_copy
        
        # find() outputs index position if the substring appears
        # also outputs a -1 if the word does not appear
        keyword_idx = MS_df_subset_i.Description.apply(lambda x: 
            x.lower().find(keyword_i))
        keyword_idx[keyword_idx >= 0] = 1
        keyword_idx[keyword_idx < 0] = 0
        
        # adding temporary column and taking subset
        MS_df_subset_i['Has_Keyword'] = keyword_idx
        MS_df_subset_i = MS_df_subset_i[MS_df_subset_i.Has_Keyword == 1]
        MS_df_subset_i = MS_df_subset_i.drop('Has_Keyword', axis=1)
        
        # concatenating dataframes
        MS_df_subset = pd.concat([MS_df_subset, MS_df_subset_i]).reset_index(drop=True)
        
    return MS_df_subset
        
    
##########################################################################################


# get "FUNCTION" section of uniprot protein information
def uniprot_to_func(uniprot_str):
            
    if uniprot_str == None:
        return None

    # if function header is not present, give up
    fam_header = 'FUNCTION: '
    fam_idx = uniprot_str.find(fam_header)
    if fam_idx == -1:
        return None

    # now left side begins with the function information
    subset_str = uniprot_str[fam_idx + len(fam_header):]

    # making string a bit shorter by going up to second period
    # if not second period, then to first period
    # if no period, then look for exclamation point
    # if no exclamation point, we're at the end of the uniprot info
    first_per = subset_str.find('.')
    second_per = subset_str[first_per+1:].find('.')
    if second_per != -1:
        second_per = second_per + first_per+1
        next_punc = second_per
    else:
        next_punc = subset_str.find('!')
        if next_punc == -1:
            next_punc = subset_str.find('---')

    subset_str = subset_str[:next_punc].replace('-', ' ')
    
    return subset_str
    

# not perfect, but still a solid function to assign protein families
# NOTE: user must enter either seq_id or description, or both
def assign_prot_fam(gene_symbol, seq_id=[], description=[]):
        
    """
    looking for: kinase, phosphatase, GEF, GAP, GTPase, 
    microtubule-associated, mitochondrial, RNA-binding protein
    """
    
    kinase_list = pd.read_csv('/Users/javier/Desktop/Mitchison_Lab/Coding_Stuff/uniprot_kinase_list_04132015.csv')
    ppase_list = pd.read_csv('/Users/javier/Desktop/Mitchison_Lab/Coding_Stuff/phosphatase_list_04132015.csv')

    # checking if it's a kinase
    if gene_symbol in kinase_list.Gene_Symbol.tolist():
        return 'Kinase'
    
    # checking if it's a phosphatase
    if gene_symbol in ppase_list.Gene_Symbol.tolist():
        return 'Phosphatase'
    
    if not isinstance(description, list):
    
        # this happens if we're using recorded uniprot info
    	if description == None:
            return 'Other'
        
        # check if the string consists of uniprot information
        if 'uniprot' in description.lower():
            
            # in this case, get only the "FUNCTION" part of the string
            description = uniprot_to_func(description)
            
            # if "FUNCTION" not present in uniprot string, return 'Other'
            if description == None:
            	return 'Other'
    
        description_case = description.replace('-',' ')
        description = description.lower().replace('-',' ')
        
        ### well-annotated protein families
        
        # checking if it is associated with mitochondria
        mito_str = 'mitochondria'
        if mito_str in description:
        	return 'Mitochondrial-associated'
        	            
        # checking if it's ribosomal
        ribo_str = 'ribosom'
        if ribo_str in description:
        	return 'Ribosomal'
        	
        # checking if it's an HDAC
        hdac_str1 = 'histone'
        hdac_str2 = 'deacetyl'
        if (hdac_str1 in description) & (hdac_str2 in description):
        	return 'HDAC'
    
        # checking if it's a transcription factor
        tf_str = 'transcription factor '
        if tf_str in description:
            return 'TF'
        
        # ok-annotated protein familes
        
        # checking if it's a GEF
        gef_str = 'guanine nucleotide exchange'
        if gef_str in description:
            if 'like' not in description[description.index(gef_str) + 
                                     len(gef_str):]:
                if 'binding' not in description[description.index(gef_str) + 
                                     len(gef_str):]:
                    return 'GEF'
            
        # checking if it's a GAP
        gap_str = 'gtpase activating protein'
        if gap_str in description:
            if ' like' not in description[description.index(gap_str) + 
                                     len(gap_str):]:
                if ' binding' not in description[description.index(gap_str) + 
                                     len(gap_str):]:
                    return 'GAP'
            
        # checking it it's a GTPase
        gtp_str = 'gtpase'
        if gtp_str in description:
            if ' like' not in description[description.index(gtp_str) + 
                                     len(gtp_str):]:
                if ' binding' not in description[description.index(gtp_str) + 
                                     len(gtp_str):]:
                    return 'GTPase'
        
        
        ## not-so-well annotated protein familes
        
        # checking if it's associated with microtubules
        mt_str = 'microtubule'
        if mt_str in description:
            return 'MAP'
           
        # checking if it's associated with actin
        act_str = 'actin'
        if act_str in description:
        
            # if 'actin' is the last word of the string, add a space at the end
            act_pos = description.find(act_str)
            if act_pos+len(act_str) == len(description):
                description = description+' '
                
            # if 'actin' is the first word of the string, add a space at the beginning
            if act_pos == 0:
                description = ' '+description
            
            # in case the word might be 'acting' or 'interacting', etc
            # in case the word might be 'centractin', etc
            let_after_str = description[description.index(act_str)+len(act_str)]
            let_before_str = description[description.index(act_str)-1]
            
            if not (let_after_str.isalpha() | let_before_str.isalpha()):
                return 'AAP'
            
            else:
                new_subset_str = description
                actin_artifact = True
                while (act_str in new_subset_str) and actin_artifact:
                    
                    # removing the artifact
                    new_subset_str = new_subset_str[new_subset_str.index(act_str)+len(act_str):]
                    
                    act_pos = new_subset_str.find(act_str)
                    if act_pos == -1:  # if actin string not found, while loop will break
                        continue
                    
                    # if actin string present, we need to check the preceding and following character
                    # this is in case actin is at the beginning or end of the string
                    if act_pos+len(act_str) == len(new_subset_str):
                        new_subset_str = new_subset_str+' '
                        
                    if act_pos == 0:
                        new_subset_str = ' '+new_subset_str
                    
                    let_after_str = new_subset_str[new_subset_str.index(act_str)+len(act_str)]
                    let_before_str = new_subset_str[new_subset_str.index(act_str)-1]
                    
                    if not (let_after_str.isalpha() | let_before_str.isalpha()):
                        actin_artifact = False
                        
                if act_str in new_subset_str:
                    return 'AAP'
                else:
                    return 'Other'
              
        # checking if it is associated with RNA processes
        rna_str = 'RNA'
        if rna_str in description_case:
            return 'RNA-associated'  
        else:
            return 'Other'
    

    if not isinstance(seq_id, list):
        # if none of the above work, check for microtubule association
        uniprot_str = get_uniprot_str(seq_id)
            
        # getting "FUNCTION" section of uniprot string
        subset_str = uniprot_to_func(uniprot_str)
        if subset_str == None:
        	return 'Other'
        
        # checking if it's associated with microtubules
        mt_str = 'microtubule'
        if mt_str in subset_str.lower():
            return 'MAP'
        
        # checking if it's associated with actin
        act_str = 'actin'
        if act_str in subset_str.lower():
        
            # making a copy of the string
            lower_str = subset_str.lower()
        
            # if 'actin' is the last word of the string, add a space at the end
            act_pos = lower_str.find(act_str)
            if act_pos+len(act_str) == len(lower_str):
                lower_str = lower_str+' '
                
            # if 'actin' is the first word of the string, add a space at the beginning
            if act_pos == 0:
                lower_str = ' '+lower_str
        
            # in case the word might be 'acting' or 'interacting', etc
            # in case the word might be 'centractin', etc
            let_after_str = lower_str[lower_str.index(act_str)+len(act_str)]
            let_before_str = lower_str[lower_str.index(act_str)-1]
            
            if not (let_after_str.isalpha() | let_before_str.isalpha()):
                return 'AAP'
            
            else:
                new_subset_str = lower_str
                actin_artifact = True
                while (act_str in new_subset_str) and actin_artifact:
                    
                    new_subset_str = new_subset_str[new_subset_str.index(act_str)+len(act_str):]
                    
                    act_pos = new_subset_str.find(act_str)
                    if act_pos == -1:  # if actin string not found, while loop will break
                        continue
                    
                    # if actin string present, we need to check the preceding and following character
                    # this is in case actin is at the beginning or end of the string
                    if act_pos+len(act_str) == len(new_subset_str):
                        new_subset_str = new_subset_str+' '
                        
                    if act_pos == 0:
                        new_subset_str = ' '+new_subset_str
                    
                    let_after_str = new_subset_str[new_subset_str.index(act_str)+len(act_str)]
                    let_before_str = new_subset_str[new_subset_str.index(act_str)-1]
                    
                    if not (let_after_str.isalpha() | let_before_str.isalpha()):
                        actin_artifact = False
                        
                if act_str in new_subset_str:
                    return 'AAP'
                
        # checking if it's a transcription factor
        tf_str = 'transcription factor '
        if tf_str in subset_str.lower():
            return 'TF'
            
        # checking if it's a GEF
        gef_str = 'guanine nucleotide exchange'
        if gef_str in subset_str.lower():
            if 'like' not in subset_str.lower()[subset_str.lower().index(gef_str) + 
                                         len(gef_str):]:
                if 'binding' not in subset_str.lower()[subset_str.lower().index(gef_str) + 
                                         len(gef_str):]:
                    return 'GEF'
                
        # checking if it's a GAP
        gap_str = 'gtpase activating protein'
        if gap_str in subset_str.lower():
            if ' like' not in subset_str.lower()[subset_str.lower().index(gap_str) + 
                                         len(gap_str):]:
                if ' binding' not in subset_str.lower()[subset_str.lower().index(gap_str) + 
                                         len(gap_str):]:
                    return 'GAP'
                
        # checking if it's a GTPase
        gtp_str = 'gtpase'
        if gtp_str in subset_str.lower():
            if ' like' not in subset_str.lower()[subset_str.lower().index(gtp_str) + 
                                         len(gtp_str):]:
                if ' binding' not in subset_str.lower()[subset_str.lower().index(gtp_str) + 
                                         len(gtp_str):]:
                    return 'GTPase'
                                             
        # check well-annotated proteins again, just in case
            
        # checking if it is associated with mitochondria
        mito_str = 'mitochondria'
        if mito_str in subset_str.lower():
            return 'Mitochondrial-associated'
                
        # checking if it's ribosomal
        ribo_str = 'ribosom'
        if ribo_str in subset_str.lower():
            return 'Ribosomal'
                
        # checking if it's an HDAC
        hdac_str1 = 'histone'
        hdac_str2 = 'deacetyl'
        if (hdac_str1 in subset_str.lower()) & (hdac_str2 in subset_str.lower()):
            return 'HDAC'
        
        # check once more if it's RNA-associated
        rna_str = 'RNA'
        if rna_str in subset_str:
            return 'RNA-associated'
        else:
            return 'Other'

            
##########################################################################################


# making function to assign colors based on column
# 'hm_color' is the color scale of the heatmap paired with this function (either rwb or rkc)
def assign_col_color(ms_df, hm_color='rkc', col='Family', prots=True, bg=[]):
    
    ms_df_copy = ms_df.copy()          # making copy of dataframe
    

    colors = [[0, 0.3, 0], 
              [1.0, 1.0, 0],
              [0, 0.9, 0.9],
              [0.95, 0, 0],
              [0.75, 0, 0.9],
              [1.0, 0.5, 0], 
              [0, 0, 0.9],
              [0.75, 0.75, 0.75],
              [0.25, 0.25, 0.25], 
              [0, 0, 0], 
              [0.5, 0.5, 0], 
              [0, 0.9, 0],
              [0, 0, 0.5]]
              
    
    if prots:
        prot_fams = ['GAP', 'GEF', 'Kinase', 'Phosphatase', 'MAP', 
                      'HDAC', 'Ribosomal', 'RNA-associated', 'Other', 
                      'Mitochondrial-associated', 'GTPase', 'TF', 'AAP']
        colors_dict = {}
        for c, prot in enumerate(prot_fams):
            colors_dict[prot] = colors[c]
            
        colors = colors_dict  # resetting variable
        
    elements = sorted(ms_df_copy[col].unique())

    # number of colors should be greater than or equal to number of locations
    assert len(colors) >= len(elements)    
    
    ms_df_copy[col+'_colors'] = np.zeros(shape=(len(ms_df_copy), 3)).tolist()
    
    # performing color assignment
    color_list = []
    for i, el in enumerate(elements):
        
        # making a element/row specific color assignment
        row_num = len(ms_df_copy[ms_df_copy[col] == elements[i]])
        
        if prots:
            element_i_colors = np.tile(colors[el], (row_num, 1)).tolist()
        else:
            element_i_colors = np.tile(colors[i], (row_num, 1)).tolist()
        
        # storing color assignment
        color_list.extend(element_i_colors)

        
    # merging color assignments
    ms_df_copy = ms_df_copy.sort(col)
    ms_df_copy[col+'_colors'] = color_list
    ms_df_copy = ms_df_copy.sort()
        
    # finalizing color assignment (format modification)
    ms_df_copy[col+'_colors'] = ms_df_copy[col+'_colors'].apply(lambda x: 
                                                           tuple(x))
    return ms_df_copy
    
    
def nan2med(array):
    array[np.isnan(array)] = np.nanmedian(array)
    return array


# making heatmap function for ms datasets
# input dataframe should already be normalized to have FC values
# 'phos_nonphos' can be 'phos', 'nonphos', or 'both'
# 'metric' can be 'spearman', 'pearson', or 'none'
def ms_heatmap(ms_df, quant_cols, xlabels=[], data_type='time series',  log2fc=1, prot_num=50, 
               metric='pearson', inc_family=False, mark_jnk=False, add_col_to_ylabel=[], figsize=(12, 8)):
    
    
    metric = metric.lower()
    fc_df = ms_df.copy().reset_index(drop=True)
    
    
    # correct septins and march proteins
    fc_df['Gene_Symbol'] = fc_df.Gene_Symbol.apply(lambda x: 
                'SEPT'+str(x)[:str(x).index('-')] if str(x)[-4:].upper() == '-SEP' else str(x).upper())
    fc_df['Gene_Symbol'] = fc_df.Gene_Symbol.apply(lambda x: 
                'MARCH'+str(x)[:str(x).index('-')] if str(x)[-4:].upper() == '-MAR' else str(x).upper())    
    
    # x-axis labels
    if len(xlabels) == 0:
        xlabels = quant_cols
    
    
    
    ### data filtering section ###
    
    # notating which rows meet the log2fc requirement
    fc_df['sig_fc'] = fc_df[quant_cols].apply(lambda x: 
                np.any(np.array(x) >= log2fc) | 
                np.any(np.array(x) <= -log2fc), axis=1)
    fc_df = fc_df[fc_df.sig_fc == True]
    
    
    # sorting by descending variance
    fc_df['Var'] = fc_df[quant_cols].apply(lambda x: np.var(np.array(x)), axis=1)

    if metric.lower() != 'none':
        fc_df = fc_df.sort('Var', ascending=False).reset_index(drop=True)
    
    # if there are fewer proteins in the resultant dataset than in the input protein number
    # take all the proteins available
    if len(fc_df) < prot_num:
        prot_num = len(fc_df)
    
    # making copy of filtered dataset
    fc_df = fc_df[fc_df.index <= prot_num-1]
    
    # make different version that replaces NaNs with row medians
    fc_df_copy = fc_df.copy()
    fc_df_copy[quant_cols] = fc_df_copy[quant_cols].apply(lambda x: nan2med(x), axis=1)
    
    print 'Number of proteins/peptides being used for clustering: %s' % str(prot_num)
    
    
    
    ### dendogram section ###
    
    # dendrogram for proteins (AXES = ax1)
    # getting array of values
    prot_array = np.array(fc_df_copy[quant_cols])
    
    # getting correlation distance
    if metric == 'spearman':
        prot_pdist = combos_spdist(prot_array)
    
    elif metric == 'pearson':
        prot_pdist = pdist(prot_array, 'correlation') 
        prot_pdist[prot_pdist < 0] = np.max(prot_pdist)   
    
    if metric != 'none':
        prot_links = sch.linkage(prot_pdist, 'average')      # getting linkage

    # setting first axis for prot dendrogram
    fig, ax1 = plt.subplots(figsize=figsize)
    
    ax1.patch.set_visible(False)
    
    # setting dendrogram color
    if metric != 'none':
        sch.set_link_color_palette(['grey'])
        
        # getting prot dendrogram
        prot_dend = sch.dendrogram(prot_links, orientation='right', 
                               color_threshold=np.inf)
        
    else: 
        sch.set_link_color_palette(['none'])
    
    yes_border_no_ticks(ax1, False)    # taking away ticks and frame
    ax1.set_xticks([])
    ax1.set_yticks([])
        
    # resetting position of prot dendrogram
    pos1 = ax1.get_position()
    pos1_new = [pos1.x0, pos1.y0, pos1.width/10., pos1.height]
    ax1.set_position(pos1_new)
    
    if metric != 'none':
 
        # getting the new prot order; will be reflected in heatmap
        new_prot_order = np.array(map(int, prot_dend['ivl']))
    
        # re-order dataframe
        fc_df = fc_df.ix[new_prot_order, :].reset_index(drop=True)
    
    # user inputs a column name to add to the y-axis labels
    if isinstance(add_col_to_ylabel, str):
        fc_df['y_labels'] = fc_df[['Gene_Symbol', 
                                   add_col_to_ylabel]].apply(lambda x: 
                                    str(x[0])+
                                    '     ('+str(x[1])+')', axis=1)
       
    # user doesn't input an extra column name
    else:
        fc_df['y_labels'] = fc_df.Gene_Symbol
        
    y_labels = np.array(list(fc_df.y_labels)).tolist()
    
    # dendrogram for cell states (AXES = ax2)
    cs_array = np.array(fc_df_copy[quant_cols]).T

    
    
    ##################################################################################
    # NOTE: do not use a fold-change dataset without setting data_type='time series' #
    
    # getting correlation distance
    if (data_type.lower() != 'time series') & (metric == 'spearman') :
        cs_pdist = combos_spdist(cs_array)
    
    elif (data_type.lower() != 'time series') & (metric == 'pearson'):
        cs_pdist = pdist(cs_array, 'correlation')
    
    # if using time series, just make an invisible dendogram as a reference for axes
    elif (data_type.lower() == 'time series') | (metric == 'none'):
        cs_pdist = pdist(cs_array, 'euclidean')  # must be euclidean or it will crash
    
    cs_links = sch.linkage(cs_pdist, 'average')
    
    ##################################################################################
    
    
    
    # setting position for cs dendogram
    pos1 = ax1.get_position()
    pos2 = [pos1.x0 + pos1.width, pos1.y0 + pos1.height, 
            pos1.width*3.0, pos1.height*2./3]

    ax2 = fig.add_axes(pos2)
    ax2.patch.set_visible(False)
    
    # getting cs dendrogram
    # even if this dendogram is not desired (i.e. time series data),
    # will still make it to have position reference for heatmap
    if data_type.lower() == 'time series':
        
        # setting dendrogram color
        sch.set_link_color_palette(['none'])
    
    if metric != 'none':
        cs_dend = sch.dendrogram(cs_links, orientation='top', 
                    color_threshold=np.inf, distance_sort='ascending')
        
        # getting new cell state order; will be reflected in the heatmap
        new_cs_order = np.array(map(int, cs_dend['ivl']))
    
    yes_border_no_ticks(ax2, False)   # taking away ticks and frame
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # if dealing with time series data, do NOT change ordering
    if (data_type.lower() != 'time series') & (metric != 'none'):
        x_labels = np.array(xlabels)[new_cs_order].tolist()
    else:
        x_labels = np.array(xlabels).tolist()
    
    
    ### label set up section ###
    
    x_num = len(quant_cols)    # number of cell states
    y_num = prot_num           # number of proteins
    
    # labels for actual plot
    pos_x = np.arange(x_num)+0.5
    pos_y = np.arange(y_num)+0.5
    
    
    # getting protein fold change values
    # reordering the columns according to the new cell state order
    if (data_type.lower() != 'time series') & (metric != 'none'):
        val_mat = np.array(fc_df[quant_cols])[:, new_cs_order]
    else: 
        val_mat = np.array(fc_df[quant_cols])
        
    max_val = np.nanmax(val_mat)    # maximum positive log2fc
    min_val = np.nanmin(val_mat)    # maximum negative log2fc
    
    # max absolute value fold change
    max_val = np.array([abs(max_val), abs(min_val)]).max() 
    
    
    
    ### heatmap plotting section (AXES = ax3) ###
    
    # plotting the figure
    pos2 = ax2.get_position()
    pos3 = [pos2.x0, pos1.y0, 
            pos2.width, pos1.height]
    
    ax3 = fig.add_axes(pos3)
    ax3.patch.set_visible(False)

    
    discrete_colors = 501
    cmap = cl.LinearSegmentedColormap.from_list(name='b_w_r', 
                                     colors=[(0.0, 0.5, 0.75), 
                                             (1.0, 1.0, 1.0), 
                                             (1.0, 0.0, 0.0)], 
                                             N=discrete_colors)

    if prot_num < 100:
        nan_color = (0.8, 0.8, 0.8)
    else:
        nan_color = 'w'
        
        
    ##
    nan_color = (0.95, 0.95, 0.95)
    ##
    
    cmap.set_bad(color=nan_color, alpha=1.0)

    
    ax3.yaxis.tick_right()     # putting labels on the right
    yes_border_no_ticks(ax3)
    
    # setting tick parameters
    ax3.set_xticks(pos_x)
    
    if (data_type != 'time series') | (True in [l > 3 for l in [len(str(x)) for x in xlabels]]):
        rot = 90.0
    else:
        rot = 0.0
        
    ax3.set_xticklabels(x_labels, rotation=rot, fontsize=8)
    ax3.set_ylim([0, len(fc_df)]) # important...
    
    
    # for ax.tick_params(), axis can equal 'x', 'y', or 'both'
    # setting y-axis label fontsize
    if prot_num > 80:
        font_size = 6
    elif prot_num > 65:
        font_size = 7
    elif prot_num > 50:
        font_size = 8
    elif prot_num > 35:
        font_size = 9
    elif prot_num > 25:
        font_size = 10
    elif prot_num > 15:
        font_size = 11
    else:
        font_size = 12
        
    # placing the heatmap; vmin and vmax set the range of the colors
    ax3.pcolormesh(np.ma.masked_invalid(val_mat), cmap=cmap, vmin=-max_val, vmax=max_val)
    ax3.patch.set(edgecolor='black')
    
        
    if inc_family | mark_jnk:
        ax3.set_yticks([])  
    elif prot_num > 100:
        ax3.set_yticks([])
    else:
        ax3.set_yticks(np.arange(prot_num)+0.5)
        ax3.set_yticklabels(y_labels)   
        ax3.tick_params(axis='y', which='major', labelsize=font_size)
    

    ### adding color legend for fold change (AXES = ax4) ###
    max_floor = int(np.floor(max_val))
    
    if (max_floor > 0) & (max_floor < 4):
        right_labels = [0] + list(np.arange(max_floor)+1)
        left_labels = list(-(np.arange(max_floor)+1))[::-1]
    
    elif (max_floor > 0) & (max_floor >= 4):
        right_labels = [0] + list(np.arange(max_floor)[1::2]+1)
        left_labels = list(-(np.arange(max_floor)[1::2]+1))[::-1]

    else:
        right_labels = [0] + [np.around(max_val, 1)]
        left_labels = [-np.around(max_val, 1)]
        

    col_labels = left_labels + right_labels


    # adding a second axes to the right
    pos3 = ax3.get_position()
    
    pos4 = [pos1.x0*1.15, pos2.y0*1.2, 
            pos1.width/2.5*12.0/figsize[0], pos2.height/2.25]
    
    ax4 = fig.add_axes(pos4)
    ax4.patch.set_visible(False)
    
    # plotting the color legend
    fill = np.tile(np.linspace(-max_val, max_val, num=500), (1, 1))
    ax4.pcolor(fill.T, cmap=cmap, vmin=-max_val, vmax=max_val)
    
    # setting xticks on color legend
    right_xticks = [abs(float(tick))*1.0/max_val*250.0+250.0 for tick in np.array(right_labels)] 
    left_xticks = [250.0-abs(float(tick))*1.0/max_val*250.0 for tick in np.array(left_labels)]
    ax4.set_yticks(left_xticks+right_xticks)
        
    ax4.set_ylim([0, 500])    
    ax4.set_yticklabels(col_labels)
    ax4.set_xticks([])
    

    ax4.set_ylabel('Log$_2$(FC)')

    
    yes_border_no_ticks(ax4)
    
    
    ### family color bar section (AXES = ax5) ###
   
    if inc_family:
    
        # assigning colors to proteins
        old_prots_fams = fc_df[['Gene_Symbol', 'Family']]
        sorted_df = pd.DataFrame(y_labels, columns=['Gene_Symbol'])
        sorted_df['Gene_Symbol'] = sorted_df.Gene_Symbol.apply(lambda x: 
                                    x[:x.index(' ')] if ' ' in x else x)
        
        # important to drop duplicates in this step when merging...
        sorted_df = sorted_df.merge(old_prots_fams[['Gene_Symbol', 'Family']].drop_duplicates(), 
                                    on='Gene_Symbol', how='left', copy=False)

        ms_col_df = assign_col_color(sorted_df)
        

        # getting color list
        fam_colors = list(ms_col_df.Family_colors)
  
        # numbers to plot to get nice bar graph
        colorx1 = np.arange(len(ms_col_df))
        colory1 = np.ones(len(ms_col_df))
    
        pos5 = [pos3.x0 + pos3.width, pos3.y0, 
                pos1.width*5.0/len(quant_cols), pos3.height]

        ax5 = fig.add_axes(pos5)
        ax5.patch.set_visible(False)
        
        # plotting horizontal bar graph
        ax5.barh(colorx1, colory1, color=fam_colors, 
                 height=1.0, edgecolor='none')
        ax5.yaxis.tick_right()
        
        ax5.set_xticks([])
        
        yes_border_no_ticks(ax5)
        
        if mark_jnk:
            ax5.set_yticks([])
        elif prot_num > 100:
            ax5.set_yticks([])
        else:
            ax5.set_yticks(np.arange(prot_num)+0.5)
            ax5.set_yticklabels(y_labels)
            ax5.tick_params(axis='y', which='major', labelsize=font_size)
            
        ax5.set_ylim([0, len(colorx1)])    
            
            
        ### adding color legend for protein family (AXES = ax6) ###
        
        # making small dataframe of unique families and colors
        unq_fam_df = pd.DataFrame(sorted(fc_df.Family.unique()), 
                                  columns=['Family'])

        unq_fam_df = assign_col_color(unq_fam_df)
        unq_fam_colors = list(unq_fam_df.Family_colors)
        unq_fam_labels = list(unq_fam_df.Family)
        
        
        # numbers to plot to get nice bar graph
        colorx2 = np.arange(len(unq_fam_labels))
        colory2 = np.ones(len(unq_fam_labels))
        
        # setting axis for family legend
        pos4 = ax4.get_position()
        
        pos6 = [pos4.x0+pos4.width-pos4.height*3./4, 
                pos4.y0+pos4.height*2.5, 
                pos4.height*3./4, pos4.width]
        
        ax6 = fig.add_axes(pos6)
        ax6.patch.set_visible(False)
        
        # plotting vertical bar graph
        
        # alternative horizontal bar graph (i.e. vertical plot)
        ax6.barh(colorx2, colory2, color=unq_fam_colors, 
                 height=1.0, edgecolor='none')
        ax6.set_xticks([])
        ax6.set_yticks(np.arange(len(colorx2))+0.5)
        ax6.set_ylim([0, len(colorx2)])
        ax6.set_yticklabels(unq_fam_labels)
        
        # setting legend font size
        if len(unq_fam_colors) > 9:
            ax6.tick_params(axis='y', which='major', labelsize=12)
        else:
            ax6.tick_params(axis='y', which='major', labelsize=14)
            
        yes_border_no_ticks(ax6)
        
        
    ### marking phosphorylated peptide sequences section ###
        
    if mark_jnk:
        
#         # indicating whether phos peptide dependent on JNK
#         mark_jnk_df = fc_df[['Gene_Symbol', 'Sequence', 'JNK_dep']].copy()
#         mark_jnk_df['Mark_JNK'] = mark_jnk_df.JNK_dep.apply(lambda x: 
#                                         (0.0, 0.0, 0.0) if (x == True) else (1.0, 1.0, 1.0))
        
#         # marking proline-directed sites
#         mark_jnk_df = fc_df[['Gene_Symbol', 'Sequence']].copy()
#         mark_jnk_df['Mark_JNK'] = mark_jnk_df.Sequence.apply(lambda x: 
#                     (0.0, 0.0, 0.0) if ('S#P' in x) | ('T#P' in x) else (0.0, 0.0, 0.0) if 'Y#P' in x else (1.0, 1.0, 1.0))
        
        # marking nuclear proteins
        mark_jnk_df = fc_df[['Gene_Symbol', 'Location']].copy()
        mark_jnk_df['Mark_JNK'] = mark_jnk_df.Location.apply(lambda x: 
                      (0, 0, 0) if (('nucleus' in str(x).lower()) & 
                      (('membrane' not in str(x).lower()) & 
                      ('envelope' not in str(x).lower()))) else (1, 1, 1))
        
        
        
        
        # getting color list
        mark_jnk_colors = list(mark_jnk_df.Mark_JNK)
  
        # numbers to plot to get nice bar graph
        colorx3 = np.arange(len(mark_jnk_df))
        colory3 = np.ones(len(mark_jnk_df))
        
        # alternative horizontal bar graph (i.e. vertical plot)
        
        if inc_family:
            pos5 = ax5.get_position()
        
            pos7 = [pos5.x0 + pos5.width, pos5.y0,
                    pos1.width*5.0/len(quant_cols), pos5.height]
            
        else:

            pos7 = [pos3.x0 + pos3.width, pos3.y0, 
                pos1.width*5.0/len(quant_cols), pos3.height]
            
        ax7 = fig.add_axes(pos7)
        ax7.patch.set_visible(False)

        ax7.barh(colorx3, colory3, color=mark_jnk_colors, 
                 height=1.0, edgecolor='none')
        ax7.yaxis.tick_right()
        
        ax7.set_xticks([])
        
        if prot_num > 100:
            ax7.set_yticks([])
        else:
            ax7.set_yticks(np.arange(prot_num)+0.5)
            ax7.set_yticklabels(y_labels)
            ax7.tick_params(axis='y', which='major', labelsize=font_size)
        
        ax7.set_ylim([0, len(mark_jnk_colors)])
        
        yes_border_no_ticks(ax7)

   
    # showing the plot
    plt.show()
    
    
    # outputing the truncated dataframep
    if metric != 'none':
        output_df = fc_df.ix[::-1, :].reset_index(drop=True)
    
        # get rid of unnecessary added columns
        output_df = output_df.drop(['sig_fc', 'Var', 'y_labels'], axis=1)
    
        return output_df, fig
     

##########################################################################################

# these functions allow the user to compute p-values for MS data with replicates
# would be better to calculate confidence intervals instead if possible

# better shuffle function
def shuffle_array(array):
    
    array_copy = array.copy()
    random.shuffle(array_copy)
    
    # returning shuffled array
    return array_copy
    
    
# making function to compute confidence score from euclidean dist calculations
# this function assumes duplicates are present, but no triplicates
# also assuming that the corresponding duplicates are in the same order
def get_dist_conf(array, dup_idx, testing_num=10000):
    
    """ 
    array: numpy array that we are testing distances for
    
    num_dups: number of duplicates in the array
    
    dup_idx: indices of the duplicate measurements; nested list 
             i.e. first inner list corresponds to the indices of
             the first set of duplicates (dup_idx[0])
    
    """
    
    dup1_idx = dup_idx[0]          # duplicate index arrays
    dup2_idx = dup_idx[1]
    dup1_array = array[dup1_idx]   # duplicate arrays
    dup2_array = array[dup2_idx]
    
    # number of duplicates
    num_dups = len(dup1_idx)
   
    # original calculated distance
    orig_dist = distance.euclidean(dup1_array, dup2_array)
    
    # making tiled array
    tiled_array = np.tile(array, (testing_num, 1))
    
    # shuffling each row of tiled_array
    for row in range(testing_num):
        tiled_array[row, :] = shuffle_array(tiled_array[row, :])
    
    # making dataframes with shuffled versions of "duplicates"
    # i.e. taking the duplicate indices as before
    col_labels = ['col'+str(i+1) for i in range(num_dups)]
    
    shuffle_vec1 = pd.DataFrame(tiled_array[:, dup1_idx], 
                                 columns=col_labels)
    shuffle_vec2 = pd.DataFrame(tiled_array[:, dup2_idx], 
                                 columns=col_labels)
    
    # putting arrays within single cells 
    shuffle_vec1['vector'] = shuffle_vec1[col_labels].apply(tuple, 
                                                          axis=1)
    shuffle_vec1['vector'] = shuffle_vec1['vector'].apply(lambda x: 
                                                        list(x))
    shuffle_vec2['vector'] = shuffle_vec2[col_labels].apply(tuple, 
                                                          axis=1)
    shuffle_vec2['vector'] = shuffle_vec2['vector'].apply(lambda x: 
                                                        list(x))
   
    # making new dataframe with only the two vectors
    vectors_df = pd.DataFrame(np.array(shuffle_vec1['vector']), 
                                  columns=['vector1'])
    vectors_df['vector2'] = np.array(shuffle_vec2['vector'])
        
    # calculating new distances
    vectors_df['dist'] = vectors_df[['vector1', 
                                     'vector2']].apply(lambda x: 
                            distance.euclidean(x[0], x[1]), axis=1)
    
    # getting all distances
    all_dist_array = vectors_df.dist.values
    
    pval = round(np.sum(all_dist_array < orig_dist)*1.0/testing_num, 5)
    conf = 1.0 - pval
    
    return conf  
                  

##########################################################################################


# these are functions that characterize protein sequences

# making function to get percent of charged AAs
# 'seq' is a string of the protein sequence
def charged_aa(seq):

    num_charged = seq.count('D') + seq.count('E') + seq.count('K') + seq.count('R')
    return num_charged*1.0/len(seq)
 
    
# making function to get percent of non-polar AAs
def nonpolar_aa(seq):
    
    num_nonpolar = seq.count('A') + seq.count('V') + seq.count('L') + seq.count('I'
                    ) + seq.count('M') + seq.count('W') + seq.count('F') + seq.count('P')  
    return num_nonpolar*1.0/len(seq)


# making function to get number of theoretical peptides
# written for trypsin as digestive enzyme
def theor_peptide(seq, enzyme='lysc_trypsin', miscleav_num=False):
    
    # NOTE: trypsin does not cleave lys or arg if followed by proline
    
    # cutting after lysine
    peptides_k = seq.replace('K', 'K-').split('-')
    
    # NOTE: lysc only cleaves after lysine
    if 'trypsin' in enzyme.lower():
        
        # cutting after arginine
        peptides_r = sum([i.replace('R', 'R-').split('-') for i in peptides_k], [])
    else:
        peptides_r = peptides_k
    
    # if last AA is a lysine or arginine, the above lines will leave an empty string
    if peptides_r[-1] == '':
        peptides_r.pop(-1)
    
    # identifying peptides that start with proline
    # need to connect these with the preceding peptide
    if 'lysc' not in enzyme.lower():
        KP_RP_inst = []
        for i in range(len(peptides_r)):

            if peptides_r[i][0] == 'P':
                peptides_r[i-1] = peptides_r[i-1] + peptides_r[i]
                KP_RP_inst.append(i)

        # removing the peptides that start with proline
        # starting with the last of such peptides and working backwards
        if len(KP_RP_inst) > 0:
            for i in range(len(KP_RP_inst)):
                pop_idx = KP_RP_inst[-(i+1)]
                peptides_r.pop(pop_idx)
    
    if not miscleav_num:
        return len(peptides_r)
    else:
        return len(peptides_r)-1
    
    
# making function to get number of short theoretical peptides
def short_peptide(seq):
    
    # NOTE: trypsin does not cleave lys or arg if followed by proline
    
    # cutting after lysine
    peptides_k = seq.replace('K', 'K-').split('-')
    
    # cutting after arginine
    peptides_r = sum([i.replace('R', 'R-').split('-') for i in peptides_k], [])
    
    # if last AA is a lysine or arginine, the above lines will leave an empty string
    if peptides_r[-1] == '':
        peptides_r.pop(-1)
    
    # identifying peptides that start with proline
    # need to connect these with the preceding peptide
    KP_RP_inst = []
    for i in range(len(peptides_r)):
        if peptides_r[i][0] == 'P':
            
            peptides_r[i-1] = peptides_r[i-1] + peptides_r[i]
            KP_RP_inst.append(i)
    
    # removing the peptides that start with proline
    # starting with the last of such peptides and working backwards
    if len(KP_RP_inst) > 0:
        for i in range(len(KP_RP_inst)):
            pop_idx = KP_RP_inst[-(i+1)]
            peptides_r.pop(pop_idx)
    
    short_peptide_num = sum([int(len(i) <= 3) for i in peptides_r])
    return short_peptide_num
    
    
# function to calculate protein/peptide molecular weight
def get_prot_mw(sequence, kda=False):
     
    aa_dict = {}
    aa_dict['G'] = 57.05; aa_dict['A'] = 71.09; aa_dict['S'] = 87.08; aa_dict['T'] = 101.11;
    aa_dict['C'] = 103.15; aa_dict['V'] = 99.14; aa_dict['L'] = 113.16; aa_dict['I'] = 113.16;
    aa_dict['M'] = 131.19; aa_dict['P'] = 97.12; aa_dict['F'] = 147.18; aa_dict['Y'] = 163.18;
    aa_dict['W'] = 186.21; aa_dict['D'] = 115.09; aa_dict['E'] = 129.12; aa_dict['N'] = 114.11;
    aa_dict['Q'] = 128.14; aa_dict['H'] = 137.14; aa_dict['K'] = 128.17; aa_dict['R'] = 156.19
    
    aa_df = pd.DataFrame(pd.Series(aa_dict), columns=['MW'])
    
    aa_df['AA'] = aa_df.index
    aa_df.index = np.arange(len(aa_df))
 
    if sequence == None:
        return None
    
    # merge amino acid weights
    seq_df = pd.DataFrame(list(sequence), columns=['AA'])
    seq_df = seq_df.merge(aa_df, on='AA', how='left', copy=False)
    
    # get protein molecular weight
    if not kda:
        prot_mw = round(seq_df.MW.sum(), 2)
    else:
        prot_mw = round(seq_df.MW.sum()/1000.0, 2)
    
    return prot_mw
    
    
##########################################################################################


# some peptides will be very short
# from phospho data, I see that the shortest peptides have 10 AAs
# 'phospho' designation only outputs peptides with S, T, or Y in them
# NOTE: this function is case-sensitive (all should be upper-case)
def digest_prot(seq, min_length=10, lysC=True, phospho=False):
    
    # NOTE: trypsin does not cleave lys or arg if followed by proline
    
    # cutting after lysine
    peptides_k = seq.replace('K', 'K-').split('-')
    
    # cutting after arginine
    peptides_r = sum([i.replace('R', 'R-').split('-') for i in peptides_k], [])
    
    # if last AA is a lysine or arginine, the above lines will leave an empty string
    if peptides_r[-1] == '':
        peptides_r.pop(-1)
    
    # lysC will cut after proline
    if lysC == False:
        # identifying peptides that start with proline
        # need to connect these with the preceding peptide
        KP_RP_inst = []
        for i in range(len(peptides_r)):
    
            if peptides_r[i][0] == 'P':
                peptides_r[i-1] = peptides_r[i-1] + peptides_r[i]
                KP_RP_inst.append(i)
    
        # removing the peptides that start with proline
        # starting with the last of such peptides and working backwards
        if len(KP_RP_inst) > 0:
            for i in range(len(KP_RP_inst)):
                pop_idx = KP_RP_inst[-(i+1)]
                peptides_r.pop(pop_idx)
      
    # identifying and filtering out short peptides
    if min_length > 0:
            
        short_pep_idx = []
        for i in range(len(peptides_r)):
            if len(peptides_r[i]) < min_length:
                short_pep_idx.append(i)
                
        # starting with the last of such peptides and working backwards
        if len(short_pep_idx) > 0:
            for i in range(len(short_pep_idx)):
                pop_idx = short_pep_idx[-(i+1)]
                peptides_r.pop(pop_idx)
                
    # identifying and filtering out peptides with no S, T, or Y
    if phospho == True:
        
        no_STY_idx = []
        for i in range(len(peptides_r)):
            pep = peptides_r[i]
            if ('S' not in pep) & ('T' not in pep) & ('Y' not in pep):
                no_STY_idx.append(i)
          
        # starting with the last of such peptides and working backwards
        if len(no_STY_idx) > 0:
            for i in range(len(no_STY_idx)):
                pop_idx = no_STY_idx[-(i+1)]
                peptides_r.pop(pop_idx)
    
    return peptides_r


##########################################################################################


# list of proteins in MS_data
def find_ms_prots(MS_df, letters):
    
    letters = letters.upper()
    
    prot_list = sorted(list(MS_df.Gene_Symbol.unique()))
    subset = [i for i in prot_list if i[:len(letters)].upper() == letters]

    return subset
    
    
##########################################################################################


# making simple function to get antilog of column values
def delog_df(df, quant_cols):
    
    df_copy = df.copy()
    
    for col in quant_cols:
        df_copy[col] = df_copy[col].apply(lambda x: 2.**(x))
        
    return df_copy


##########################################################################################


# making function to obtain partion coefficients 
# 'col_pairs': pairs of column indices, taken in as nested lists, e.g. [[num, denom],...etc]
# where 'num' and 'denom' are the appropriate indices of the columns in 'quant_cols'
# NOTE: the number of quant_cols will be halved
def get_part_coeff_df(ms_df, quant_cols, col_pairs, new_col_names=[], include_inf=False):
    
    ms_df_copy = ms_df.copy()
        
    # iterate through column pairs
    for i,pair in enumerate(col_pairs):
        num_col = quant_cols[pair[0]]    # numerator
        other_col = quant_cols[pair[1]]  # other column
        
        if len(new_col_names) == 0:
            ms_df_copy[num_col+'_pcoeff'] = ms_df_copy[num_col] / ms_df_copy[[num_col, 
                                            other_col]].apply(lambda x: np.sum(x), axis=1)
        
            # recording new column name
            new_col_names.append(num_col+'_pcoeff')
        else:
            ms_df_copy[new_col_names[i]] = ms_df_copy[num_col] / ms_df_copy[[num_col, 
                                            other_col]].apply(lambda x: np.sum(x), axis=1)
            
            
    # getting rid of any rows with NaNs
    # first, finding the number of NaN in each row
    # then, identifying rows that have no NaNs
    row_num_nums = ms_df_copy[new_col_names].applymap(np.isnan).apply(np.sum, 
                                                             axis=1).apply(lambda x: x == 0)
      
    # getting rid of rows where there are any NaNs
    # this is because the median in these cases would be equal to zero --> divide by zero
    ms_df_copy = ms_df_copy.ix[row_num_nums, :]
    

    # some rows will have -inf in them 
    # if those rows are not desired
    if not include_inf:
        row_num_fin = ms_df_copy[new_col_names].applymap(np.isinf).apply(np.sum, 
                                                            axis=1).apply(lambda x: x == 0)
        ms_df_copy = ms_df_copy.ix[row_num_fin, :]
    
    
    # dropping old quant cols
    for col in quant_cols:
        ms_df_copy = ms_df_copy.drop(col, axis=1)
    
    # resetting index
    ms_df_copy = ms_df_copy.reset_index(drop=True)
        
    return ms_df_copy


##########################################################################################


# function to find residual distances of peptide trends
# these trends are used for quantifying protein amounts
def sum_residual(array, total_sum=False):
    
    """
    'array' is a numpy array of peptide rows and independent variable 
    (e.g. time) columns
    
    function will calculate residual distances from the mean for each column
    and output the total residual sum
    """
    
    rows, cols = np.shape(array)
    
    # get column means, then tile array
    mean_array = np.mean(array, axis=0)
    tile_mean = np.tile(mean_array, (rows, 1))
    
    # calculate residuals
    resid_array = np.abs(array - tile_mean)
    
    # return peptide residuals
    if not total_sum:
        return np.sum(resid_array, axis=1)
    
    # return total residual sum
    else:
        return np.sum(np.sum(resid_array))
        
        
##########################################################################################


# making function to find a consensus sequence between two proteins
# currently, it does not take into account gaps or deletions
# 'num_AAs' is desired number of AAs in consensus sequence
# 'percent_id' is the identity desired between the proteins in the consensus sequence
def find_cons_seq(seq1, seq2, num_AAs=10, percent_id=0.8):

    # dataframe is useful for indexing
    seq2_df = pd.DataFrame(list(seq2), columns=['AA'])
    
    # things to preset
    cseq_list = []
    idx_i = 0
    cons_seq = False
    
    # while-loop will be more efficient than for-loop
    # make sure we're not looking less than 10 AAs away from C-terminus
    while idx_i < len(seq1)-num_AAs:
        
        if cons_seq == True:
            idx_i = idx_i+num_AAs
        
        i = seq1[idx_i]     # getting the amino acid
    
        # this is a list of indices of where the seq1 AA occurs in seq2
        pos_list = seq2_df.index[seq2_df.AA == i]
        
        # make sure we're not looking less than 10 AAs away from C-terminus
        pos_list = pos_list[pos_list < len(seq2)-num_AAs]
    
        # continue if the AA does not appear in seq2
        if len(pos_list) < 1:
            idx_i += 1
            continue
    
        miss_list = []
        miss, idx_j = 0, 0
        while miss <= (1-percent_id)*num_AAs:
            j = seq2[pos_list[0]:][idx_j]     # getting the amino acid
            
            # noting if the seq1 AA and seq2 AA do not match
            if j != seq1[idx_i + idx_j]:
                
                miss_list.append(idx_j)
                miss += 1
    
            # if we've gotten enough AAs for the consensus sequence, record, then break loop
            if idx_j == num_AAs-1:
                cseq = seq2[pos_list[0]:pos_list[0] + idx_j+1]     # the initial consensus
                
                # if there were misses in the consensus sequence, turning them into 'X'
                cseq = list(cseq)
                for k in miss_list:
                    cseq[k] = 'X'
                    
                cseq = "".join(cseq)
                
                # appending to the list of consensus sequences
                cseq_list.append(cseq)
    
                cons_seq = True
                break
            
            idx_j += 1
        idx_i += 1
        
    if len(cseq_list) < 1:
        return None
    else:
        return cseq_list
    
    
##########################################################################################


# function that counts how many sets of 10 are present in an array
def count_10plex_reps(array):
    
    # total number of NaNs in array
    nan_num = np.isnan(array).sum()
    
    # total number of non-NaNs in array
    non_nan_num = len(array) - nan_num
    
    # total number of non-NaN 10-plexes in array
    tot_reps = int(non_nan_num / 10.0)
    
    return tot_reps
    

##########################################################################################


# function that infers missing values in a large dataset
# NOTE: all data columns within a given 10-plex experiment should be grouped
# e.g. [exp1 col1, exp1 col2, exp1 col3,...exp2 col1, exp2 col2, exp3 col2, etc]
def infer_NaNs(data_df, quant_cols, nplex=10, min_plex_occur=2, log2FC_cutoff=1.0, 
               n_clusters=16):
    
    
    
    ## data truncation section
    
    # make copy to avoid bugs
    data_copy = data_df.copy()
    
    # only retain data that meets log2FC_cutoff
    data_copy = data_copy.loc[data_copy[quant_cols].apply(lambda x: 
                            np.nanmax(np.abs(x)) >= log2FC_cutoff, axis=1), :]
    
    # only retain peptide rows where there are at least nplex * min_plex_occur non-NaNs
    max_NaN_cutoff = len(quant_cols) - nplex * min_plex_occur
    
    
    data_copy = data_copy.loc[data_copy[quant_cols].apply(lambda x: 
                            len(x[np.isnan(x)]) <= max_NaN_cutoff, axis=1), :]
    
    
    
    ## fuzzy c-means clustering section
    
    # determine number of multiplexed groups present
    num_groups = int(len(quant_cols) * 1.0 / nplex)
    
    # make a list of 3-wise (or min_plex_wise + 1) combinations of these groups
    combos_list = [list(c) for c in list(combinations(np.arange(num_groups)*nplex, min_plex_occur+1))]
    exp_combos_list = [list(c) for c in list(combinations(np.arange(num_groups), min_plex_occur+1))]
    
    # also make a fuller version of this list of combinations
    all_full_combos = []  # 3D nested list
    for combo in combos_list:
        
        full_combos = []
        # for each starting index, produce all indices of that group
        for start_idx in combo:
            plex_idx_array = (np.arange(nplex) + start_idx).tolist()
            full_combos.append(plex_idx_array)
        all_full_combos.append(full_combos)   
        
    # also define a list of all of the experimental groups
    all_experiment_idc = [(np.arange(nplex)+nplex*e).tolist() for e in np.arange(num_groups)]
    
    # iteratively obtain cluster means for each combination of columns
    fcmc_mean_arrays = []
    for idx_combo in all_full_combos:
        
        # denest each 3-wise (or min_plex_occur + 1) combo
        denest_idx_combo = sum(idx_combo, [])
        
        # take respective subset of columns and data
        col_subset = np.array(quant_cols)[denest_idx_combo].tolist()
        data_subset = data_copy.loc[data_copy[col_subset].apply(lambda x: 
                                        np.all(~np.isnan(x)), axis=1)]
        data_subset_array = data_subset[col_subset].values
        
        # perform fuzzy c-means clustering
        u_array, clust_assign, clust_score = fuzzy_cmc(data_subset_array, 
                 clust_num=n_clusters, exp_m=1.01, remove_empty=True, print_jm=False)
        
        # getting average cluster trend
        weight_cluster_mean_array = nan_dot_product(u_array, data_subset_array, normalize=True)
        fcmc_mean_arrays.append(weight_cluster_mean_array)
        
        
        
    ## peptide value inference section
    
    # initiating array for which values will be inferred
    infer_data_array = data_copy[quant_cols].values
    
    # iterate through each peptide
    for pep_row in np.arange(np.shape(infer_data_array)[0]):
        
        # getting array of original peptide values (with NaNs and all)
        pep = infer_data_array[pep_row, :]
        
        # record where NaN values are in the peptide
        pep_NaN_idc = np.arange(len(pep))[np.isnan(pep)]
        pep_nonNaN_idc = np.arange(len(pep))[~np.isnan(pep)]
        
        # determine empty experiments for this peptide
        empty_group_list = []
        nonempty_group_list = []
        for idx in np.arange(num_groups):  # iterating through the n-plexes
            
            # get corresponding sample indices of experiment
            group = all_experiment_idc[idx]
            
            # determine if indices from a given group all have NaNs in this peptide 
            count = 0
            for jdx in group:  # iterating through the channels of the n-plex
                if jdx in pep_NaN_idc:
                    count += 1
                    
            # in this case, this group is an empty group
            if count == len(group):
                empty_group_list.append(idx)
            else:
                nonempty_group_list.append(idx)
            
        # iterate through each experimental set of values that needs to be inferred
        for group in empty_group_list:
            
            # now make a list of the "min_plex_occur" combinations of non-NaN experiments
            nonempty_combos = list(combinations(nonempty_group_list, min_plex_occur))
            nonempty_combos = [list(c) for c in nonempty_combos]

            # initiate an array with "nonempty_combos" rows and "nplex" columns (e.g. 2 by 10)
            pep_infer_array = np.zeros((len(nonempty_combos), nplex))
            for idx, ncombo in enumerate(nonempty_combos):

                # take respective values of the peptide
                all_pep_combo_idc = sum(np.array(all_experiment_idc)[ncombo].tolist(), [])
                pep_nonempty_vals = pep[all_pep_combo_idc]

                # take respective cluster mean array;
                # i.e. must contain the same nonempty experiments plus the empty experiment
                clust_exp_combo = ncombo + [group] 
                
                # we need to sort these to be consistent with the cluster list
                clust_exp_combo_sort = sorted(clust_exp_combo)
                
                # but we need to know the new index positions of the empty group in question
                clust_exp_combo_argsort = np.argsort(clust_exp_combo).tolist()
                new_idx_pos_of_group = clust_exp_combo_argsort.index(len(clust_exp_combo)-1)
                
                # the cluster indices of every element of the group
                clust_idc_of_group = np.arange(nplex) + new_idx_pos_of_group*nplex
                
                # also get the new index positions of the nonempty groups
                nonempty_idx_pos = [pos for pos in np.arange(len(
                                    clust_exp_combo)) if pos != new_idx_pos_of_group]
                nonempty_clust_idc = sum([(np.arange(nplex) + nidx*nplex).tolist(
                                      ) for nidx in nonempty_idx_pos], [])
                
                # find the cluster that has these experiments included
                cluster_idx = exp_combos_list.index(clust_exp_combo_sort)
                cluster_mean_array = fcmc_mean_arrays[cluster_idx]
                
                # make truncated form of cluster mean array with only the nonempty experiments
                # i.e. remove the columns that correspond to the peptide's empty experiment
                trunc_cluster_mean_array = cluster_mean_array[:, nonempty_clust_idc]
                clust_rows = np.shape(trunc_cluster_mean_array)[0]
                 
                # tile truncated peptide array accordingly
                pep_nonempty_vals_tile = np.tile(pep_nonempty_vals, (clust_rows, 1))
                
                # determine euclidean distance between the peptide and each cluster mean array
                # then take reciprocal
                recip_dist_vec = 1.0 / np.sqrt(np.sum(
                        (trunc_cluster_mean_array - pep_nonempty_vals_tile)**2.0, 1))
                
                # now take part of cluster corresponding to the empty values in peptide
                empty_pos_cluster_mean_array = cluster_mean_array[:, clust_idc_of_group]
                
                # use reciprocal distance vector as weights
                # need to take dot product with empty positions of cluster mean array
                inferred_pep_vals = np.dot(recip_dist_vec, empty_pos_cluster_mean_array)
                
                # record peptide inference array
                pep_infer_array[idx, :] = inferred_pep_vals
                
                
            # now take average of all peptide inference arrays
            avg_pep_infer = np.mean(pep_infer_array, axis=0)
                
            # record the inferred values of this experimental set 
            # get the original set of nplex indices that correspond to this experimental group
            orig_exp_idc = all_experiment_idc[group]
            infer_data_array[pep_row, orig_exp_idc] = avg_pep_infer

            
    # replace data array with inferred data array
    data_copy[quant_cols] = infer_data_array
    
    return data_copy


##########################################################################################


# making core function to compute average CV from multiple groups of replicate values
# given one peptide array
def representative_cv_core(peptide_array, rep_idx_groups, method='max', log2_vals=True):
    
    # initiate array where we will record CVs for the given replicates
    rep_group_cv_array = np.zeros(len(rep_idx_groups))
    for i, rep_group in enumerate(rep_idx_groups):
        
        # get the raw values of the array
        rep_group_vals = peptide_array[rep_group]
        
        # if the values are log2-transformed, de-log them to avoid any divisions by zero
        if log2_vals:
            # de-log the values to avoid division by zero
            rep_group_vals = 2.0**(rep_group_vals)
            
        # record current replicate CV
        rep_group_cv_array[i] = np.nanstd(rep_group_vals) / np.nanmean(rep_group_vals)
        
    # compute max or avg CV of all replicate groups
    if method == 'max':
    	repr_pep_cv = np.nanmax(rep_group_cv_array)
    else:
    	repr_pep_cv = np.nanmean(rep_group_cv_array)
    
    # if no replicates found, set the average CV arbitrarily large
    if np.isnan(repr_pep_cv):
        repr_pep_cv = 10.0**6.0 
    
    return repr_pep_cv


# making function to compute confidence score from euclidean dist calculations
# this function assumes duplicates are present, but no triplicates
# also assuming that the corresponding duplicates are in the same order
def get_cv_conf(peptide_array, rep_idx_groups, testing_num=1000, log2_vals=True):
    
    """ 
    array: numpy array that we are testing replicate matching for
    
    rep_idx_groups: list of groups of indices; each group is a nested list that corresponds to 
                    a condition that is replicated (number of indices corresponds to
                    number of replicates for that group)
                    e.g. [[0, 2, 4], [1, 5], [3, 6]]
    
    """
    
    
    # verify peptide array is a numpy array
    peptide_array = np.array(peptide_array)
    
    # getting the max CV of all of the replicate values in the peptide array
    orig_repr_pep_cv = representative_cv_core(peptide_array, rep_idx_groups, log2_vals)
    

    # shuffling each row of tiled_array then recalculating random CV
    tiled_array = np.tile(peptide_array, (testing_num, 1))
    nh_wins = 0
    for row in range(testing_num):
        tiled_array[row, :] = shuffle_array(tiled_array[row, :])
        
    # make dataframe with columns for to accomodate the shuffled arrays
    shuffle_cols = ['Shuffle_Idx'+str(i) for i in range(len(peptide_array))]
    shuffle_df = pd.DataFrame(tiled_array, columns=shuffle_cols)
    shuffle_df['Shuffle_CVs'] = shuffle_df[shuffle_cols].apply(lambda x: 
                                representative_cv_core(x, rep_idx_groups, log2_vals), axis=1)
        
    # null hypothesis wins if the shuffled CV is smaller than the original CV
    nh_wins = shuffle_df.Shuffle_CVs.apply(lambda x: x <= orig_repr_pep_cv).sum()
       
    # calculate p-value and confidence value
    pval = nh_wins*1.0  / testing_num
    conf = 1.0 - pval
    
    return conf  


##########################################################################################

##########################################################################################

##########################################################################################

