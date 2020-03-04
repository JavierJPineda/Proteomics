## This file contains functions useful for phosphoproteomic analysis

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

# suppress annoying "deprecation warnings"
import warnings
warnings.filterwarnings("ignore")

# import all of my useful functions
from JP_ms_functions import *
from JP_plotting_functions import *
from JP_stat_functions import *
from JP_uniprot_blast_functions import *

##########################################################################################


# function to get one phosphosite per peptide sequence
# used for correct motif assignment (via motifx)
# if include gene, need 'Site_ID_Prot_ID' column
def get_unique_sites(phospho_df, include_gene=True):
    
    ms_df = phospho_df.copy()
    
    ms_df['parent_pep'] = ms_df.Sequence.apply(lambda x: 
            x.replace('S#', 'S').replace('T#', 'T').replace('Y#', 'Y'))
                                        
    # remove res that follows the second period
    ms_df['parent_pep'] = ms_df.parent_pep.apply(lambda x: 
                x[:x[2:].index('.')+2].replace('.', '').replace('-', 
                '') if '.' in x[2:] else x.replace('.', 
                '').replace('-',''))
    
    unq_peptides = ms_df.parent_pep.unique().tolist()

    all_pep_all_sites = pd.DataFrame()
    for pep in unq_peptides:
    
        # grab all peptide states associated with parent peptide
        ass_peptides = ms_df.ix[ms_df.Sequence.apply(lambda x: 
                        x.replace('.', '').replace('-', '').replace('S#', 
                       'S').replace('T#', 'T').replace('Y#', 
                       'Y') == pep), :].reset_index(drop=True)
        
        ass_peptides['Site_Position'] = ass_peptides.Site_Position.apply(lambda x: 
                     x.split(';'))
        
        # this gets the protein id (e.g. 'MAP4_HUMAN')
        if include_gene:
            prot_id = ass_peptides.Site_ID_Prot_ID.tolist()[0]
            first_pipe = prot_id.index('|')
            second_pipe = prot_id[first_pipe+1:].index('|')+first_pipe+1
            prot_id = prot_id[second_pipe+1:]
                
        site_idc = list(set(sum(ass_peptides.Site_Position.tolist(), [])))
        seq_start = ass_peptides.Seq_Start.tolist()[0]
        
        if seq_start == 0:
            continue
        
        site_idc = [int(i)-seq_start for i in site_idc]
        
        if len(site_idc) > 1:
            site_idc = sorted(site_idc)
        
        # start background to avoid screwing up indexing
        clarified_peps = []
        for site in site_idc:
            clarified_peps.append(pep[:site+1]+'#'+pep[site+1:])
            
        pep_all_sites = pd.DataFrame(clarified_peps, 
                                     columns=['Site_Sequence'])
        pep_all_sites['Parent_Peptide'] = pep
        pep_all_sites['Site'] = site_idc + seq_start
        
        if include_gene:
            pep_all_sites['Protein_ID'] = prot_id
        
        # concatenate peptide dataframes
        all_pep_all_sites = pd.concat([all_pep_all_sites, 
                                       pep_all_sites])
    all_pep_all_sites = all_pep_all_sites.sort(['Parent_Peptide', 
                                        'Site']).reset_index(drop=True)
    all_pep_all_sites['Site_Sequence'] = all_pep_all_sites.Site_Sequence.apply(lambda x: 
        x.replace('S#', 's').replace('T#', 't').replace('Y#', 'y'))

    return all_pep_all_sites


##########################################################################################


## these functions allow for putative assignment of kinases to phosphopeptides

# one of the functions will require a kinase-motif reference dataframe
# use the following csv file:
# k_substrate_df = pd.read_csv("Kinase_Motif_Ref.csv")

# function to convert motif sequence to list format 
# apart from the list format, the sequences will have the
# same format as those in the phospho dataset, EXCEPT:
# phosphorylated S, T, Y ==> lowercase s, t, y
def motif_to_list(motif):
    
    # convert sequence form in kinase-motif ref; pS --> s, pT --> t, pY --> y 
    motif = motif.replace('pS', 's').replace('pT', 't').replace('pY', 'y')
    
    # removing any asterisks in motif
    motif = motif.replace('*', '')
    
    # NOTE: I DON'T KNOW WHAT THE PURPOSE OF THE ASTERISK WAS
    
    # split on open bracket
    motif_list = motif.split('[')
    
    # split on close bracket; then flatten nested lists that result
    motif_list = sum([i.split(']') for i in motif_list], [])
    
    # get rid of empty strings
    motif_list = [i for i in motif_list if i != '']
    
    # split between all characters, unless '/' is between them
    # then take out '/'
    motif_list = [[i.replace('/', '')] if '/' in i else list(i) for i in motif_list]

    # flatten nested list
    motif_list = sum(motif_list, [])

    return motif_list


# better than the normal find() function
# def find_idx(string, char):
#     return [idx for idx, letter in enumerate(string) if letter == char]
def find_idx(sequence, pattern):
    
    idc=[]; last_trunc=0
    while pattern in sequence:

        idx = sequence.index(pattern)
        idc.append(idx + last_trunc)
        
        if isinstance(sequence, str):
            sequence = sequence[idx + len(pattern):]
            last_trunc = last_trunc + idx + len(pattern)
        else: # then assume a list
            sequence = sequence[idx + 1:]
            last_trunc = last_trunc + idx + 1
    
    return idc


# clarifies phosphopeptide 'Sequence' based on 'Site_Position'
# if a site is not quantified (i.e. not in 'Site_Position'), but
# appears as a phosphosite in the sequence, it will be re-written
# without the phospho designation (i.e. '#')
# all inputs are strings
# NOTE: returned pseq is the same but is stripped of periods and dashes
# 'just_pos' is if the user only wants the starting position of peptide
#  'seq_id': can either be sequence ID or PROTEIN sequence
def clarify_pseq(seq_id, pseq, site_pos, just_pos=False):
    
    if not isinstance(site_pos, str):
        return np.nan
    
    # if there are periods or dashes, remove them
    # remove residue that follows second period (not really included)
    if '.' in pseq[2:]:
        pseq = pseq[:pseq[2:].index('.')+2]
    pseq = pseq.replace('.','').replace('-','')
    
    # make sure site_pos is a string
    site_pos = str(site_pos)
    
    # converting site_pos numbers to integer for comparison later
    site_pos = sorted([int(i) for i in site_pos.split(';')])
    
    # if same number of designators as number of quantified sites 
    if pseq.count('#') == site_pos.count(';')+1:
        if just_pos == False:
            return pseq
    
    # if NaN was inputted for the seq_id/protein sequence 
    # just return the original peptide sequence
    if not isinstance(seq_id, str):
        return pseq
    
    # all sequence IDs have numbers
    elif len(filter(str.isdigit, seq_id)) > 0:
        full_seq = get_prot_seq(seq_id)    # getting full protein sequence
    
    # no protein sequences should have numbers
    else:
        full_seq = seq_id
    
    # if we can't get the full sequence, 
    # deal with it on a one-by-one basis later
    if full_seq == None:
        if just_pos == False:
            return pseq
        else:
            return None
    
    # getting index position for start of phosphopeptide
    # must get rid of designators for alignment
    # will make phosphosites lower case
    
    pseq_raw=False
    if '#' not in pseq:
        pseq_raw = True
    
    pseq = pseq.replace('S#','s').replace('T#','t').replace('Y#',
                        'y')
    pseq_start = full_seq.find(pseq.upper())+1 # starting index position
    
    if just_pos == True:
        return pseq_start
    
    # if dealing with raw peptide
    if pseq_raw == True:
        for site in site_pos[::-1]:
            site_idx = site - pseq_start
            
            pseq = pseq[:site_idx+1]+'#'+pseq[site_idx+1:] 
            
        pseq = pseq.replace('S#','s').replace('T#','t').replace('Y#',
                        'y')

    # getting positions of designator
    pseq_hash = pseq.replace('s','#').replace('t','#').replace('y','#')
    
    pseq_hash_idx = np.array(find_idx(pseq_hash, '#')) + pseq_start
    pseq_hash_idx = [int(i) for i in pseq_hash_idx] # just in case
    
    sites_df = pd.DataFrame(pseq_hash_idx, columns=['hash_idx'])
    
    # getting site positions that are not quantified
    noquant_sites = sites_df.hash_idx[~(sites_df.hash_idx.isin(site_pos))].tolist()
    
    # re-indexing for modification to follow
    noquant_sites = [site-pseq_start for site in noquant_sites]
    
    for site in noquant_sites:
        pseq = pseq[:site] + pseq[site].upper() + pseq[site+1:]
            
        
    # putting in hashes for quantified sites
    pseq = pseq.replace('s','S#').replace('t','T#').replace('y','Y#')
    
    return pseq

    
# function to get positions of desired amino acids in a peptide sequence
# 'AA' is a string with the desired AAs (case sensitive)
def get_AA_sites(pep_seq, prot_seq, AA='STY'):

    if not isinstance(prot_seq, str):
        return str(np.nan)
            
    # verifying sequences are string
    pep_seq = str(pep_seq)
    prot_seq = str(prot_seq)


    # strip peptide sequence of any non-alpha characters
    pep_seq = re.sub(r'\W+', '', pep_seq)
    
    pep_start = prot_seq.upper().find(pep_seq.upper())+1  # accounting for zero indexing
    
    if pep_start == 0:  # peptide not found for whatever reason
        return str(np.nan)
    
    # STY indices
    AA_idc = []
    for aa in list(AA):
        AA_idc += find_idx(pep_seq, aa)
    
    # sorting then adding the start position
    AA_pos = list(np.array(sorted(AA_idc))+pep_start)
    
    # adding separator between sites
    AA_sep = [str(s)+';' for s in AA_pos]

    # returning as string
    return ''.join(AA_sep)[:-1]

    
# function to compare amino acids from a peptide sequence to 
# those in a motif
def aa_compare(seq_aa, motif_aa):
    
    if motif_aa.upper() == 'X':
        return True
    
    elif seq_aa in motif_aa:
        return True
    
    else:
        return False

    
# function to search for phospho motif in a phosphopeptide sequence
def find_phos_motif(seq, motif_df1, kinase_info=False, shuffle=False):
    
    motif_df = motif_df1.copy()
    motif_df['Motif_List'] = motif_df.Motif_List.apply(lambda x: ast.literal_eval(x))
        
    # clean sequence; remove periods, dashes, @ symbols, and asterisks
    seq = seq.replace('.', '').replace('-', '').replace('@', '').replace('*', '')
    
    # rewrite S#, T#, Y# as lowercase s, t, y
    seq = seq.replace('S#', 's').replace('T#', 't').replace('Y#', 'y')
    
    if shuffle:
        seq = ''.join(shuffle_array(np.array(list(seq))).tolist())
        
    # determining index positions of the phosphosites in the sequence
    p_seq_forward_idx = sorted(find_idx(list(seq), 's')+find_idx(list(seq), 
                                                       't')+find_idx(list(seq), 'y'))
    
    # initiations
    motifs_found = []      # motifs found in phosphopeptide sequence
    motif_len_list = []    # motif lengths for potential kinases
    motif_xnum_list = []   # num of 'x's in motif for potential kinases
    
    if kinase_info == True:
        potential_kinases = [] # kinases associated with motif

    # iterating through possible motifs
    for line in range(len(motif_df)):
        
        motif_list = motif_df.Motif_List[motif_df.index == line].tolist()[0]
        
        ##
        # determine number of AAs that precede/follow the motif phosphosite(s)
        p_motif_forward_idx = sorted(find_idx(motif_list, 's')+find_idx(motif_list, 
                                                   't')+find_idx(motif_list, 'y'))
        AAs_before_site = p_motif_forward_idx[0]
        
        
        ## build dataframe with seq and motif in side-by-side columns
        seq_len = len(seq)            # length of sequence
        motif_len = len(motif_list)   # length of motif
        
        # motif should be shorter than peptide sequence
        if motif_len > seq_len:
            continue
        
        # motif can be same size as peptide sequence
        elif seq_len == motif_len:
            
            # intialize comparison df
            compare_df = pd.DataFrame(list(seq), columns=['seq_aa'])
            compare_df['motif_aa'] = motif_list
            compare_df['match'] = compare_df.apply(lambda x: 
                                aa_compare(x['seq_aa'], x['motif_aa']), axis=1)
            match_sum = compare_df.match.sum()
            
            if match_sum != motif_len:
                continue
                
            else:
                # grab associated motif(s)/protein(s)
                motif = motif_df.Motif[motif_df.index == 
                                       line].tolist()[0]
                motifs_found.append(motif)
                motif_len_list.append(motif_len)
                motif_xnum_list.append(motif_list.count('X'))  
                
                if kinase_info == True:
                    kinase = motif_df.Acting_Kinase[motif_df.index == 
                                                line].tolist()[0]
                    potential_kinases.append(kinase)
            
        else:
            
            # iterating through phosphosites in the sequence
            for p_idx in p_seq_forward_idx:
                
                # truncate the sequence to match length of motif
                trunc_seq = seq[p_idx-AAs_before_site:p_idx-AAs_before_site+len(motif_list)]
                
                # this means we've reached the end of the sequence
                # but truncated sequence length should still be greater than length of motif
                # if not, move on to next motif
                if len(trunc_seq) < motif_len:
                    break
                
                # intialize comparison df
                compare_df = pd.DataFrame(list(trunc_seq), columns=['seq_aa'])
                compare_df['motif_aa'] = motif_list
    
                # determining if sequence matches motif
                compare_df['match'] = compare_df.apply(lambda x: 
                                    aa_compare(x['seq_aa'], x['motif_aa']), axis=1)
            
                # this should equal the length of the motif if perfect match
                match_sum = compare_df.match.sum()
                
                if match_sum == motif_len:
                    # grab protein(s) associated with the matching motif
                    motif = motif_df.Motif[motif_df.index == 
                                       line].tolist()[0]
                    motifs_found.append(motif)
                    motif_len_list.append(motif_len)
                    motif_xnum_list.append(motif_list.count('X'))
                    
                    if kinase_info == True:
                        kinase = motif_df.Acting_Kinase[motif_df.index == 
                                                    line].tolist()[0]
                        potential_kinases.append(kinase)
                            
                    break    # don't need to search for this motif anymore
        
    if len(motifs_found) != 0:
        # sort by descending ratios of motif length-to-x number    
        sort_df = pd.DataFrame(motifs_found, columns=['motifs'])
        sort_df['motif_len'] = motif_len_list
        sort_df['motif_xnum'] = motif_xnum_list
        
        if kinase_info == True:
            sort_df['potential_kinases'] = potential_kinases
            
        sort_df = sort_df.sort(['motif_len', 'motif_xnum'],
                               ascending=[0, 1]).reset_index(drop=True)
        if kinase_info == True:
            potential_kinases = sum(sort_df.potential_kinases.tolist(), 
                                    [])
        motifs_found = sort_df.motifs.tolist()
    
    # only return unique motifs/kinase lists
    if kinase_info == True:
        return list(set(motifs_found)), list(set(potential_kinases))
    else:
        return list(set(motifs_found))
       
       
##########################################################################################
 
        
# function to search for a motif in a protein sequence
# site_AAs: string with desired AAs to be counted as sites (e.g. site_AAs='RK')
def find_motif(seq, motifs_df, site_AAs, shuffle=False):
    
    motif_df = motifs_df.copy()
    
    if isinstance(motif_df.Motif_List.tolist()[0], str):
        motif_df['Motif_List'] = motif_df.Motif_List.apply(lambda x: ast.literal_eval(x))
        
    # clean sequence; remove periods, dashes, @ symbols, and asterisks
    seq = seq.replace('.', '').replace('-', '').replace('@', '').replace('*', '')
    
    if shuffle:
        seq = ''.join(shuffle_array(np.array(list(seq))).tolist())
        
    # determining index positions of the Arg/Lys in the sequence
    site_seq_forward_idx = sum([find_idx(list(seq), AA) for AA in site_AAs], [])
    site_seq_forward_idx = sorted(site_seq_forward_idx)
    
    # initiations
    motifs_found = []      # motifs found in phosphopeptide sequence
    motif_len_list = []    # motif lengths for potential kinases
    motif_xnum_list = []   # num of 'x's in motif for potential kinases
    

    # iterating through possible motifs
    for line in range(len(motif_df)):
        
        motif_list = motif_df.Motif_List[motif_df.index == line].tolist()[0]
        
        # determine number of AAs that precede/follow the motif phosphosite(s)
        site_motif_forward_idx = sum([find_idx(motif_list, AA) for AA in site_AAs], [])
        site_motif_forward_idx = sorted(site_motif_forward_idx)
        
        AAs_before_site = site_motif_forward_idx[0]
        
        
        ## build dataframe with seq and motif in side-by-side columns
        seq_len = len(seq)            # length of sequence
        motif_len = len(motif_list)   # length of motif
        
        # motif should be shorter than peptide sequence
        if motif_len > seq_len:
            continue
        
        # motif can be same size as peptide sequence
        elif seq_len == motif_len:
            
            # intialize comparison df
            compare_df = pd.DataFrame(list(seq), columns=['seq_aa'])
            compare_df['motif_aa'] = motif_list
            compare_df['match'] = compare_df.apply(lambda x: 
                                aa_compare(x['seq_aa'], x['motif_aa']), axis=1)
            match_sum = compare_df.match.sum()
            
            if match_sum != motif_len:
                continue
                
            else:
                # grab associated motif(s)/protein(s)
                motif = motif_df.Motif[motif_df.index == 
                                       line].tolist()[0]
                motifs_found.append(motif)
                motif_len_list.append(motif_len)
                motif_xnum_list.append(motif_list.count('X'))  
            
        else:
            
            # iterating through phosphosites in the sequence
            for site_idx in site_seq_forward_idx:
                
                # truncate the sequence to match length of motif
                trunc_seq = seq[site_idx-AAs_before_site:site_idx-AAs_before_site+len(motif_list)]
                
                # this means we've reached the end of the sequence
                # but truncated sequence length should still be greater than length of motif
                # if not, move on to next motif
                if len(trunc_seq) < motif_len:
                    break
                
                # intialize comparison df
                compare_df = pd.DataFrame(list(trunc_seq), columns=['seq_aa'])
                compare_df['motif_aa'] = motif_list
    
                # determining if sequence matches motif
                compare_df['match'] = compare_df.apply(lambda x: 
                                    aa_compare(x['seq_aa'], x['motif_aa']), axis=1)
            
                # this should equal the length of the motif if perfect match
                match_sum = compare_df.match.sum()
                
                if match_sum == motif_len:
                    # grab protein(s) associated with the matching motif
                    motif = motif_df.Motif[motif_df.index == 
                                       line].tolist()[0]
                    motifs_found.append(motif)
                    motif_len_list.append(motif_len)
                    motif_xnum_list.append(motif_list.count('X'))
                            
                    break    # don't need to search for this motif anymore
        
    if len(motifs_found) != 0:
        # sort by descending ratios of motif length-to-x number    
        sort_df = pd.DataFrame(motifs_found, columns=['motifs'])
        sort_df['motif_len'] = motif_len_list
        sort_df['motif_xnum'] = motif_xnum_list
        sort_df = sort_df.sort(['motif_len', 'motif_xnum'], ascending=[0, 
                                                                1]).reset_index(drop=True)
        motifs_found = sort_df.motifs.tolist()
    
    # only return unique motif list
    return list(set(motifs_found))
    

##########################################################################################


# function to simplify outputted motifs from motif-x
# i.e. gets rid of redundant motifs
# 'motifs': list of motifs from motif-x output
def simplify_motifx(motifs):
        
    # reformatting into dataframe
    motif_df = pd.DataFrame(motifs, columns=['Motif'])
    motif_df['Motif_List'] = motif_df.Motif.apply(lambda x: 
                            motif_to_list(x))
    
    
    redund_motifs = [] # motifs that are present in other motifs
                       # NOTE: simpler motifs are preferred
                       # 'RXXs' is preferred over 'RXXsD'
            
    # comparing motifs (in list form)
    for motif_i in motif_df.Motif.tolist():
        
        motif_list_i = motif_df.Motif_List[motif_df.Motif == 
                                         motif_i].tolist()[0]
        
        motif_i_len = len(motif_list_i)
        
        matches = 0   # number of times motif is found in other motifs
        compare_df = pd.DataFrame(motif_list_i, columns=['motif_i_aa'])
        
        # iterating through possible motifs
        for motif_j in motif_df.Motif.tolist():
            motif_list_j = motif_df.Motif_List[motif_df.Motif == 
                                               motif_j].tolist()[0]
            
            ## build dataframe with the two motifs in side-by-side cols
    
            motif_j_len = len(motif_list_j)   # length of motif
            
            # motif_j should not be longer than motif_i
            if motif_j_len > motif_i_len:
                continue
            
            # motif_j can be same size as motif_i
            elif motif_j_len == motif_i_len:
                
                compare_df['motif_j_aa'] = motif_list_j
                compare_df['match'] = compare_df.apply(lambda x: 
                        aa_compare(x['motif_i_aa'], x['motif_j_aa']), 
                        axis=1)
                
                match_sum = compare_df.match.sum()
                
                if match_sum != motif_j_len:
                    continue
                    
                else:
                    matches += 1
                    
            else:
                # will tack on zeros onto end of motif to equal length of sequence
                num_0s = motif_i_len - motif_j_len
                
                # making the same length 
                motif_list_j = motif_list_j + ['0' for i in range(num_0s)]
                
                for shift in range(num_0s + 1):
                    compare_df['motif_j_aa'] = motif_list_j
        
                    # determining if sequence matches motif
                    compare_df['match'] = compare_df.apply(lambda x: 
                        aa_compare(x['motif_i_aa'], x['motif_j_aa']), 
                        axis=1)
                
                    # this should equal the length of the motif if perfect match
                    match_sum = compare_df.match.sum()
                    
                    if match_sum != motif_j_len:
                        
                        # shift rightward by one character
                        motif_list_j = list(motif_list_j[-1]) + motif_list_j[:-1]
                        continue # try another frame shift
                    
                    else:
                        matches += 1      
                        break    # don't need any more frame shifts
    
        # every motif will match itself
        if matches > 1:
            redund_motifs.append(motif_i)
 
    motif_df = motif_df[~(motif_df.Motif.isin(redund_motifs))]
    
    # return non-redundant motifs
    return motif_df.Motif.tolist()


##########################################################################################


# these functions are useful for determining phosphosite directionality
# NOTE 1: the deeper the proteomics, the better
# NOTE 2: it may be extremely helpful/crucial to also have phospho-depleted data

# reduces equation output of linear system solving to logic
def get_logic(eq):
    
    # turning expression into list
    eq = str(eq).replace(' ','')

    # finding denominators (unnecessary weights)
    divs = find_idx(eq, '/')
    
    # removing all divisions
    # i.e. remove '/' and every character that comes between it
    # and the next sign ('+' or '-') or the end of the equation
    for idx in divs[::-1]:
        
        # find the next sign if there are any
        next_plus = eq[idx:].find('+')
        next_minus = eq[idx:].find('-')
        
        if next_plus == -1:
            next_plus = 100
        if next_minus == -1:
            next_minus = 100
            
        next_sign = min([next_plus, next_minus])
        
        if next_sign == 100:
            eq = eq[:idx]
        else:
            eq = eq[:idx] + eq[next_sign+idx:]

    # next step is easier with the string reversed
    eq_r = eq[::-1] 
    
    # finding multiplications (unnecessary weights)
    mults = find_idx(eq_r, '*')
        
    # removing all multiplications
    # i.e. remove '*' and every character that comes before it
    # and after the next sign ('+', '-') or the start of the equation
    for idx in mults[::-1]:
        
        # find the next sign if there are any
        next_plus = eq_r[idx:].find('+')
        next_minus = eq_r[idx:].find('-')
        
        if next_plus == -1:
            next_plus = 100
        if next_minus == -1:
            next_minus = 100
            
        next_sign = min([next_plus, next_minus])
        
        if next_sign == 100:
            eq_r = eq_r[:idx]
        else:
            eq_r = eq_r[:idx] + eq_r[next_sign+idx:]
    
    eq = eq_r[::-1] # reverse back
        
    # make sure every number/variable is preceded by a sign
    # if no sign in front, put the implicit positive sign
    if eq[0] not in ['-', '+']:
        eq = '+' + eq
        
    # returning simplified logical expression as string
    return eq


# function to do actual deconvolution step
# 'logic_exp' = list of logical expressions
# 'corr_mat' = array of pairwise peptide correlations
# 'ass_pep_df' = dataframe of associated peptides
# 'corr_thresh' = cut-off for correlation (could use p-values instead)
def deconv_core(logic_exp, corr_mat, ass_pep_df, corr_thresh=0.2):
    
    num_sites = len(logic_exp) # number of sites to deconvolve
    
    # iterating through each logical expression
    # each logical expression corresponds to a single site
    site_direcs = list(np.zeros(shape=(num_sites, 1))) # site directions
    for idx, ex in enumerate(logic_exp):
        
        # getting correlation conditions as defined in logical expression
        # '+' => positively correlated, '-' => negatively correlated
        ex_space = ex.replace('+',' +').replace('-',' -')
        
        # getting peptide indices
        pep_idc = [int(i[1:]) for i in ex_space.split(' ')[1:]]
        
        # getting peptide signs
        pep_signs = [i[0] for i in ex_space.split(' ')[1:]]
        
        # assuming only two terms in logical expression
        unq_signs = list(set(pep_signs))
        
        # setting correlation condition
        if len(unq_signs) == 1:
            corr_cond = '+corr' # positive correlation
        else:
            corr_cond = '-corr' # negative correlation
        
        # grab corresponding peptide correlation
        pep_corr_val = corr_mat[pep_idc[0]][pep_idc[1]]
        
        # check for satisfaction of correlation condition
        # need high enough correlation for comparison
        if abs(pep_corr_val) < corr_thresh:
            site_direcs[idx] = 'unclear'         # inconclusive
            
        else:
            if corr_cond == '+corr':             # positive correlation
                if pep_corr_val < 0:
                    site_direcs[idx] = 'unclear' # inconclusive
                else:
                    site_direcs[idx] = 'phos'
            elif corr_cond == '-corr':           # negative correlation
                if pep_corr_val > 0:
                    site_direcs[idx] = 'unclear' # inconclusive
                else:
                    site_direcs[idx] = 'dephos'
                    
    # list of site directions
    return site_direcs            


# 'unq_peptide_list' does not distinguish phosphorylated sites
# i.e. 'SGCN' instead of 'S#GCN'
# 'ms_df' needs quant columns, peptide sequence, peptide start site, 
# site positions, and peptide slopes (i.e. column labeled 'slope')
def deconv_site(unq_peptide, ms_df, quant_cols):
    
    ms_df = ms_df.copy()
    
    
    ## Checking if deconvolution is possible
    
    # grab all peptide states associated with parent peptide
    ass_peptides = ms_df.ix[ms_df.Sequence.apply(lambda x: 
                    x.replace('.', '').replace('-', '').replace('S#', 
                   'S').replace('T#', 'T').replace('Y#', 
                   'Y') == unq_peptide), :].reset_index(drop=True)
    
    # order peptides by descending number of phosphorylated sites
    ass_peptides['num_sites'] = ass_peptides.Site_Position.apply(lambda x: 
                                x.count(';')+1)
    ass_peptides = ass_peptides.sort('num_sites', 
                                ascending=False).reset_index(drop=True)

    # number of theoretically phosphorylatable sites (S, T, Y)
    num_psites = unq_peptide.count('S')+unq_peptide.count('T')+unq_peptide.count('Y')
    
    ## Try to deconcolve site directions
    
    # get list of all phosphorylatable sites
    all_STY = find_idx(unq_peptide, 'S') + find_idx(unq_peptide, 
                                    'T') + find_idx(unq_peptide, 'Y')

    # NOTE: some peptides occur more than once in a protein
    if len(ass_peptides.Seq_Start.unique()) > 1:
        
        # for simplicity, we're ignoring these
        return None
      
    # getting absolute site positions
    all_STY = list(np.array(sorted(all_STY)) + 
                ass_peptides.Seq_Start.unique())
    all_STY_idx_df = pd.DataFrame(list(np.arange(len(all_STY))*2), 
                                  columns=['idx'])
    
    # building new dataframe with a row for each site
    site_df = pd.DataFrame(all_STY, columns=['Site_Position'])
    site_df['Site_Position'] = site_df.Site_Position.apply(lambda x: 
                                                           str(x))
    site_df['Gene_Symbol'] = ass_peptides.Gene_Symbol.unique().tolist()[0]
    site_df['Sequence'] = unq_peptide
    site_df['Sequence_ID'] = ass_peptides.Sequence_ID.unique().tolist()[0]
    
    # initializing 'site_directions' list for later
    site_directions = ['unclear' for i in range(num_psites)]
    
    # need same number of peptide states as sites
    if len(ass_peptides) < num_psites:
        
        # returning all site phosphorylation states as 'unclear'
        site_df['phos_state'] = site_directions
        return site_df
    
    # if only one phosphorylatable site, don't need to deconvolve
    if len(all_STY) == 1:
        
        # add phos/dephos/unknown designation
        if ass_peptides.slope.tolist()[0] > 0:
            site_df['phos_state'] = 'phos'
        else:
            site_df['phos_state'] = 'dephos'
            
        return site_df
    
    
    ## Iterate through combinations of peptides
    
    # if more than N peptides present, try different combos of N rows
    row_combos = list(combinations(np.arange(len(ass_peptides)), 
                                                 num_psites))
    for idc in row_combos:
        
        # only need N peptides (one per site)
        # NOTE: resetting indices here
        ass_peptides_trunc = ass_peptides[ass_peptides.index.isin(idc)].reset_index(drop=True)
        
        # get list of phosphosite combos present
        psite_combos = ass_peptides_trunc.Site_Position.apply(lambda x: 
                       x.split(';')).tolist()  
    
        # build matrix: making 2Nx2N matrix for N sites
        deconv_mat = np.zeros(shape=(2*num_psites,
                                     2*num_psites)) # initializing
    
        # setting freebee eq.s, e.g. S1 + S1- = 0 (one for each site)
        for idx in range(num_psites):
            deconv_mat[idx, idx*2] = 1
            deconv_mat[idx, idx*2+1] = 1
    
        # setting equations based on peptides, e.g. S1 + S2 = M12
        for idx, psite_combo in enumerate(psite_combos):
            
            site_idc = []
            for site in psite_combo:
                
                # getting index for correct assignment in matrix
                site_int = int(site)
                site_idx = all_STY.index(site_int)
                site_idc.append(site_idx)
                
            site_idc = np.array(site_idc)*2
    
            # fill in ones for sites present in the given peptide
            deconv_mat[idx+num_psites, site_idc] = 1
            
            # fill in ones for sites not present in a given combo/peptide
            sites_not_present = all_STY_idx_df.idx[~(all_STY_idx_df.idx.isin(site_idc))].tolist()
            sites_not_present = np.array(sites_not_present)+1
    
            if len(sites_not_present) > 0:
                deconv_mat[idx+num_psites, sites_not_present] = 1
            
        # specify Matrix (not array)
        deconv_mat = Matrix(deconv_mat)
        
        # make sure determinant isn't zero (i.e. not singular matrix)
        det_of_mat = det(deconv_mat)
            
        # if determinant does not equal zero
        # record sites that are deconvolvable
        # then reiterate to try to deconvolve more sites
        if det_of_mat != 0:
                        
            # making 'b' array; will include symbols instead of numbers
            freebee_const = list(np.zeros(num_psites))
            
            ms_idx = ass_peptides_trunc.index.tolist() # peptide indices
            ms_idc = [Symbol(str(i)) for i in ms_idx]  # generating symbols
            
            b_array = Matrix(freebee_const+ms_idc)
            
            
            ## Solve matrix for logical expressions
            
            # solution with symbols (i.e. indices) instead of all numbers
            soln = deconv_mat.LUsolve(b_array) 
            
            # simplifying logic encoded in solutions
            # only keeping psite equations (not nullsite equations)
            # i.e. skip every other equation
            soln = [get_logic(sol) for sol in soln][::2]
            
            # peptide correlation matrix
            pep_corr_mat = np.corrcoef(np.array(ass_peptides_trunc[quant_cols]))  
                
            # get phos/dephos/unknown designation
            site_directions_trial = deconv_core(soln, pep_corr_mat, 
                                          ass_peptides_trunc)
            
            # assign directions for sites that are currently unclear
            for idx, direc in enumerate(site_directions_trial):
                if site_directions[idx] == 'unclear':
                    site_directions[idx] = site_directions_trial[idx]    
    
    # assign phosphorylation direction to output dataframe
    site_df['phos_state'] = site_directions
    
    return site_df


# making function to extract site direction from deconvolved dataframe
# 'gene' is a gene symbol
# 'sites' are string numbers; two sites separated by semi-colon
def get_site_info(gene, sites, deconv_df):
    
    sites_list = [int(i) for i in sites.split(';')]
    
    # make sure everything is consistent for comparison
    deconv_df['Site_Position'] = deconv_df.Site_Position.apply(lambda x: 
                                int(x))
    
    directions=[]
    for site in sites_list:
        site_direction = list(deconv_df.phos_state[(deconv_df.Gene_Symbol == 
                        gene) & (deconv_df.Site_Position == site)])  
        
        if len(site_direction) == 0:
            directions.append('unclear')
        else:
            directions.append(site_direction[0])
    
    return directions


##########################################################################################


# NOTE: motifs must have same length
# 'motif1': motif coming from the phospho data set
# 'motif2': motif we're testing
# if using 'weighted' comparison, make sure motif is centered around site of interest 
# i.e. should have odd number of AAs
def motif_compare_core(motif1, motif2, weight=True):
    
    if len(motif1) != len(motif2):
        print 'Motifs must have same length.'
        return None  
    else:
        motif_len = len(motif1) * 1.0  
    
    match_array = (np.array(list(motif1)) == np.array(list(motif2))).astype(float)
    
    # unweighted identity
    identity = np.sum(match_array) * 1.0 / motif_len
    
    if weight:
        center = (motif_len - 1.0) / 2.0           # center position (length of left/right)
    
        # making linear function for weighting
        # match on ends --> 1.0, match on center --> (motif_len - 1.0) / 2.0
        left_fn = lambda x: x + 1.0                          
        right_fn = lambda x: -x + motif_len
        fn = lambda x: left_fn(x) if x <= center else right_fn(x)
        
        # making array of weights, then applying
        weight_array = np.array([fn(i) for i in np.arange(motif_len)])
        weight_match_array = match_array * weight_array
        
        tot_match = np.sum(weight_match_array)  # total weighted match
        max_match = np.sum(weight_array)        # max possible match
        
        # weighted identity
        identity = tot_match * 1.0 / max_match
        
    return identity


##########################################################################################


# 'site': should only have numerical characters, but should be string
# 'motif': sequence string of site in question
# 'site_database': dataframe with gene symbols, sites, and motifs
# 'attr_col': column in database with desired attributes (e.g. 'Fqn_Prc_Ixn' or 'Kinase')
def motif_compare(gene, site, motif, site_database, attr_col, 
                  weight=1, center_pos=6, FDR=0.01):
    
        
    ## first look for site-specific match

    # take data subset
    site_subset = site_database[((site_database.Gene_Symbol == gene) & 
                                (site_database.Site_Position == site)) & 
                                (site_database.Organism == 'human')]
    
    # obtain site attribute if possible
    if len(site_subset) > 0:  
    
        # obtain site attribute, position, and database motifs
        site_attr = site_subset[attr_col].values.tolist()
        site_pos_list = site_subset.Site_Position.values.tolist()
        database_motif = site_subset.Motif.values.tolist()
        
        # identities equal one since these are bona-fide matches
        identity_list = np.ones(len(site_subset)).tolist()
            
        # p-values equal zero since these are bona-fide matches
        pval_list = np.zeros(len(site_subset)).tolist()

        # getting difference in site position
        site_pos_diff_list = []
        for pos in site_pos_list:
            site_pos_diff = str(np.abs(int(site) - int(pos)))
            site_pos_diff_list.append(site_pos_diff)
            
        if attr_col != 'Kinase':
            # if only looking at functionality, will only keep first match we find
            id_types = 'site (site delta = '+site_pos_diff_list[0]+')' 
            pval_list = pval_list[0]
            identity_list = identity_list[0]
            site_attr = site_attr[0]
            
        else:
            # keep all matches for kinases (there might be more than 1 kinase per site)
            in_vivo = site_subset['InVivo'].values.tolist()
            id_types = ['site in vivo' if i == 1 else 'site in vitro' for i in in_vivo]
            id_types = [id_types[i]+' (site delta = '+site_pos_diff_list[i]+')' 
                        for i in np.arange(len(site_subset))] 

        return str(site_attr), str(identity_list), str(pval_list), str(id_types)


    
    ## next look for similarity between organisms
    
    # need to perform null hypothesis testing now for motif comparison
    # make a dataframe of decoy motifs
    decoy_motifs = site_database[['Gene_Symbol', 'Organism', 'Motif']]
    decoy_motifs['Decoy_Motif'] = decoy_motifs.Motif.apply(lambda x: 
                    ''.join(shuffle_array(np.array(list(x[:center_pos]+x[center_pos+1:])))))
    decoy_motifs['Decoy_Motif'] = decoy_motifs[['Decoy_Motif', 'Motif']].apply(lambda x: 
                                x[0][:center_pos]+x[1][center_pos]+x[0][center_pos:], axis=1)
    
    # take randomized subset (N=1000) of the decoy motifs
    shuffle_idc = shuffle_array(np.arange(len(decoy_motifs)))[:1000]
    decoy_motifs = decoy_motifs.ix[shuffle_idc, :].reset_index(drop=1)        
        
    # now replace the original database motif with the inputted substrate motif
    # and get decoy identities
    decoy_motifs['Motif'] = motif
    decoy_motifs['Identity'] = decoy_motifs[['Motif', 'Decoy_Motif']].apply(lambda x: 
                                motif_compare_core(x[0], x[1], weight), axis=1)
    
    # take data subset
    site_subset = site_database[(site_database.Gene_Symbol == gene) & 
                                (site_database.Organism != 'human')]

    if len(site_subset) > 0:
        # compare matches in subset
        site_subset['Substrate_Motif'] = motif
        site_subset['Identity'] = site_subset[['Substrate_Motif', 'Motif']].apply(lambda x: 
                                    motif_compare_core(x[0], x[1], weight), axis=1)

        site_attr = site_subset[attr_col].values.tolist()
        site_pos_list = site_subset.Site_Position.values.tolist()
        identity_list = site_subset.Identity.values.tolist()
        
        # obtain p-values
        pval_list = []
        for identity in identity_list:
            decoy_motifs['Null_Hypo_Wins'] = decoy_motifs.Identity.apply(lambda x: x > identity)
            pval = decoy_motifs.Null_Hypo_Wins.sum() * 1.0 / len(decoy_motifs)
            pval_list.append(pval)

        # getting difference in site position
        site_pos_diff_list = []
        for pos in site_pos_list:
            site_pos_diff = str(np.abs(int(site) - int(pos)))
            site_pos_diff_list.append(site_pos_diff)
        
        if attr_col != 'Kinase':
            id_types = ['similarity (site delta = '+site_pos_diff_list[i]+')' 
                        for i in np.arange(len(site_subset))]   
            
        else:
            # keep all matches for kinases (there might be more than 1 kinase per site)
            in_vivo = site_subset['InVivo'].values.tolist()
            id_types = ['similarity in vivo' if i == 1 else 'similarity in vitro' for i in in_vivo]
            id_types = [id_types[i]+' (site delta = '+site_pos_diff_list[i]+')' 
                        for i in np.arange(len(site_subset))] 
            
        # sort by p-values (use numpy)
        argsort = np.argsort(pval_list)
        site_attr = np.array(site_attr)[argsort]
        identity_list = np.array(identity_list)[argsort]
        pval_list = np.array(pval_list)[argsort]
        id_types = np.array(id_types)[argsort]
        
        # only keep significant matches
        pval_bool = pval_list < FDR
        
        if np.sum(pval_bool) > 0:
            site_attr = site_attr[pval_bool].tolist()
            identity_list = identity_list[pval_bool].tolist()
            pval_list = pval_list[pval_bool].tolist()
            id_types = id_types[pval_bool].tolist()

            # if looking at site functionality, only retain one record
            if attr_col != 'Kinase':
                site_attr = site_attr[0]
                identity_list = identity_list[0]
                pval_list = pval_list[0]
                id_types = id_types[0]
                
                return site_attr, identity_list, pval_list, id_types
            
            return str(site_attr), str(identity_list), str(pval_list), str(id_types)

    # if no statistically significant match found
    if attr_col != 'Kinase':
        return '', '', '', ''
    else:
        return '[]', '[]', '[]', '[]'


##########################################################################################


# takes single-site kinase assignments and maps them onto a dataset with composite sites
def recollapse_psites(orig_data, singlesite_data, 
                      attr_col='Kinase', identity_col='Identity_K', identity_cutoff=0.7):
    
    site_df = singlesite_data.copy()  # necessary to avoid bug
    output_df = orig_data.copy()      # necessary to avoid bug
    
    # convert stringed lists into actual lists
    if attr_col == 'Kinase':
        try:
            site_df['Kinase'] = site_df.Kinase.apply(lambda x: ast.literal_eval(x))
            singled_infer_sites[identity_col] = singled_infer_sites[identity_col].apply(lambda x: 
                            ast.literal_eval(x))
        except:
            pass
        
    # add 'Internal_GS_STY' column to original data
    # this will produce a list of GS_STY keys corresponding to the different sites present
    output_df['Internal_GS_STY'] = output_df[['Gene_Symbol', 'Site_Position']].apply(lambda x: 
                        [str(x[0])+' '+site for site in x[1].split(';')], axis=1)
    
    # put a GS_STY column in the single site dataframe
    site_df['GS_STY'] = site_df[['Gene_Symbol', 'Site_Position']].apply(lambda x: 
                        str(x[0])+' '+str(x[1]), axis=1)
    
    # remove all rows that do not meet the identity cutoff
    if attr_col == 'Kinase':
        site_df['Kinase'] = site_df[['Kinase', identity_col]].apply(lambda x: 
                [x[0][i] for i in np.arange(len(x[0])) if x[1][i] > identity_cutoff], axis=1)
        
        # only take unique kinases per site
        site_df['Kinase'] = site_df.Kinase.apply(lambda x: list(set(x)))
        
        # add on column of kinase assignments to original data
        # keep all replicate kinase assignments for a given site(s); important for later
        output_df['Kinase'] = output_df.Internal_GS_STY.apply(lambda x: 
                        [site_df.Kinase[site_df.GS_STY == 
                        key].values for key in x])
        
        output_df['Kinase'] = output_df.Kinase.apply(lambda x: 
                                        sum([k[0] for k in x if len(k) > 0], []))
        
    else:
        site_df[attr_col] = site_df[[attr_col, identity_col]].apply(lambda x: 
                x[0] if x[1] > identity_cutoff else '', axis=1)
    
                       
        # add on column of annotation to original data
        # keep all annotations for a given site(s)
        output_df[attr_col] = output_df.Internal_GS_STY.apply(lambda x: 
                            [site_df[attr_col][site_df.GS_STY == 
                            key].tolist() for key in x])
        
        output_df[attr_col] = output_df[attr_col].apply(lambda x: 
                                        [k[0] for k in x if len(k) > 0])
    
    # drop unnecessary column
    output_df = output_df.drop('Internal_GS_STY', 1)
    
    return output_df, site_df
    

##########################################################################################


# function that computes likely activity profile of kinases based on matched phosphopeptides
def get_kinase_activities(phos_df, singled_sites_df, quant_cols, 
                          cv_col='True_CV', cv_cutoff=0.1, log2fc_cutoff=1.0):

    ## get kinase activities

    # make copy of the datasets, just in case
    phos_copy = phos_df.copy()
    singled_sites_copy = singled_sites_df.copy()

    # compute max absolute fold-change for each peptide
    phos_copy['Max_Abs_FC'] = phos_copy[quant_cols].apply(lambda x: 
                                             np.max(np.abs(x)), axis=1)

    # also compute GS STY KEYs for each internal site for each peptide
    phos_copy['Internal_GS_STY'] = phos_copy[['Gene_Symbol', 'Site_Position']].apply(lambda x: 
                            [str(x[0])+' '+site for site in x[1].split(';')], axis=1)

    # also construct GS STY keys for each singled site
    singled_sites_copy['GS_STY_KEY'] = singled_sites_copy[['Gene_Symbol', 
                                                 'Site_Position']].apply(lambda x: 
                                                  str(x[0])+' '+str(x[1]), axis=1)
    
    # take subset of data that meets the indicated cutoffs
    phos_trunc = phos_copy[(phos_copy[cv_col] < cv_cutoff) & 
                                      (phos_copy.Max_Abs_FC >= log2fc_cutoff)]
    phos_trunc_insig = phos_copy[(phos_copy[cv_col] < cv_cutoff) & 
                                      (phos_copy.Max_Abs_FC < log2fc_cutoff)]
     
    # take corresponding sites from singled sites dataset
    kept_peptides = sum(phos_trunc.Internal_GS_STY.values.tolist(), [])
    lost_peptides = sum(phos_trunc_insig.Internal_GS_STY.values.tolist(), [])
    
    singled_sites_trunc = singled_sites_copy[singled_sites_copy.GS_STY_KEY.isin(kept_peptides)]
    singled_sites_insig = singled_sites_copy[singled_sites_copy.GS_STY_KEY.isin(lost_peptides)]
    
    # get all unique kinases and record number of appearances in truncated dataset
    all_kinases = sum(singled_sites_trunc.Kinase.values.tolist(), [])
    all_kinases_insig = sum(singled_sites_insig.Kinase.values.tolist(), [])
    unq_kinases = list(set(all_kinases))
    kin_occur_dict = {}
    remove_kinases = []
    for kin in unq_kinases:
        
        kinase_occur = len(find_idx(all_kinases, kin))
        kinase_insig_occur = len(find_idx(all_kinases_insig, kin))
        enrich_ratio = kinase_occur * 1.0 / (kinase_insig_occur + 1E-6) * len(all_kinases_insig) / len(all_kinases)
        
        # record enrichment ratio
        kin_occur_dict[kin] = enrich_ratio
        
        if enrich_ratio <= 1.0:
            remove_kinases.append(kin)
            
    # remove insignificant kinases
    singled_sites_trunc['Kinase'] = singled_sites_trunc.Kinase.apply(lambda x: 
                                    [k for k in x if k not in remove_kinases])

    # for a site with more than one kinase, 
    # give to the kinase that has greatest number of appearances
    singled_sites_trunc['Kinase_Enrich_Ratios'] = singled_sites_trunc.Kinase.apply(lambda x: 
                                                [kin_occur_dict[k] for k in x])
    
    singled_sites_trunc['Occams_Kinases'] = singled_sites_trunc[['Kinase', 
                        'Kinase_Enrich_Ratios']].apply(lambda x: 
                str([k for idx, k in enumerate(x[0]) if (x[1][idx] == np.max(x[1]))]), axis=1)
    singled_sites_trunc['Occams_Kinases'] = singled_sites_trunc.Occams_Kinases.apply(lambda x: 
                         ast.literal_eval(x))
    
    # keep all replicate kinase assignments for a given site; important for later
    phos_trunc['Kinase'] = phos_trunc.Internal_GS_STY.apply(lambda x: 
                    [singled_sites_trunc.Kinase[singled_sites_trunc.GS_STY == 
                    key].values for key in x])
    phos_trunc['Kinase'] = phos_trunc.Kinase.apply(lambda x: 
                                    sum([k[0] for k in x if len(k) > 0], []))

    # calculate contribution score for each kinase assigned to a given peptide
    # if kinase assigned twice to the same peptide (e.g. composite site), it gets double the score
    # ('MAPK8', 'CDK1', 'MAPK8') --> (0.33, 0.33, 0.33)
    all_final_kinases = sum(phos_trunc.Kinase.values.tolist(), [])
    unq_final_kinases = sorted(list(set(all_final_kinases)))

    kinase_activity_list = []
    peptide_num_list = []
    peptide_seq_list = []
    peptide_gene_symbol_list = []
    peptide_site_pos_list = []

    for kin in unq_final_kinases:

        # take all peptides where that kinase is assigned
        kin_peptides = phos_trunc.loc[phos_trunc.Kinase.apply(lambda x: kin in x), :]
        num_peptides = len(kin_peptides)
        peptide_gene_symbols = kin_peptides.Gene_Symbol.tolist()
        peptide_site_pos = kin_peptides.Site_Position.tolist() 
        peptide_seqs = kin_peptides.Sequence.tolist()

        # kinase contribution is the number of times the kinase is assigned 
        # divided by the number of kinases assigned (should be equal to 1 for most peptides)
        kin_peptides['Kinase_Contribution'] = kin_peptides.Kinase.apply(lambda x: 
                                            len(find_idx(x, kin)) * 1.0 / len(x))


        # get activity scores for the quantified columns
        peptide_weights = kin_peptides.Kinase_Contribution.values 
        peptide_weights_tile = np.tile(peptide_weights, (len(quant_cols), 1)).T
        peptide_data_array = kin_peptides[quant_cols].values
        kin_activity_vals = (np.sum(peptide_weights_tile * peptide_data_array, 
                                0) / np.sum(peptide_weights_tile, 0)).tolist()

        # record values
        kinase_activity_list.append(kin_activity_vals)
        peptide_num_list.append(num_peptides)
        peptide_gene_symbol_list.append(peptide_gene_symbols)
        peptide_site_pos_list.append(peptide_site_pos)
        peptide_seq_list.append(peptide_seqs)

    # convert to array
    kinase_activity_array = np.array(kinase_activity_list)
    kinase_activity_df = pd.DataFrame(kinase_activity_array, columns=quant_cols)
    kinase_activity_df['Gene_Symbol'] = unq_final_kinases
    kinase_activity_df['Num_Peptides'] = peptide_num_list
    kinase_activity_df['Peptide_Gene_Symbol'] = peptide_gene_symbol_list
    kinase_activity_df['Peptide_Site_Pos'] = peptide_site_pos_list
    kinase_activity_df['Peptide_Sequence'] = peptide_seq_list
    
    # return kinase activity dataframe
    # NOTE: the "Gene_Symbol" column refers to the kinase
    return kinase_activity_df


##########################################################################################

##########################################################################################

##########################################################################################

##########################################################################################

