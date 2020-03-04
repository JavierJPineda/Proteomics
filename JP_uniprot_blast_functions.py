## This file contains functions useful for characterizing proteins from Uniprot and BLAST outputs

# importing useful modules
import numpy as np
import networkx as nx
import pandas as pd

# scientific and computing stuff
import uniprot
import random
import re
import string
import glob
import os
import json
import ast
import urllib, urllib2

# importing stuff from biopython
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_protein
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML    
from Bio._py3k import StringIO 
from Bio._py3k import _as_string, _as_bytes 
from Bio._py3k import urlopen as _urlopen 
from Bio._py3k import urlencode as _urlencode 
from Bio._py3k import Request as _Request
from Bio.Blast.Applications import NcbiblastpCommandline
from Bio.Emboss.Applications import NeedleCommandline
from Bio.Align.Applications import ClustalwCommandline as clustalw_cline

# suppress annoying "deprecation warnings"
import warnings
warnings.filterwarnings("ignore")


##########################################################################################


def get_prot_seq(seq_id):
    
    print seq_id
    
    # just in case uniprot tries to sever the connection
    try:
        uniprot_data = uniprot.batch_uniprot_metadata([seq_id])
    except:
        try:
            uniprot_data = uniprot.batch_uniprot_metadata([seq_id])
        except:
            try:
                uniprot_data = uniprot.batch_uniprot_metadata([seq_id])
            except:
                uniprot_data = uniprot.batch_uniprot_metadata([seq_id])

    
    prot_content = uniprot_data.get(seq_id)
    
    if prot_content == None:
        return None
    
    sequence = str(prot_content.get('sequence'))
    return sequence


##########################################################################################


# defining function to get protein location from uniprot
# ok to input protein content string instead (in this case, there will be spaces)
def get_uniprot_loc(seq_id):
    
    # input should be a string in any case
    if not isinstance(seq_id, str):
        return None
    
    # if there are less than 3 spaces
    elif ' ' not in seq_id:
        # get all protein info
        uniprot_data = uniprot.batch_uniprot_metadata([seq_id])
        prot_content = uniprot_data.get(seq_id)
        prot_content = str(prot_content.get('comment'))
        
    else:
        prot_content = seq_id
    
    if prot_content == None:
        return None
    
    
    # looking for location
    loc_header = 'SUBCELLULAR LOCATION: '
    
    loc_idx = prot_content.find(loc_header)
    
    # if the location is present
    if loc_idx > -1:
        
        # isolating location information (not including header)
        # now left side begins with the location
        prot_content = prot_content[loc_idx + len(loc_header):]
        
        # making string a bit shorter by going up to next exclamation mark
        next_exclam = prot_content.find('!')
        prot_content = prot_content[:next_exclam]
        
        # finding where all punctuation marks occur in what's remaining
        punct_df = pd.DataFrame(list(string.punctuation), columns=['mark'])
        punct_df['pos'] = punct_df.mark.apply(lambda x: prot_content.find(x))
        
        # ignoring punctuations that do not occur
        punct_df = punct_df[punct_df.pos > -1]
        
        # finding the mark that occurs first
        punct_df = punct_df.sort('pos').reset_index(drop=True)
        first_punct = punct_df.pos[punct_df.index == 0].values[0]
        
        # getting location; removing leading and trailing spaces
        location = prot_content[:first_punct].strip()
        
        return location

    else:
        return None


##########################################################################################


# getting the word in a string in which a sequence of letters occurs
def get_word(prot_string, keyletters):
    
    '''
    In the sentence: "this is a methyltransferase protein here,"
    this function will output "methyltransferase" using the key letters "ase"
    '''
    
    # sometimes articles are picked up; don't want to include these
    articles = ['the ', 'of ', 'both ', 'an ', 'a ', 'on ', 'is ', 'in ', 'this ']
    art_df = pd.DataFrame(articles, columns=['article'])
    
    # getting the position where the key letters are
    keyletters_pos = prot_string.find(keyletters)

    # if the key letters are not found, then return None
    if keyletters_pos == -1:
        return None
    
    # taking up to last letter in the key letters, then taking reverse string
    bw_string = prot_string[:keyletters_pos+len(keyletters)][::-1]

    # getting the word
    # at this point if the inputted keyletters ended with a space,
    # ignore that space (it's already done it's job in specifying the word position)
    if keyletters[-1] == ' ':
        bw_string = bw_string[1:]
    
    
    # if the keyword occurs in the middle of the string, there should be a space before it
    if bw_string.find(' ') > -1:
    
        # alone, this will miss if the keyword occurs at the beginning of the string
        word_pos = len(bw_string) - bw_string.find(' ')     # the position of the word
    
    # if there's no space before the keyword, it's probably at the beginning of the string
    else:
        word_pos = 0
        
    output = prot_string[word_pos:keyletters_pos+len(keyletters)]
        
    # strip leading and tailing spaces
    output = output.strip()
        
    # make sure there's still a word
    if len(output) > 0:
        
        # if there's a dash in the output, make sure there's more than one letter before
        # if only one letter before the dash, get the word before
        # unless that word is an article
        first_dash = output.find('-')
        if first_dash > -1:
            
            if first_dash == 1:

                # it's possible that the word just starts with one character before a dash
                output_try = get_word_before(prot_string, output)
                
                if output_try != None:
                
                    # seeing if any articles are at the beginning of the string
                    art_test = art_df.article.apply(lambda x: 
                                                    output_try.find(x) == 0).tolist()
                    
                    # if there are no articles at the beginning of the new string
                    if True not in art_test:
                        output = output_try
        
        # if the first letter is capitalized
        if output[0].isupper():
            
            # if the second letter is not capitalized, make the first one lowercase
            if not output[1].isupper():
                output = output[0].lower() + output[1:]
        
        return output
    else:
        return None
    
    
##########################################################################################


# getting a word before another word in a string
def get_word_before(prot_string, keyword):
    
    '''
    In the sentence: "the protein has a kinase domain," 
    this function will output "kinase domain"
    '''
    
    # sometimes articles are picked up; don't want to include these
    articles = ['the ', 'of ', 'both ', 'an ', 'a ', 'on ', 'is ', 'in ', 'this ']
    
    # getting the position of the space before the keyword
    keyword = ' ' + keyword
    keyword_pos = prot_string.find(keyword)
    
    # if the key word is not present
    if keyword_pos == -1:
        return None
    
    # taking up to right before that position, then taking reverse string
    bw_string1 = prot_string[:keyword_pos][::-1] 
    
    # getting the position of the space right before the word preceding the keyword
    # this is assuming that the word occurs in the middle of the string
    last_space1 = bw_string1.find(' ')
    if last_space1 > -1:
        
        firstword_pos = len(bw_string1) - last_space1
        
    # if there's no space before the keyword, it's probably at the beginning of the string
    else:
        firstword_pos = 0
    
    # getting the output string
    output = prot_string[firstword_pos:keyword_pos + len(keyword)]
    
    # if the word before is one character, we may also want the word before that
    # unless that word is an article
    word_b4_len = len(prot_string[firstword_pos:keyword_pos])
    
    if word_b4_len == 1:
        
        # this goes up to the space before the firstword_pos
        bw_string2 = prot_string[:firstword_pos - 1][::-1]
        
        last_space2 = bw_string2.find(' ')
        if last_space2 > -1:
            
            extraword_pos = len(bw_string2) - last_space2 
        else:
            extraword_pos = 0
   
        # if we just picked up an article, go back to what we had before
        if prot_string[extraword_pos:firstword_pos-1] in articles:
            
            output = prot_string[firstword_pos:keyword_pos + len(keyword)]
            
        else:
            output = prot_string[extraword_pos:keyword_pos + len(keyword)]

    # if there are punctuation marks (e.g. parentheses), stripping them out
    # unless they are dashes
    output = output.replace('_', '').replace('-', '_')
    output = re.sub(r'[^\w\s]', '', output).replace('_', '-')

    # strip out leading and tailing spaces
    output = output.strip()
    
    
    if len(output) > 0:
        
        # if the first letter is capitalized
        if output[0].isupper():
            
            # if the second letter is not capitalized, make the first one lowercase
            # if the second character is a space or number, 
            # this will make the first letter lowercase
            if not output[1].isupper():
                output = output[0].lower() + output[1:]
        
        return output
    else:
        return None


##########################################################################################


# function to strip out '\n' from a string
def strip_slantn(string1):
    
    if string1 != None:
        string1 = string1.replace(' \\n', ' ')
        string1 = string1.replace('\\n ', ' ')
        string1 = string1.replace('\\n', ' ')
        
        return string1
    else:
        return None


##########################################################################################


# making function to get uniprot string
def get_uniprot_str(seq_id, just_gene=False):
    
    # get all protein info
    uniprot_data = uniprot.batch_uniprot_metadata([seq_id])
    prot_content = uniprot_data.get(seq_id)
    
    if prot_content == None:
        return None
    
    if not just_gene:
        prot_content = json.dumps(prot_content.get('comment'))
    else:
        prot_content = json.dumps(prot_content.get('gene'))[1:-1]
    # stripping out all '\n' carefully
    prot_content = strip_slantn(prot_content)
    
    return prot_content


##########################################################################################


# making function to get blast string
def get_blast_string(sequence, fasta_name, max_targets=3):
    
    output_name = fasta_name + '.xml'          # setting filenames
    fasta_name = fasta_name + '.faa'
    
    fasta = SeqRecord(Seq(str(sequence)))      # making fasta representation
    SeqIO.write(fasta, fasta_name, 'fasta')    # writing fasta file
    
    # writing command line argument
    # need to figure out how to reset default path
    comm_line = NcbiblastpCommandline(query=fasta_name, 
            db='Cdd', 
            out=output_name, outfmt=5, max_target_seqs=max_targets)

    stdout, stderr = comm_line()                   # running blastp
        
    # check if there was an output file generated; if not, return None
    if not os.path.exists(output_name):
        return None
    
    xml_file = open(output_name,'r')           # reading in results file
    xml_str = xml_file.read()
    
    # removing unnecessary files
    os.remove(fasta_name)
    os.remove(output_name)
    
    # make sure the string is not empty
    if len(xml_str) == 0:
        return None
    
    else:
        # returning all hits in one string
        return xml_str


##########################################################################################

# function to get sequence IDs from a gene symbol; gets ALL sequence IDs that match the organism criteria
# 'org': organism; string or list of strings (e.g. 'human' or 'homo sapiens')
# org=[] --> all organisms are returned
def gene2seqid(gene_symbol, org='homo sapiens'):

    if isinstance(org, str):
        org = [org]
    
    ### this code was taken from http://www.uniprot.org/help/programmatic_access example code
    
    url = 'http://www.uniprot.org/uploadlists/'

    params = {
    'from':'GENENAME',
    'to':'ACC',
    'format':'tab',
    'query':gene_symbol
    }

    data = urllib.urlencode(params)
    request = urllib2.Request(url, data)
    contact = 'javierjpineda13@gmail.com'
    request.add_header('User-Agent', 'Python %s' % contact)
    
    ###
    
    # just in case uniprot tries to sever the connection
    try: 
        response = urllib2.urlopen(request)
    except:
        try: 
            response = urllib2.urlopen(request)
        except:
            try: 
                response = urllib2.urlopen(request)
            except:
                response = urllib2.urlopen(request)
                    
    data_df = pd.read_csv(StringIO(response.read()), delimiter='\t')
    
    # make sure we are not considering protein fragments
    data_df = data_df.ix[data_df['Protein names'].apply(lambda x: 
                                 'fragment' not in x.lower()), :]
    
    # prioritize 'reviewed' sequence IDs and sort by sequence length (i.e. longer is preferred)
    data_df = data_df.sort(['Status', 'Length'], ascending=[1, 0])
    
    # retain only sequence ID and organism info
    data_df = data_df[['Entry', 'Organism']].rename(columns={'Entry':'Sequence_ID'}).reset_index(drop=True)
      
    if len(org) > 0:
        # only return desired organisms
        data_df = data_df.ix[data_df.Organism.apply(lambda x: sum([o in str(x).lower() for o in org]) > 0), :].reset_index(drop=True)
        
    # trim organism names
    data_df['Organism'] = data_df.Organism.apply(lambda x: x[:x.index(' (')].lower() if ' (' in x else x)
    
    return data_df


##########################################################################################


# function to take desired organisms and iterate through sequence IDs until a sequence is found in uniprot
def seq_iter(seq_df):
    
    orgs = seq_df.Organism.unique().tolist()
    
    seq_id_list = []
    seq_list = []
    output_orgs = []
    for org in orgs:
        seq_ids = seq_df.Sequence_ID[seq_df.Organism == org].tolist()
        
        for seq_id in seq_ids:
            seq = get_prot_seq(seq_id)
            
            if seq != None:
                seq_id_list.append(seq_id)
                seq_list.append(seq)
                output_orgs.append(org)
                break
                
        # if no sequence found in the end
        if seq == None:
            seq_id_list.append(np.nan)
            seq_list.append(np.nan)
            output_orgs.append(org)
                
    return seq_id_list, seq_list, output_orgs


##########################################################################################


# for simplicity, set 'fasta_name' to gene name
# 'org_list': list of organisms (strings)
def get_clustalw_alignment(seq_list, org_list, fasta_name, ref_species='homo sapiens', return_all=False):
    
    # verify we are only taking strings
    real_idc = []
    for idx, seq in enumerate(seq_list):
        
        if isinstance(seq, str):
            real_idc.append(idx)
            
    # get rid of nan seqs/organisms
    seq_list = list(np.array(seq_list)[real_idc])
    org_list = list(np.array(org_list)[real_idc])
    
    # replace spaces with underscores in organism names
    org_list = [o.replace(' ','_') for o in org_list]
    ref_species = ref_species.replace(' ','_')
    
    output_name = fasta_name + '.xml'          # setting filenames
    fasta_name = 'Fasta_Files/'+fasta_name+'.faa'
    
    for idx, seq in enumerate(seq_list):
        
        # NOTE: each sequence must be given a different ID (arbitrary)
        prot_seq = SeqRecord(Seq(str(seq)), id=org_list[idx])      # making fasta representation
        SeqIO.write(prot_seq, fasta_name+org_list[idx], 'fasta')    # writing fasta file
    
    # concatenating fasta files
    concat_fasta = ''.join([open('Fasta_Files/'+f).read() for f in os.listdir('Fasta_Files') if '.faa' in f])
    
    # removing unnecessary files so that python doesn't get confused
    delete = [os.remove('Fasta_Files/'+f) for f in os.listdir('Fasta_Files')]
    
    final_fasta = open(fasta_name,'w')
    final_fasta.write(concat_fasta)
    final_fasta.close()
    
    cwcline = clustalw_cline(cmd='/Applications/clustalw2', infile=fasta_name)
    stdout_str, stderr_str = cwcline()
    
    # remove fasta file once command executed
    os.remove(fasta_name)
    
    # read in alignment file
    aln_file = open(fasta_name[:fasta_name.index('.faa')]+'.aln', 'r')
    aln_df = pd.read_csv(StringIO(aln_file.read()))
    aln_df = aln_df.rename(columns={'CLUSTAL 2.1 multiple sequence alignment':'Align'})
    
    # cleaning up dataframe
    aln_df['Org'] = aln_df.Align.apply(lambda x: x[:x.index(' ')])
    aln_df['Align'] = aln_df[['Org', 'Align']].apply(lambda x: 
                      x[1][x[1].index(x[0])+len(x[0]):].replace(' ',''), axis=1)
    aln_df = aln_df[aln_df.Org.isin(org_list)].reset_index(drop=True)
    
    # getting aligned sequences
    align_seqs = aln_df.groupby('Org').Align.apply(sum).values.tolist()
    align_seqs = np.array([list(s) for s in align_seqs]).T  # converting sequences to array
    align_orgs = aln_df.groupby('Org').Align.apply(sum).index.tolist()  # corresponding organisms
    
    # making new dataframe
    seq_df = pd.DataFrame(align_seqs, columns=align_orgs)
    
    # finding conserved sites (NOTE: as written, this includes dashes)
    cons_site_df = seq_df.eq(seq_df[align_orgs[0]], axis='index')
    cons_site_idx = cons_site_df[align_orgs].apply(lambda x: np.sum(x) == len(align_orgs), axis=1)
    
    # marking conserved STY sites
    seq_df['Conserved_Site'] = cons_site_idx
    
    seq_df['Conserved_STY'] = seq_df[['Conserved_Site']+align_orgs].apply(lambda x: 
                              'S' if (x[0] == True) & (x[1] == 'S') else 'T' if (x[0] == True) & 
                                     (x[1] == 'T') else 'Y' if (x[0] == True) & 
                                     (x[1] == 'Y') else np.nan, axis=1)
    
    # remove breaks in human sequence (i.e. where human residue is a dash)
    seq_df = seq_df[~(seq_df[ref_species] == '-')].reset_index(drop=True)
    
    # assign residue numbers with respect to human sequence
    seq_df['Res_Num'] = np.arange(len(seq_df))+1
    
    if return_all:
        return seq_df
    
    
    # return conserved STY sites
    conserved_STY_df = seq_df.ix[seq_df.Conserved_STY.apply(lambda x: isinstance(x, str)), :].reset_index(drop=True)
    
    # joining residue AA and site position
    conserved_STY = conserved_STY_df[[ref_species, 'Res_Num']].apply(lambda x: 
                                    str(x[0])+str(int(x[1])), axis=1).values.tolist()
    
    # removing unnecessary files so that python doesn't get confused
    delete = [os.remove('Fasta_Files/'+f) for f in os.listdir('Fasta_Files')]
    
    # output the conserved STY sites
    return conserved_STY


##########################################################################################


# 'desired_orgs': list reference species first
# hard-coded to use human as the reference species
def get_cons_STY(gene_symbol, ref_sequence_id,
                 desired_orgs=['homo sapiens', 'xenopus tropicalis', 
                               'canis lupus', 'mus musculus'], demand_xenopus=True, return_all=False):
    
    # getting sequence IDs for other organisms
    seq_id_info = gene2seqid(gene_symbol, desired_orgs)
    
    # getting proteins sequences and final organism list
    # NOTE: this also includes a human sequence ID that may differ from the reference sequence ID inputted
    prot_seq_ids, prot_seqs, org_list = seq_iter(seq_id_info)
    
    # verify that reference organism is still present
    if desired_orgs[0] not in org_list:
        return ref_sequence_id, np.nan
    
    # verify that xenopus is present so that we can do a proper evolution search
    if demand_xenopus:
        if 'xenopus tropicalis' not in org_list:
            print 'Xenopus tropicalis is not present...'
            return ref_sequence_id, np.nan
    
    ref_idx = org_list.index(desired_orgs[0])  # get index of reference attributes
    
    if ref_sequence_id not in prot_seq_ids:
        ref_seq = get_prot_seq(ref_sequence_id)  # human sequence
    else:
        ref_seq = prot_seqs[ref_idx]
    
    # prioritize reference sequence ID for output
    if ref_seq != None:
        
        # reset reference attributes
        prot_seq_ids = prot_seq_ids[:ref_idx] + [ref_sequence_id] + prot_seq_ids[ref_idx+1:]
        prot_seqs = prot_seqs[:ref_idx] + [ref_seq] + prot_seqs[ref_idx+1:]
        
        final_ref_id = ref_sequence_id  # setting output sequence ID
        
    else:
        final_ref_id = prot_seq_ids[ref_idx]
    
    # getting conserved STY sites
    cons_STY = get_clustalw_alignment(prot_seqs, org_list, gene_symbol, return_all=return_all)
    
    return final_ref_id, cons_STY


##########################################################################################


# 'psite_col': could be 'Site_Position' or 'STY_sites'
def iter_cons_STY(phos_pep_df, psite_col='Site_Position'):
    
    # getting unique sequence IDs and corresponding gene symbols
    unq_prots_seqids = phos_pep_df[['Gene_Symbol', 'Sequence_ID']].drop_duplicates('Sequence_ID').reset_index(drop=True)
    
    # output values
    seqid_list = []
    cons_STY_list = []
    
    # just in case there are connection issues along the way
    none_error_gene = []
    none_error_seqid = []
    
    # iterate through sequence IDs
    for idx in range(len(unq_prots_seqids)):
        
        gene = unq_prots_seqids.Gene_Symbol[unq_prots_seqids.index == idx].values[0]
        seq_id = unq_prots_seqids.Sequence_ID[unq_prots_seqids.index == idx].values[0]
        
        try:
            seqid_output, sty_output = get_cons_STY(gene, seq_id)
            
            seqid_list.append(seqid_output)
            cons_STY_list.append(sty_output)

        except:
            seqid_list.append(seq_id)
            cons_STY_list.append(np.nan)
            
            none_error_gene.append(gene)
            none_error_seqid.append(seq_id)
    
    # recording output values in dataframe
    unq_prots_seqids['Sequence_ID'] = seqid_list
    unq_prots_seqids['Conserved_STY'] = cons_STY_list
    
    # get rid of any duplicated sequence IDs
    unq_prots_seqids = unq_prots_seqids.drop_duplicates('Sequence_ID').reset_index(drop=True)
    
    # re-try for the error-associated seqids
    retry_error_seqid = []
    retry_error_STY = []
    
    for idx in range(len(none_error_gene)):
        
        gene = none_error_gene[idx]
        seq_id = none_error_seqid[idx]
        
        try:
            seqid_output, sty_output = get_cons_STY(gene, seq_id)
            
            retry_error_seqid.append(seqid_output)
            retry_error_STY.append(sty_output)
            
        except:
            retry_error_seqid.append(seq_id)
            retry_error_STY.append(np.nan)
            
    # making a dict so we can reassign the relevant sequence IDs
    retry_seqid_dict = dict(zip(none_error_seqid, retry_error_seqid))
    retry_STY_dict = dict(zip(retry_error_seqid, retry_error_STY))
    
    # reassign sequence IDs for those that are error-associated
    unq_prots_seqids['Sequence_ID'] = unq_prots_seqids.Sequence_ID.apply(lambda x: 
                                        retry_seqid_dict[x] if x in none_error_seqid else x)
    
    # reassign conserved_STY for those that are error-associated
    unq_prots_seqids['Conserved_STY'] = unq_prots_seqids[['Sequence_ID', 'Conserved_STY']].apply(lambda x: 
                                        retry_STY_dict[x[0]] if x[0] in retry_error_seqid else x[1], axis=1)
    
    # once again remove duplicate sequence IDs
    unq_prots_seqids = unq_prots_seqids.drop_duplicates('Sequence_ID').reset_index(drop=True)
                
    # map back to phosphopeptide dataframe
    # merge first by sequence ID
    phos_pep_df = phos_pep_df.merge(unq_prots_seqids[['Sequence_ID', 'Conserved_STY']], 
                                    how='left', on='Sequence_ID', copy=False)
    
    # then merge by gene symbol for those that are NaN in the 'Conserved_STY' column
    nonnull_df = phos_pep_df[~phos_pep_df.Conserved_STY.isnull()]
    null_df = phos_pep_df[phos_pep_df.Conserved_STY.isnull()]
    
    null_df = null_df.drop('Conserved_STY', 1)  # drop column then merge back
    null_df = null_df.merge(unq_prots_seqids.drop_duplicates('Gene_Symbol')[['Gene_Symbol', 'Conserved_STY']], 
                                    how='left', on='Gene_Symbol', copy=False)
    
    # put df back together
    phos_pep_df = pd.concat([nonnull_df, null_df]).reset_index(drop=True)
    
    # now determine which phosphosites are conserved
    # drop letters in the conserved STY column
    phos_pep_df['Conserved_STY'] = phos_pep_df.Conserved_STY.apply(lambda x: 
                                                    [s[1:] for s in x] if isinstance(x, list) else x)

    # get boolean corresponding to conserved STY sites
    phos_pep_df['Site_Cons_Bool'] = phos_pep_df[[psite_col, 'Conserved_STY']].apply(lambda x: 
                                [s in x[1] for s in x[0].split(';')] if isinstance(x[1], list) else np.nan, axis=1)
    
    # output
    return phos_pep_df


##########################################################################################



