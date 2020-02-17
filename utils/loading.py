"""
Loading functions to retrieve convenient databases from initial GreenGene data
"""
# Package
from os import listdir
from os.path import isfile, join

import pandas as pd

# Constant - Folder paths
folder_paths = {'Sequence': 'D:\\0 - Boulot\\5 - X4\\16. Research Paper\\data\\gg_13_5_otus\\rep_set\\',
                'Taxonomy': 'D:\\0 - Boulot\\5 - X4\\16. Research Paper\\data\\gg_13_5_otus\\taxonomy\\',
                'data': 'D:\\0 - Boulot\\5 - X4\\16. Research Paper\\data\\'}


# Main Function
def main_loading(sampling: float = 1) -> None:
    """
    Load all required files and save them in 'data' folder above github project.
    :return:
    """
    sequence_df = get_complete_df(type_to_extract='Sequence')
    taxonomy_df = get_complete_df(type_to_extract='Taxonomy')
    if sampling != 1:
        sampled_sequences = sequence_df.sample(frac=sampling)
        sampled_sequences.to_csv(folder_paths['data'] + 'gg_13_5_otus_rep_set_sampled_{}_percent.csv'.format(
            int(sampling * 100)), index=False)
        ref_for_taxonomy = sampled_sequences.reference.unique()
        sampled_taxonomy = taxonomy_df.loc[taxonomy_df.reference.isin(ref_for_taxonomy)]
        sampled_taxonomy.to_csv(folder_paths['data'] + 'gg_13_5_taxonomy_sampled_{}_percent.csv'.format(
            int(sampling * 100)), index=False)
    else:
        sequence_df.to_csv(folder_paths['data'] + 'gg_13_5_otus_rep_set_complete.csv', index=False)
        taxonomy_df.to_csv(folder_paths['data'] + 'gg_13_5_taxonomy_complete.csv', index=False)
    return


# Read saved file main function
def read_saved_file(type_to_read: str = '', sampling: float = 1) -> pd.DataFrame:
    """
    Give access to the saved dataframe
    :param type_to_read: desired file to retrieve (Sequence or Taxonomy)
    :param sampling: desired sampling frac
    :return: pd.DataFrame object
    """
    if type_to_read == 'Sequence':
        if sampling == 1:
            path = folder_paths['data'] + 'gg_13_5_otus_rep_set_complete.csv'
        else:
            path = folder_paths['data'] + 'gg_13_5_otus_rep_set_sampled_{}_percent.csv'.format(int(sampling * 100))
    elif type_to_read == 'Taxonomy':
        if sampling == 1:
            path = folder_paths['data'] + 'gg_13_5_taxonomy_complete.csv'
        else:
            path = folder_paths['data'] + 'gg_13_5_taxonomy_sampled_{}_percent.csv'.format(int(sampling * 100))
    else:
        ValueError('Asked type does not exist, check type_to_read argument')
        path = ''
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        FileNotFoundError('File not loaded yet, create the desired file with the main_loading function first')


# Load primers fasta data already saved
def load_primers(start_or_end: str = '') -> pd.DataFrame:
    """
    Load primers from data fasta file
    :param start_or_end: string telling if start or end primers are to be loaded
    :return:
    """
    seq_file = open(folder_paths['data'] + '{}_primers.fasta'.format(start_or_end))
    line_to_read = True
    list_of_reads = []
    while line_to_read:
        hvr = seq_file.readline()[1:-1]
        seq = seq_file.readline()[:-2]
        if hvr == '':
            line_to_read = False
        else:
            list_of_reads.append([hvr, seq, start_or_end])
    seq_file.close()
    return pd.DataFrame(list_of_reads, columns=['hvr', 'sequence', 'start_or_end'])


# Functions
def get_complete_df(type_to_extract: str = '') -> pd.DataFrame:
    """
    Return the complete DataFrame with all sequences extracted from all fasta files
    :param type_to_extract: Sequence or Taxonomy
    :return: pd.DataFrame with all the sequences / taxonomy
    """
    if not (type_to_extract in list(folder_paths.keys())):
        raise ValueError('type_to_extract should refer to a folder_paths key.')
    files_to_load = get_files_to_load(type_to_extract)
    number_to_load = [int(file_name.split('_')[0]) for file_name in files_to_load]
    final_df = pd.DataFrame()
    for num_file in number_to_load:
        print('Reading {} file {}'.format(type_to_extract, num_file))
        if type_to_extract == 'Sequence':
            final_df = pd.concat([extract_data_from_fasta(num_file=num_file), final_df])
        elif type_to_extract == 'Taxonomy':
            final_df = pd.concat([extract_taxonomy(num_file=num_file), final_df])
    return final_df


def get_files_to_load(type_to_extract: str = ''):
    """

    :param type_to_extract: Sequence or Taxonomy
    :return: list of files that can be read
    """
    onlyfiles = [f for f in listdir(folder_paths[type_to_extract]) if isfile(join(folder_paths[type_to_extract], f))]
    return onlyfiles


def extract_data_from_fasta(num_file: int = 0) -> pd.DataFrame:
    """
    Extract the sequences from a given fasta file.
    :param num_file: File number in which to look for sequences and references
    :return: Pandas DataFrame with the references, sequences, and sequences sizes
    """
    seq_file = open(folder_paths['Sequence'] + '{}_otus.fasta'.format(str(num_file)))
    line_to_read = True
    list_of_reads = []
    while line_to_read:
        ref = seq_file.readline()[1:-2]
        seq = seq_file.readline()[:-2]
        if ref == '':
            line_to_read = False
        else:
            list_of_reads.append([int(ref), seq, len(seq), num_file])
    seq_file.close()
    return pd.DataFrame(list_of_reads, columns=['reference', 'sequence', 'sequence_size', 'file_num'])


def extract_taxonomy(num_file: int = 0) -> pd.DataFrame:
    """
    Extract the taxonomy for a given fil number
    :param num_file: File number in which to look for sequences and references
    :return: Pandas DataFrame with the references, and all taxonomy info
    """
    seq_file = open(folder_paths['Taxonomy'] + '{}_otu_taxonomy.txt'.format(str(num_file)))
    list_of_reads = []
    line = seq_file.readline()[:-2].replace('[', '').replace(']', '').upper().split()
    while line:
        ref = int(line[0])
        k = line[1][3:-1]
        p = line[2][3:-1]
        c = line[3][3:-1]
        o = line[4][3:-1]
        f = line[5][3:-1]
        g = line[6][3:-1]
        s = line[7][3:-1]
        list_of_reads.append([ref, num_file, k, p, c, o, f, g, s])
        line = seq_file.readline()[:-2].replace('[', '').replace(']', '').upper().split()
    seq_file.close()
    return pd.DataFrame(list_of_reads,
                        columns=['reference', 'file_num', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus',
                                 'species'])
