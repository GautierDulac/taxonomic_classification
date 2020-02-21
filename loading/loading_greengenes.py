"""
Loading functions to retrieve convenient databases from initial GreenGene data
"""
# Package
from os import listdir
from os.path import isfile, join

import pandas as pd

from utils.utils import folder_paths


# Main Function
def main_loading_greengenes() -> None:
    """
    Load all required files and save them in 'data' folder above github project.
    :return:
    """
    sequence_df = get_complete_df(type_to_extract='Sequence')
    taxonomy_df = get_complete_df(type_to_extract='Taxonomy')
    joined_df = pd.merge(sequence_df, taxonomy_df, on=['reference', 'file_num'], how='left')

    sequence_df.to_csv(folder_paths['data'] + 'gg_13_5_otus_rep_set_complete.csv', index=False)
    taxonomy_df.to_csv(folder_paths['data'] + 'gg_13_5_taxonomy_complete.csv', index=False)
    joined_df.to_csv(folder_paths['data'] + 'gg_13_5_joined_complete.csv', index=False)
    return


# Read saved file main function
def read_saved_file(type_to_read: str = '') -> pd.DataFrame:
    """
    Give access to the saved dataframe
    :param type_to_read: desired file to retrieve (Sequence or Taxonomy)
    :return: pd.DataFrame object
    """
    if type_to_read == 'Sequence':
        path = folder_paths['data'] + 'gg_13_5_otus_rep_set_complete.csv'
    elif type_to_read == 'Taxonomy':
        path = folder_paths['data'] + 'gg_13_5_taxonomy_complete.csv'
    else:
        ValueError('Asked type does not exist, check type_to_read argument')
        path = ''
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        FileNotFoundError('File not loaded yet, create the desired file with the main_loading function first')


# Functions
def get_complete_df(type_to_extract: str = '') -> pd.DataFrame:
    """
    Return the complete DataFrame with all sequences extracted from all fasta files
    :param type_to_extract: Sequence or Taxonomy
    :return: pd.DataFrame with all the sequences / taxonomy
    """
    if not (type_to_extract in list(folder_paths.keys())):
        raise ValueError('type_to_extract should refer to a folder_paths key.')
    # files_to_load = get_files_to_load(type_to_extract)
    # number_to_load = [int(file_name.split('_')[0]) for file_name in files_to_load]
    # TODO Understand if necessary to work only with files 97 and 99 ?
    number_to_load = [97, 99]
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
