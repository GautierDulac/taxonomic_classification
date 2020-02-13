"""
Loading functions to retrieve convenient databases from initial GreenGene data
"""
# Package
from os import listdir
from os.path import isfile, join

import pandas as pd


# Main Function
def main_loading(sampling:float = 1) -> None:
    """
    Load all required files and save them in 'data' folder.
    :return:
    """
    sequence_df = get_complete_df()
    sequence_df.sample(frac=sampling).to_csv('..\\data\\complete_gg_13_5_otus_rep_set.csv', index=False)

    return


# Functions
def get_complete_df(
        folder_path: str = 'D:\\0 - Boulot\\5 - X4\\16. Research Paper\\taxonomic_classification\\data\\gg_13_5_otus\\rep_set/') -> pd.DataFrame:
    """
    Return the complete DataFrame with all sequences extracted from all fasta files
    :param folder_path: folder in which all fasta files are saved
    :return: pd.DataFrame with all the sequences
    """
    files_to_load = get_files_to_load(folder_path)
    number_to_load = [int(file_name.split('_')[0]) for file_name in files_to_load]
    final_df = pd.DataFrame()
    for num_file in number_to_load:
        print('Reading file {}'.format(num_file))
        final_df = pd.concat([extract_data_from_fasta(num_file), final_df])
    return final_df


def get_files_to_load(folder_path):
    """

    :param folder_path: folder in which all fasta files are saved
    :return: list of files that can be read
    """
    onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    return onlyfiles


def extract_data_from_fasta(num_file: int) -> pd.DataFrame:
    """
    Extract the sequences from a given fasta file.
    :param num_file: File number in which to look for sequences and references
    :return: Pandas DataFrame with the references, sequences, and sequences sizes
    """
    seq_file = open(
        'D:\\0 - Boulot\\5 - X4\\16. Research Paper\\taxonomic_classification\\data\\gg_13_5_otus\\rep_set\\{}_otus.fasta'.format(
            str(num_file)))
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
