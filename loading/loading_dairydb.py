"""
Loading functions to retrieve convenient databases from DairyDB curated base
"""
import pandas as pd

from utils.utils import folder_paths


# Functions
def extract_data_from_dairydb() -> pd.DataFrame:
    """

    :return: Pandas DataFrame with the taxonomy, sequences, and sequences sizes
    """
    seq_file = open(folder_paths['data'] + 'DAIRYdb_v1.2.0_20190222_IDTAXA.fasta')
    line_to_read = True
    list_of_reads = []
    while line_to_read:
        taxo = seq_file.readline()[1:-2].upper().split(';')
        seq = seq_file.readline()[:-2]
        if seq == '':
            line_to_read = False
        else:
            list_of_reads.append([seq, len(seq)] + taxo)
    seq_file.close()
    df = pd.DataFrame(list_of_reads,
                      columns=['sequence', 'sequence_size', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus',
                               'species'])
    df.to_csv(folder_paths['data'] + 'dairydb_df.csv', index=False)
    return df
