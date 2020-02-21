"""
Loading functions to retrieve all databases from downloaded ressources
"""
# Packages
import pandas as pd

from utils.utils import folder_paths


# Main function
def main_hvr_loading(desired_sequences: str = '', desired_primers: str = 'DairyDB') -> pd.DataFrame:
    """
    Depending on the desired df, return the complete data (seq and taxonomy available)
    :return: pd.DataFrame
    """
    if desired_primers == 'DairyDB':
        if desired_sequences == 'DairyDB':
            hvr_df = pd.read_csv(folder_paths['data'] + 'hvr_DB_sequences_DB_primers.csv')
        elif desired_sequences == 'GreenGenes':
            hvr_df = pd.read_csv(folder_paths['data'] + 'hvr_GG_sequences_DB_primers.csv')
        else:
            hvr_df_GG_DB = pd.read_csv(folder_paths['data'] + 'hvr_DB_sequences_DB_primers.csv')
            hvr_df_DB_DB = pd.read_csv(folder_paths['data'] + 'hvr_DB_sequences_DB_primers.csv')
            hvr_df = pd.concat(hvr_df_DB_DB, hvr_df_GG_DB)
        return hvr_df
    elif desired_primers == 'Chaudhary':
        if desired_sequences == 'DairyDB':
            hvr_df = pd.read_csv(folder_paths['data'] + 'hvr_DB_sequences_CH_primers.csv')
        elif desired_sequences == 'GreenGenes':
            hvr_df = pd.read_csv(folder_paths['data'] + 'hvr_GG_sequences_CH_primers.csv')
        else:
            hvr_df_GG_CH = pd.read_csv(folder_paths['data'] + 'hvr_DB_sequences_CH_primers.csv')
            hvr_df_DB_CH = pd.read_csv(folder_paths['data'] + 'hvr_DB_sequences_CH_primers.csv')
            hvr_df = pd.concat(hvr_df_DB_CH, hvr_df_GG_CH)
        return hvr_df
