"""
Loading functions to retrieve all databases from downloaded ressources
"""
# Packages
import pandas as pd

from utils.utils import folder_paths


# Main function
def main_sequence_loading(desired_df: str = '') -> pd.DataFrame:
    """
    Depending on the desired df, return the complete data (seq and taxonomy available)
    :return: pd.DataFrame
    """
    #greengenes_df = pd.read_csv(folder_paths['data'] + 'gg_13_5_joined_complete.csv')
    #greengenes_df.drop(['file_num', 'reference'], axis=1, inplace=True)
    #greengenes_df['initial_db'] = 'GreenGenes'
    dairy_df = pd.read_csv(folder_paths['data'] + 'dairydb_df.csv')
    dairy_df['initial_db'] = 'DairyDB'
    #complete_df = pd.concat([greengenes_df, dairy_df])
    complete_df = dairy_db
    complete_df.reset_index(inplace=True, drop=True)
    complete_df.reset_index(inplace=True)
    if desired_df == '':
        return complete_df
    else:
        return complete_df.loc[complete_df.initial_db == desired_df]
