"""
Loading functions to retrieve HVR statistics on loaded data
"""
# Package
import os

import pandas as pd

from utils.logger import StatLogger
from utils.utils import folder_paths


# Main function
def main_stats(sequence_df: pd.DataFrame, hvr_df: pd.DataFrame, sequence_origin: str = '',
               primers_origin: str = '') -> None:
    """
    Compute relevant statistical analysis on HVR results
    :param sequence_df: sequence on which hvr are retrieved
    :param hvr_df: computed hvr from sequence_origin db, with primers_origin
    :param sequence_origin: initial sequences
    :param primers_origin: chosen primers
    :return: No return, only save analysis in results/statistics folder
    """
    analysis_folder = 'from_{}_sequences_with_{}_primers\\'.format(sequence_origin, primers_origin)
    analysis_path = folder_paths['stats_results'] + analysis_folder
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)

    # Basic information on sequence_df
    get_basics_sequence(sequence_df, analysis_path=analysis_path)

    # Basic information on found HVR (count, size)
    get_basics_hvr(hvr_df, analysis_path=analysis_path)

    # Combined hvr depending on taxonomic groups
    get_hvr_by_taxonomies(sequence_df, hvr_df, analysis_path=analysis_path)
    return


# Stats functions
def get_basics_sequence(sequence_df: pd.DataFrame, analysis_path: str = '') -> None:
    """

    :param sequence_df:
    :param analysis_path:
    :return:
    """
    log_path = analysis_path + 'sequences_log.txt'
    logger = StatLogger(log_path=log_path)
    logger.log(title='Global information', text='Total number of observations: {}'.format(len(sequence_df)))

    return


def get_basics_hvr(hvr_df, analysis_path):
    pass


def get_hvr_by_taxonomies(sequence_df, hvr_df, analysis_path):
    pass
