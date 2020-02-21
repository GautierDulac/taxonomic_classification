"""
Loading functions to retrieve HVR statistics on loaded data
"""
# Package
import os

import numpy as np
import pandas as pd

from utils.logger import StatLogger
from utils.utils import folder_paths


# Main function
def main_stats_hvr(sequence_df: pd.DataFrame, hvr_df: pd.DataFrame, sequence_origin: str = '',
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
    Save basics stats on the given sequence_df
    :param sequence_df: dataframe with sequences
    :param analysis_path: saved path
    :return: None
    """
    log_path = analysis_path + 'sequences_log.txt'
    logger = StatLogger(log_path=log_path)
    # Global information on the sequence file
    logger.log(title='Global information', text='Total number of observations: {}'.format(len(sequence_df)))
    logger.log(text='Average sequence size: {}'.format(np.round(np.mean(sequence_df.sequence_size), 2)))
    logger.log(text='Standard deviation: {}'.format(np.round(np.std(sequence_df.sequence_size), 2)))
    logger.log(text='Minimal size: {}'.format(np.min(sequence_df.sequence_size)))
    logger.log(text='Maximal size: {}'.format(np.max(sequence_df.sequence_size)))
    # Taxonomy information on the sequence file
    logger.log(title='Taroxomy information')
    for depth in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
        logger.log(subtitle=depth)
        depth_value_counts = sequence_df[depth].value_counts()
        logger.log(text='Total number of unique {}: {}'.format(depth, len(depth_value_counts)))
        for rank in range(min(len(depth_value_counts), 5)):
            logger.log(text='Rank {} in {}: {} with {} occurences'.format(rank + 1,
                                                                          depth,
                                                                          depth_value_counts.index[rank],
                                                                          depth_value_counts[rank]))
        logger.log(text='{} missing values'.format(sum(sequence_df[depth].isna())))

    return


def get_basics_hvr(hvr_df: pd.DataFrame, analysis_path: str = '') -> None:
    """
    Save basics stats on the given hvr dataframe
    :param hvr_df: dataframe with retrieved HVRs
    :param analysis_path: saved path
    :return: None
    """
    log_path = analysis_path + 'hvr_log.txt'
    csv_path = analysis_path + 'hvr.csv'
    logger = StatLogger(log_path=log_path, csv_path=csv_path,
                        label_list=['HVR', 'occurences', 'percent', 'avg_size', 'min_size', 'max_size'])
    # Global information on the sequence file
    num_seq = len(hvr_df)
    num_hvr = len(hvr_df.columns) - 1
    logger.log(title='Global information', text='Total number of sequence analyzed: {}'.format(num_seq))
    logger.log(text='Number of HyperVariable regions tested: {}'.format(num_hvr))
    number_of_found_hvr = 0
    presents = dict()
    lower_hvr_value = num_seq
    lower_hvr = ''
    higher_hvr_value = 0
    higher_hvr = ''
    for hvr in hvr_df.columns[1:]:
        missings = sum(hvr_df[hvr].isna())
        presents[hvr] = num_seq - missings
        number_of_found_hvr += num_seq - missings
        if presents[hvr] < lower_hvr_value:
            lower_hvr_value = presents[hvr]
            lower_hvr = hvr
        if presents[hvr] > higher_hvr_value:
            higher_hvr_value = presents[hvr]
            higher_hvr = hvr
    total_potential = number_of_found_hvr * 100 / (num_seq * num_hvr)
    logger.log(text='Number of found HVR: {} - {}% of total potential'.format(number_of_found_hvr, np.round(
        total_potential, 2)))
    logger.log(text='Most represented HVR: {} - {} occurences - {}% of potential'.format(higher_hvr, higher_hvr_value,
                                                                                         np.round(
                                                                                             higher_hvr_value * 100 / num_seq,
                                                                                             2)))
    logger.log(text='Less represented HVR: {} - {} occurences - {}% of potential'.format(lower_hvr, lower_hvr_value,
                                                                                         np.round(
                                                                                             lower_hvr_value * 100 / num_seq,
                                                                                             2)))
    # HVR information on the sequence file
    logger.log(title='HyperVariable Regions specific information')
    for hvr in hvr_df.columns[1:]:
        logger.log(subtitle=hvr)
        if presents[hvr] > 0:
            percents = np.round(presents[hvr] * 100 / num_seq, 2)
            logger.log(
                text='{} found {} - {}% of total potential'.format(presents[hvr], hvr, percents))
            list_of_hvrs = list(hvr_df[hvr].dropna())
            list_of_sizes = [len(list_of_hvrs[i]) for i in range(len(list_of_hvrs))]
            avg_size = np.round(np.mean(list_of_sizes), 2)
            logger.log(text='Average size of {}: {}'.format(hvr, avg_size))
            min_size = np.min(list_of_sizes)
            logger.log(text='Minimal size of {}: {}'.format(hvr, min_size))
            max_size = np.max(list_of_sizes)
            logger.log(text='Maximal size of {}: {}'.format(hvr, max_size))
            logger.add_point(write_list=[hvr, presents[hvr], percents, avg_size, min_size, max_size])
        else:
            logger.log(text='No {} found in this database'.format(hvr))

    return


def get_hvr_by_taxonomies(sequence_df: pd.DataFrame, hvr_df: pd.DataFrame, analysis_path: str = '') -> None:
    """
    Get statistics of the interactions between taxonomies and found hvrs
    :param sequence_df: dataframe with sequences
    :param hvr_df: dataframe with retrieved HVRs
    :param analysis_path: saved_path
    :return: None
    """
    log_path = analysis_path + 'sequence_hvr_log.txt'
    logger = StatLogger(log_path=log_path)
    logger.log(title='Analysis of HyperVariable regions by Taxonomic Rank')
    joined_df = pd.merge(sequence_df, hvr_df, on='index', how='left')

    for depth in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
        logger.log(subtitle=depth)
        depth_value_counts = sequence_df.loc[:, depth].value_counts()
        total_groups = len(depth_value_counts)
        logger.log(text='Total number of unique {}: {}'.format(depth, total_groups))
        missings = sum(sequence_df.loc[:, depth].isna())
        logger.log(
            text='{} missing values - {}% of total'.format(missings, np.round(missings * 100 / len(sequence_df), 2)))
        for hvr in hvr_df.columns[1:]:
            local_df = joined_df[[depth, hvr]].dropna(axis=0)
            depth_hvr_value_counts = local_df[depth].value_counts()
            remaining_groups = len(depth_hvr_value_counts)
            percents = np.round(remaining_groups * 100 / total_groups)
            if remaining_groups > 0:
                logger.log(
                    text='Total number of remaining {} with at least one {} found: {} - {}% of total number of groups'.format(
                        depth, hvr, remaining_groups, percents))

    return
