import os
import unittest

import pandas as pd

from utils.loading import main_loading, folder_paths

project_directory = os.path.dirname(os.path.dirname(os.getcwd()))


class MyTestCase(unittest.TestCase):
    def test_main_loading(self):
        sampling = 1
        main_loading(sampling=sampling)
        df = pd.read_csv(folder_paths['data'] + 'gg_13_5_otus_rep_set_complete.csv'.format(
            int(sampling * 100)))
        target_columns = ['reference', 'sequence', 'sequence_size', 'file_num']
        self.assertEqual(len(df.columns), len(target_columns))
        for col_id in range(len(target_columns)):
            self.assertEqual(df.columns[col_id], target_columns[col_id])
        self.assertIn(len(df), [int(386957 * sampling) - 1, int(386957 * sampling), int(386957 * sampling) + 1])
        df = pd.read_csv(folder_paths['data'] + 'gg_13_5_taxonomy_complete.csv'.format(
            int(sampling * 100)))
        target_columns = ['reference', 'file_num', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        self.assertEqual(len(df.columns), len(target_columns))
        for col_id in range(len(target_columns)):
            self.assertEqual(df.columns[col_id], target_columns[col_id])
        self.assertIn(len(df), [391467])
        sampling = 0.01
        main_loading(sampling=sampling)
        df = pd.read_csv(folder_paths['data'] + 'gg_13_5_otus_rep_set_sampled_{}_percent.csv'.format(
            int(sampling * 100)))
        self.assertIn(len(df), [int(386957 * sampling) - 1, int(386957 * sampling), int(386957 * sampling) + 1])




if __name__ == '__main__':
    unittest.main()
