import os
import unittest

import pandas as pd

from loading.loading_greengenes import main_loading, folder_paths, read_saved_file
from loading.loading_primers import load_primers, get_dict_of_primers

project_directory = os.path.dirname(os.path.dirname(os.getcwd()))


class MyTestCase(unittest.TestCase):
    def test_loading_greengenes_1(self):
        main_loading()
        df = pd.read_csv(folder_paths['data'] + 'gg_13_5_otus_rep_set_complete.csv')
        target_columns = ['reference', 'sequence', 'sequence_size', 'file_num']
        self.assertEqual(len(df.columns), len(target_columns))
        for col_id in range(len(target_columns)):
            self.assertEqual(df.columns[col_id], target_columns[col_id])
        self.assertIn(len(df), [299499])

        df = pd.read_csv(folder_paths['data'] + 'gg_13_5_taxonomy_complete.csv')
        target_columns = ['reference', 'file_num', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        self.assertEqual(len(df.columns), len(target_columns))
        for col_id in range(len(target_columns)):
            self.assertEqual(df.columns[col_id], target_columns[col_id])
        self.assertIn(len(df), [302774])

        df = pd.read_csv(folder_paths['data'] + 'gg_13_5_joined_complete.csv')
        self.assertIn(len(df), [299499])

    def test_read_saved_file(self):
        df = read_saved_file(type_to_read='Sequence')
        target_columns = ['reference', 'sequence', 'sequence_size', 'file_num']
        self.assertEqual(len(df.columns), len(target_columns))

    def test_load_primers(self):
        df1 = load_primers(article='Chaudhary', forward_or_reverse='forward')
        self.assertEqual(len(df1), 14)
        df2 = load_primers(article='DairyDB', forward_or_reverse='reverse')
        self.assertEqual(len(df2), 17)

        dict_of_primers = get_dict_of_primers(article='Chaudhary')
        self.assertEqual(len(dict_of_primers), 14)





if __name__ == '__main__':
    unittest.main()
