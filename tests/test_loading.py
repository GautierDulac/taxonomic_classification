import os
import unittest

import pandas as pd

from utils.loading import main_loading

project_directory = os.path.dirname(os.getcwd())


class MyTestCase(unittest.TestCase):
    def test_loading_sequences(self):
        sampling = 0.01
        main_loading(sampling=sampling)
        df = pd.read_csv(project_directory + '\\data\\complete_gg_13_5_otus_rep_set.csv')
        target_columns = ['reference', 'sequence', 'sequence_size', 'file_num']
        self.assertEqual(len(df.columns), len(target_columns))
        for col_id in range(len(target_columns)):
            self.assertEqual(df.columns[col_id], target_columns[col_id])
        self.assertIn(len(df), [int(386957 * sampling) - 1, int(386957 * sampling), int(386957 * sampling) + 1])


if __name__ == '__main__':
    unittest.main()
