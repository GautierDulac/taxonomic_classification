from typing import List


class StatLogger(object):
    """
    StatLogger saves the stats results
    """

    def __init__(self, log_path: str = None, csv_path: str = None, label_list: List = None, title_size: int = 120):
        """
        Initialize the log_path
        :param log_path: (string): where to store the log file
        :param csv_path: (string): where to store the csv file
        :param label_list: (List[str]): used as multi columns input
        :param title_size: (int): total length of the titles
        1. log_path must be provided to use the log() method. If the log file already exists, it will be deleted when Logger is initialized.
        2. If csv_path is provided, then one record will be write to the file everytime add_point() method is called.
       """
        self.title_size = title_size
        self.label_list = label_list
        self.log_path = log_path
        self.csv_path = csv_path
        self.log_file = None
        self.csv_file = None
        self.log_file = open(log_path, 'w')
        self.log_file.write('#' * self.title_size)
        if csv_path is not None:
            self.csv_file = open(csv_path, 'w')
            if label_list is not None:
                first_line = ''
                for name in label_list[:-1]:
                    first_line = first_line + name + ','
                first_line = first_line + label_list[-1] + '\n'
                self.csv_file.write(first_line)
            self.csv_file.flush()

    def log(self, text: str, title: str = None) -> None:
        """
        Write the text to log file then print it.
        :param text: text(string): text to log
        :param title: title name to put into brackets in the txt file
        :return: None
        """
        if title is not None:
            asked_title_size = len(title)
            left_hashtag_nb = (self.title_size - asked_title_size - 2) // 2
            right_hashtag_nb = self.title_size - asked_title_size - 2 - left_hashtag_nb
            self.log_file.write('\n')
            self.log_file.write('-' * self.title_size + '\n')
            self.log_file.write('-' * left_hashtag_nb + ' ' + title.upper() + ' ' + '-' * right_hashtag_nb + '\n')
            self.log_file.write('-' * self.title_size + '\n')
            self.log_file.write('\n')
        self.log_file.write(text + '\n')
        self.log_file.flush()
        print(text)

    def add_point(self, write_list=None) -> None:
        """
        Add a point to the plot
        :param write_list: list of coordinate to save when multiples
        :return:
        """
        if len(write_list) != len(self.label_list):
            raise ValueError('List of parameters to add should be the same length as the label list')
        else:
            line = ''
            for value in write_list[:-1]:
                line = line + str(value) + ','
            line = line + str(write_list[-1]) + '\n'

        # If csv_path is not None then write parameters to file
        if self.csv_path is not None:
            self.csv_file.write(line)
            self.csv_file.flush()

    def close_file(self) -> None:
        """
        Close the created file objects
        :return: None
        """
        if self.log_path is not None:
            self.log_file.close()
        if self.csv_path is not None:
            self.csv_file.close()
