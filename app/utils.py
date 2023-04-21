import pandas as pd

class WineSample:
    def __init__(self, data_dict):
        self.df = pd.DataFrame(data_dict)

    def get_dataframe(self):
        return self.df
