import pandas as pd
from app.models import wine_quality_schema,wine_quality_schema_csv
from voluptuous import MultipleInvalid

# util class to validate wine sample objects
class WineSample:
    def __init__(self, data_dict,is_cvs = False):
        try:
            if(is_cvs):
                wine_quality_schema_csv(data_dict)
                # add values to array for pd
                converted_dictionary =   {
                    'fixed acidity': [data_dict['fixed acidity']],
                    'volatile acidity': [data_dict['volatile acidity']],
                    'citric acid': [data_dict['citric acid']],
                    'residual sugar': [data_dict['residual sugar']],
                    'chlorides': [data_dict['chlorides']],
                    'free sulfur dioxide': [data_dict['free sulfur dioxide']],
                    'density': [data_dict['density']],
                    'pH': [data_dict['pH']],
                    'sulphates': [data_dict['sulphates']],
                    'alcohol': [data_dict['alcohol']],
                }   
            else:
                wine_quality_schema(data_dict)
                # change the dictionary keys to match csv file
                converted_dictionary =   {
                    'fixed acidity': [data_dict['fixedAcidity']],
                    'volatile acidity': [data_dict['volatileAcidity']],
                    'citric acid': [data_dict['citricAcid']],
                    'residual sugar': [data_dict['residualSugar']],
                    'chlorides': [data_dict['chlorides']],
                    'free sulfur dioxide': [data_dict['freeSulfurDioxide']],
                    'density': [data_dict['density']],
                    'pH': [data_dict['pH']],
                    'sulphates': [data_dict['sulphates']],
                    'alcohol': [data_dict['alcohol']],
                }       
            self.validated = True
            self.df = pd.DataFrame(converted_dictionary)
        except MultipleInvalid as e:
            error_messages = {}
            for error in e.errors:
                key = str(error.path[0])
                error_messages[key] = error.msg
            self.errors = error_messages
            self.validated = False

    # dataframes are used to predict 
    def get_dataframe(self):
        return self.df
    

