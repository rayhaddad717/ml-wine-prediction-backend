import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# this is an unmodified code from the tutorial
def createModelVerbose():
    df = pd.read_csv('resources/wine-quality.csv')
    print(df.head())
    df.info()
    df.describe().T
    df.isnull().sum()
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
    
    df.isnull().sum().sum()
    df.hist(bins=20, figsize=(10, 10))
    plt.show()
    plt.bar(df['quality'], df['alcohol'])
    plt.xlabel('quality')
    plt.ylabel('alcohol')
    plt.show()
    plt.figure(figsize=(12, 12))
    sb.heatmap(df.corr() > 0.7, annot=True, cbar=False)
    plt.show()
    df = df.drop('total sulfur dioxide', axis=1)
    df['best quality'] = [1 if x > 5 else 0 for x in df.quality]
    df.replace({'white': 1, 'red': 0}, inplace=True)
    features = df.drop(['quality', 'best quality'], axis=1)
    target = df['best quality']
    
    xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.2, random_state=40)
    
    xtrain.shape, xtest.shape
    norm = MinMaxScaler()
    xtrain = norm.fit_transform(xtrain)
    xtest = norm.transform(xtest)
    models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]
    
    for i in range(3):
        models[i].fit(xtrain, ytrain)
    
        print(f'{models[i]} : ')
        print('Training Accuracy : ', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
        print('Validation Accuracy : ', metrics.roc_auc_score(ytest, models[i].predict(xtest)))
        print()
    metrics.plot_confusion_matrix(models[1], xtest, ytest)
    plt.show()
    print(metrics.classification_report(ytest,models[1].predict(xtest)))

# created model class and modified code
class MLModel:
    def __init__(self):
        return
    def createAndTrainModel(self):
        print("Starting to train model")
        df = pd.read_csv('resources/wine-quality.csv')
        df.isnull().sum()
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mean())
        
        df.isnull().sum().sum()
        df.hist(bins=20, figsize=(10, 10))
        df = df.drop('total sulfur dioxide', axis=1)
        df['best quality'] = [1 if x > 5 else 0 for x in df.quality]
        df.replace({'white': 1, 'red': 0}, inplace=True)
        features = df.drop(['quality', 'best quality'], axis=1)
        self.columns=features.columns
        target = df['best quality']
        
        xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.2, random_state=40)
        
        xtrain.shape, xtest.shape
        norm = MinMaxScaler()
        xtrain = norm.fit_transform(xtrain)
        xtest = norm.transform(xtest)
        models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]
        
        for i in range(3):
            models[i].fit(xtrain, ytrain)
        self.model = models[0]
        self.norm = norm
        print("Model Trained Successfully")
    def predict(self,sample):
         # Preprocess the new data and normalize it using the same MinMaxScaler as the training data
        sample.replace({'white': 1, 'red': 0}, inplace=True)
        sample = sample[self.columns]
        new_data_norm = self.norm.transform(sample)

        # Load the trained model and use it to predict on the new data
        prediction = self.model.predict(new_data_norm)

        print(f"the prediction is {prediction}")  # this will print the predicted class label for the new data (0 or 1)
        return prediction[0]


# test data
datas = (
    pd.DataFrame({
        'fixed acidity': [7.5],
        'volatile acidity': [0.3],
        'citric acid': [0.45],
        'residual sugar': [2.0],
        'chlorides': [0.08],
        'free sulfur dioxide': [18.0],
        'density': [0.9966],
        'pH': [3.2],
        'sulphates': [0.6],
        'alcohol': [10.0],
        'type': ['white']
    }),
    pd.DataFrame({
        'fixed acidity': [7.2],
        'volatile acidity': [0.4],
        'citric acid': [0.26],
        'residual sugar': [1.8],
        'chlorides': [0.065],
        'free sulfur dioxide': [15.0],
        'density': [0.9962],
        'pH': [3.3],
        'sulphates': [0.5],
        'alcohol': [9.5],
        'type': ['white']
    }),
    pd.DataFrame({
    'fixed acidity': [6.5],
    'volatile acidity': [0.2],
    'citric acid': [0.35],
    'residual sugar': [1.5],
    'chlorides': [0.05],
    'free sulfur dioxide': [12.0],
    'density': [0.9955],
    'pH': [3.1],
    'sulphates': [0.7],
    'alcohol': [11.5],
    'type': ['red']
})
)

# creating and training the model
model = MLModel()
model.createAndTrainModel()

# testing the data
for data in datas:
    model.predict(data)




