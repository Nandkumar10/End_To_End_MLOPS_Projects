import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from mlflow.models import infer_signature
import os
from sklearn.model_selection import train_test_split,GridSearchCV
from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URL'] = 'https://dagshub.com/nandkumar.admane1/Ml_DagsHub.mlflow'
os.environ["MLFLOW_TRACKING_USERNAME"] = 'nandkumar.admane1'
os.environ["MLFLOW_TRACKING_PASSWORD"] = '2e6fb9e80b97ec909220f88d441d16cee73cff63'

## Load the params.yaml 

params = yaml.safe_load(open("params.yaml"))['train']

def evaluate(data_path,model_path):
        data = pd.read_csv(data_path)
        x = data.drop(columns=['Outcome'])
        y = data['Outcome']

        mlflow.set_tracking_uri('https://dagshub.com/nandkumar.admane1/Ml_DagsHub.mlflow')

        #Load the model from the disk
        model = pickle.load(open(model_path,'rb'))

        predctions = model.predict(x)
        accuracy = accuracy_score(y,predctions)

        #Log metricx to mlflow
        mlflow.log_metric('accuracy',accuracy)
        print('model accuracy' , accuracy)

if __name__ == "__main__":
        evaluate(params['data'],params['model'])