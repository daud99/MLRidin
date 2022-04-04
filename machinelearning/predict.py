
import logging.config
logger = logging.getLogger(__name__)

from datetime import datetime
import json
import re
import pickle
import numpy as np
import pandas as pd

from machinelearning import ClassDetail

def readCSVRealTime(output_file, buffer_size=800):
    line_count = 0
    lines = []
    with open(output_file) as csv_file:
        while(True):
            line = csv_file.readline()
            if not line:
                # print('continuing')
                continue
            lines.append(line)
            line_count += 1
            # print('line_count')
            # print(line_count)
            if(line_count == buffer_size):
                # print("yielding")
                # print('line_count')
                # print(line_count)
                yield lines
                line_count = 0
                lines = []

def getAlertLevel(probability, predicted_class):
    if(probability <= 50 or predicted_class == 'normal'):
        return 'Level 0'
    elif (probability >= 60 and probability <= 80):
        return 'Level 1'
    elif (probability >= 80 and probability <= 100):
        return 'Level 3'

def predict(output_file):
    alert = {}
    first = True
    try:
        for each in readCSVRealTime(output_file):
            if first:
                first = False
                head = each[0].rstrip().split(',')
                del each[0]
            refine = [line.rstrip().split(',') for line in each]
            df = pd.DataFrame(refine, columns=head)
            # print(df)
            ndataset=df.drop(['Src IP','Src Port','Dst IP','Dst Port','Protocol','Timestamp'], axis=1)
            # Removing whitespaces in column names.
            ncol_names = [col.replace(' ', '') for col in ndataset.columns]
            ndataset.columns = ncol_names
            nlabel_names = ndataset['Label'].unique()
            nlabel_names = [re.sub("[^a-zA-Z ]+", "", l) for l in nlabel_names]
            nlabel_names = [re.sub("[\s\s]", '_', l) for l in nlabel_names]
            nlabel_names = [lab.replace("__", "_") for lab in nlabel_names]
            nlabel_names, len(nlabel_names)	
            ndataset.dropna(inplace=True)
            # ## Removing *non-finite* values
            ndataset = ndataset.loc[:, ndataset.columns != 'Label'].astype('float64')
            # Replacing infinite values with NaN values.
            ndataset = ndataset.replace([np.inf, -np.inf], np.nan)
            # Removing new NaN values.
            ndataset.dropna(inplace=True)
            novar = "models/novariance_stack_ensemble_model_9807.sav"
            features_no_variance = pickle.load(open(novar, 'rb'))
            ndataset = ndataset.drop(columns=features_no_variance)
            nfeatures = ndataset.loc[:, ndataset.columns != 'Label'].astype('float64')
            sclr= "models/scaler_stack_ensemble_model_9807.sav"	
            scaler = pickle.load(open(sclr,'rb'))
            nfeatures=scaler.transform(nfeatures)
            modl= "models/stack_ensemble_model_9807.sav"
            model= pickle.load((open(modl,'rb')))
            npreds_classes = model.predict(nfeatures)
            x= npreds_classes.shape[0]
            labl = "models/LE_stack_ensemble_model_9807.sav"
            LE= pickle.load((open(labl,'rb')))
            names=LE.inverse_transform([0,1,2,3,4,5,6,7,8,9,10,11,12])
            # printing the tuples in object directly
            high=0
            L=[]
            for name in enumerate(names):
                y=0
                for i in npreds_classes:
                    if i==name[0]:
                        y=y+1
                L.append(((y*100)/(x)))
                if (y*100)/(x) >= high:
                    high= (y*100)/(x)
                    predicted_class= name[1]
                if name[1]=='XssWeb' or name[1] =='normal':
                    # print(name[1]+":\t\t"+str(((y*100)/x))[:6])
                    alert[name[1]] = str(((y*100)/x))[:6]
                else:
                    # print(name[1]+":\t"+str(((y*100)/x))[:6])
                    alert[name[1]] = str(((y*100)/x))[:6]


            alert['description'] = ClassDetail.CLASS_DETAIL[predicted_class]
            alert['predicted_class'] = predicted_class
            alert['predicted_class_probability'] = str(high)
            alert['level'] = getAlertLevel(high, predicted_class)
            now = datetime.now()
            alert['timestamp'] = now.strftime("%d/%m/%Y %H:%M:%S")
            with open('alerts/alerts.json', 'a+') as alerts:
                alerts.write(json.dumps(alert)+'\n')
            # predicted_malware = predicted_class
            # print(predicted_malware)
            # print(detail)
            print(alert)
    except Exception as e:
        logger.info("Error while calling sniffer.")
        logger.exception(e)


        