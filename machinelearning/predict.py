
import logging.config
logger = logging.getLogger(__name__)

from datetime import datetime
import time
from typing import Dict
import json
import re
import pickle
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from machinelearning import ClassDetail

alert_mapping = {
    "mappings": {
        "properties": {
            "DosGoldenEye": {"type": "double"},
            "DosHulk": {"type": "double"},
            "DosLOIC": {"type": "double"},
            "DosSlowHttp": {"type": "double"},
            "DosSlowloris": {"type": "double"},
            "FTPPatator": {"type": "double"},
            "SSHPatator": {"type": "double"},
            "SqlInjWeb": {"type": "double"},
            "XssWeb": {"type": "double"},
            "heartbleed": {"type": "double"},
            "httpWebAttack": {"type": "double"},
            "portscan": {"type": "double"},
            "XssWeb": {"type": "double"},
            "flow_id": {"type": "text"},
            "description": {"type": "text"},
            "predicted_class": {"type": "text"},
            "predicted_class_probability": {"type": "text"},
            "level": {"type": "text"},   
            "timestamp": {"type": "date", "format": "dd/MM/yyyy HH:mm:ss"}, # 07/05/2022 20:13:47
        }
    }
}

flow_mapping = {
    "mappings": {
        "properties": {
            "flow_id": {"type": "text"},
            "source_ip": {"type": "ip"},
            "source_port": {"type": "integer"},
            "destination_ip": {"type": "ip"},
            "destination_port": {"type": "integer"},
            "protocol": {"type": "integer"},
            "timestamp": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss"}, # 2021-08-09 22:31:49
            "flow_duration": {"type": "double"},
            "total_forward_packet": {"type": "integer"},
            "total_backward_packet": {"type": "integer"},
            "total_length_of_forward_packet": {"type": "integer"},
            "total_length_of_backward_packet": {"type": "integer"},
            "forward_packet_max_length": {"type": "double"},
            "forward_packet_min_length": {"type": "double"},
            "forward_packet_mean_length": {"type": "double"},
            "forward_packet_std_length": {"type": "double"},
            "backward_packet_max_length": {"type": "double"},
            "backward_packet_min_length": {"type": "double"},
            "backward_packet_mean_length": {"type": "double"},
            "backward_packet_std_length": {"type": "double"},
            "flow_bytes": {"type": "double"},
            "flow_packets": {"type": "double"},
            "flow_iat_mean": {"type": "double"},
            "flow_iat_std": {"type": "double"},
            "flow_iat_max": {"type": "double"},
            "flow_iat_min": {"type": "double"},
            "fwd_iat_total": {"type": "double"},
            "fwd_iat_mean": {"type": "double"},
            "fwd_iat_mean": {"type": "double"},
            "fwd_iat_std": {"type": "double"},
            "fwd_iat_max": {"type": "double"},
            "fwd_iat_min": {"type": "double"},
            "bwd_iat_total": {"type": "double"},
            "bwd_iat_mean": {"type": "double"},
            "bwd_iat_std": {"type": "double"},
            "bwd_iat_max": {"type": "double"},
            "bwd_iat_min": {"type": "double"},
            "fwd_psh_flags": {"type": "double"},
            "bwd_psh_flags": {"type": "double"},
            "fwd_urg_flags": {"type": "double"},
            "bwd_urg_flags": {"type": "double"},
            "fwd_header_length": {"type": "double"},
            "bwd_header_length": {"type": "double"},
            "fwd_packets/s": {"type": "double"},
            "bwd_packets/s": {"type": "double"},
            "packet_length_min": {"type": "double"},
            "packet_length_max": {"type": "double"},
            "packet_length_mean": {"type": "double"},
            "packet_length_std": {"type": "double"},
            "packet_length_variance": {"type": "double"},
            "fin_flag_count": {"type": "double"},
            "syn_flag_count": {"type": "double"},
            "rst_flag_count": {"type": "double"},
            "psh_flag_count": {"type": "double"},
            "ack_flag_count": {"type": "double"},
            "urg_flag_count": {"type": "double"},
            "ece_flag_count": {"type": "double"},
            "down/up_ratio": {"type": "double"},
            "average_packet_size,": {"type": "double"},
            "fwd_bytes/bulk_avg": {"type": "double"},
            "fwd_packet/bulk_avg": {"type": "double"},
            "fwd_bulk_rate_avg": {"type": "double"},
            "bwd_bytes/bulk_avg": {"type": "double"},
            "bwd_packet/bulk_avg": {"type": "double"},
            "bwd_Bulk_rate_avg": {"type": "double"},
            "fwd_init_win_bytes": {"type": "integer"},
            "bwd_init_win_bytes": {"type": "integer"},
            "fwd_act_data_pkts": {"type": "integer"},
            "fwd_seg_size_min": {"type": "integer"},
            "active_mean": {"type": "double"},
            "active_std": {"type": "double"},
            "active_max": {"type": "double"},
            "active_min": {"type": "double"},
            "idle_mean": {"type": "double"},
            "idle_std": {"type": "double"},
            "idle_max": {"type": "double"},
            "idle_min": {"type": "double"},
            "label": {"type": "text"},
        }
    }
}

def readCSVRealTime(output_file, buffer_size=50):
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


def create_index(es, index_name: str, mapping: Dict) -> None:
    """
    Create an ES index.
    :param index_name: Name of the index.
    :param mapping: Mapping of the index
    """
    if not es.indices.exists(index=index_name):
        logger.info(f"Creating index {index_name} with the following schema: {json.dumps(mapping, indent=2)}")
        es.indices.create(index=index_name, ignore=400, body=mapping)

def predict(output_file):

    try:
        print('Connecting to Elastic Search')
        es = Elasticsearch("http://127.0.0.1:9200")
        create_index(es, 'flows', flow_mapping)
        create_index(es, 'alerts', alert_mapping)
        print("Successfully connected to ElasticSearch")

    except Exception as e:
        print("Error connecting to Elastic Search")
        print(e)

    alert = {}
    first = True
    try:
        for each in readCSVRealTime(output_file):
            if first:
                first = False
                head = each[0].rstrip().split(',')
                del each[0]
            refine = [line.rstrip().split(',') for line in each]

            current_batch_id = str(time.time()).replace('.','')

            # Inserting docs in Elastic Search
            json_head = [field.replace(' ','_').lower() for field in head]
            df = pd.DataFrame(columns=json_head, data=refine)
            for doc in df.apply(lambda x: x.to_dict(), axis=1):
                doc['flow_id'] = current_batch_id
                es.index(index='flows', body=json.dumps(doc))
                
            df = pd.DataFrame(refine, columns=head)
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

            alert['flow_id'] = current_batch_id
            alert['description'] = ClassDetail.CLASS_DETAIL[predicted_class]
            alert['predicted_class'] = predicted_class
            alert['predicted_class_probability'] = str(high)
            alert['level'] = getAlertLevel(high, predicted_class)
            now = datetime.now()
            alert['timestamp'] = now.strftime("%d/%m/%Y %H:%M:%S")
            print("alert is going to be saved")
            print(json.dumps(alert))
            with open('alerts/alerts.json', 'a+') as alerts:
                alerts.write(json.dumps(alert)+'\n')
            es.index(index='alerts', body=json.dumps(alert))

            # predicted_malware = predicted_class
            # print(predicted_malware)
            # print(detail)
            # print(alert)
    except Exception as e:
        logger.info("Error while calling sniffer.")
        logger.exception(e)


        