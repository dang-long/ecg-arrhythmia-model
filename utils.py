import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import wfdb
import shutil
import os


def split_data(seed=42):
    folds = range(1, 11)
    folds = np.random.RandomState(seed).permutation(folds)
    # return folds[:8], folds[8:9], folds[9:]
    return folds[:8], folds[8:] #Modified. Long. 11.Jul.24, 
                                #original: return folds[:8], folds[8:9], folds[9:]
                                #Reason: we only need train (8) and val folds (2), no need for test fold



def prepare_input(ecg_file: str):
    if ecg_file.endswith('.mat'):
        ecg_file = ecg_file[:-4]
    ecg_data, _ = wfdb.rdsamp(ecg_file)
    nsteps, nleads = ecg_data.shape
    ecg_data = ecg_data[-15000:, :]
    result = np.zeros((15000, nleads)) # 30 s, 500 Hz
    result[-nsteps:, :] = ecg_data
    return result.transpose()


def cal_scores(y_true, y_pred, y_score):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)
    return precision, recall, f1, auc, acc


def find_optimal_threshold(y_true, y_score):
    thresholds = np.linspace(0, 1, 100)
    # f1s = [f1_score(y_true, y_score > threshold) for threshold in thresholds]
    #handle warning: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no true nor predicted samples. Use `zero_division` parameter to control this behavior.
    f1s = [f1_score(y_true, y_score > threshold, zero_division=1) for threshold in thresholds] #Modified, Long, New: 'zero_division=1'  original as comment 21.Apr.24
    return thresholds[np.argmax(f1s)]


def cal_f1(y_true, y_score, find_optimal):
    if find_optimal:
        thresholds = np.linspace(0, 1, 100)    
    else:
        thresholds = [0.5]
    f1s = [f1_score(y_true, y_score > threshold) for threshold in thresholds]
    #handle warning: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no true nor predicted samples. Use `zero_division` parameter to control this behavior.
    # f1s = [f1_score(y_true, y_score > threshold, zero_division=1) for threshold in thresholds] #Modified, Long, New: 'zero_division=1'  original as comment 21.Apr.24
    return np.max(f1s)


def cal_f1s(y_trues, y_scores, find_optimal=True):
    f1s = []
    for i in range(y_trues.shape[1]):
        f1 = cal_f1(y_trues[:, i], y_scores[:, i], find_optimal)
        f1s.append(f1)
    return np.array(f1s)


def cal_aucs(y_trues, y_scores):
    return roc_auc_score(y_trues, y_scores, average=None)

#Long add extra function
import json

# Function to load dictionary from file
def load_dictionary_from_file(filename):
    with open(filename, 'r') as file:
        dictionary = json.load(file)
    return dictionary

# Function to store dictionary to file
def store_dictionary_to_file(dictionary, filename):
    with open(filename, 'w') as file:
        json.dump(dictionary, file)

#function to store unique values to file
def store_class_label_to_file(values, filename):
    with open(filename, 'w') as file:
        for value in values:
            file.write(value + '\n')

#function to load unique values from file and store in a list
def load_class_label_from_file(filename):
    class_label = []
    with open(filename, 'r') as file:
        for line in file:
            class_label.append(line.strip())
    return class_label

# function to move files .hea and .mat from one folder to another. 
# for combination dataset, we need to move files from source folder to destination folder 
#Files name are extracted from 1st column of df
def move_files(df, source_folder, dest_folder):
    for index, row in df.iterrows():
        #get each file name from the 1st column of df
        file_name = row[0]
        #for each file name, append '.hea' and '.mat' to the file name, and move the files, one by one
        for ext in ['.hea', '.mat']:
            src_file = os.path.join(source_folder, file_name + ext)
            dest_file = os.path.join(dest_folder, file_name + ext)
            shutil.move(src_file, dest_file)
    print('Done moving files')
