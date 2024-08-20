import wfdb
import numpy as np
import pandas as pd

from glob import glob
import argparse
import os


#handle split_path in Windows or Linux
split_type = os.path.sep #newly added. Long. 23.Mar.24

def gen_reference_csv(data_dir, reference_csv):
    if not os.path.exists(reference_csv):
        recordpaths = glob(os.path.join(data_dir, '*.hea'))
        results = []
        for recordpath in recordpaths:
            patient_id = recordpath.split(split_type)[-1][:-4]
            _, meta_data = wfdb.rdsamp(recordpath[:-4]) 
            sample_rate = meta_data['fs']
            signal_len = meta_data['sig_len']
            age = meta_data['comments'][0]
            sex = meta_data['comments'][1]
            dx = meta_data['comments'][2]
            age = age[5:] if age.startswith('Age: ') else np.NaN
            sex = sex[5:] if sex.startswith('Sex: ') else 'Unknown'
            dx = dx[4:] if dx.startswith('Dx: ') else ''
            results.append([patient_id, sample_rate, signal_len, age, sex, dx])
        df = pd.DataFrame(data=results, columns=['patient_id', 'sample_rate', 'signal_len', 'age', 'sex', 'dx'])
        df.sort_values('patient_id').to_csv(reference_csv, index=None)


def gen_label_csv(label_csv, reference_csv, dx_dict, classes):
    if not os.path.exists(label_csv):
        results = []
        df_reference = pd.read_csv(reference_csv)
        for _, row in df_reference.iterrows():
            patient_id = row['patient_id']
            dxs = [dx_dict.get(code, '') for code in row['dx'].split(',')]
            # labels = [0] * 9
            labels = [0] * len(classes) #Modified to handle additional classes. Long. 21.Apr.24
            for idx, label in enumerate(classes):
                if label in dxs:
                    labels[idx] = 1
            results.append([patient_id] + labels)
        
        df = pd.DataFrame(data=results, columns=['patient_id'] + classes)
        n = len(df)
        folds = np.zeros(n, dtype=np.int8)
        for i in range(10):
            start = int(n * i / 10)
            end = int(n * (i + 1) / 10)
            folds[start:end] = i + 1
        df['fold'] = np.random.permutation(folds)
        columns = df.columns

           
        df[columns].to_csv(label_csv, index=None)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/CPSC', help='Directory to dataset')
    parser.add_argument('--num-classes', type=int, default=9, help='Num of diagnostic classes')
    args = parser.parse_args()

    data_dir = args.data_dir
    num_classes = args.num_classes

    reference_csv = os.path.join(data_dir, f'reference_{args.num_classes}_classes.csv')
    label_csv = os.path.join(data_dir, f'labels_{args.num_classes}_classes.csv')
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    if num_classes != 8:
        dx_dict = {
            '426783006': 'SNR', # Normal sinus rhythm
            '164889003': 'AF', # Atrial fibrillation
            '270492004': 'IAVB', # First-degree atrioventricular block
            '164909002': 'LBBB', # Left bundle branch block
            '713427006': 'RBBB', # Complete right bundle branch block
            '59118001': 'RBBB', # Right bundle branch block
            '284470004': 'PAC', # Premature atrial contraction
            '63593006': 'PAC', # Supraventricular premature beats
            '164884008': 'PVC', # Ventricular ectopics
            '429622005': 'STD', # ST-segment depression
            '164931005': 'STE', # ST-segment elevation
        }    
        classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
    
    else:
        dx_dict = {
            '426783006': 'SNR', # Normal sinus rhythm
            '164889003': 'AF', # Atrial fibrillation
            '270492004': 'IAVB', # First-degree atrioventricular block
            '164909002': 'LBBB', # Left bundle branch block
            '713427006': 'RBBB', # Complete right bundle branch block
            '59118001': 'RBBB', # Right bundle branch block
            '284470004': 'PAC', # Premature atrial contraction
            '63593006': 'PAC', # Supraventricular premature beats
            # PVC removed as it is not existed in the test set
            '429622005': 'STD', # ST-segment depression
            '164931005': 'STE', # ST-segment elevation
        }
        classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'STD', 'STE'] #removed 'PVC' as it is not existed in the test set

    gen_reference_csv(data_dir, reference_csv)
    gen_label_csv(label_csv, reference_csv, dx_dict, classes)

