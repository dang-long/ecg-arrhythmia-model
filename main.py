import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset import ECGDataset
from resnet import resnet34, resnet50
from utils import cal_f1s, cal_aucs, split_data
import matplotlib.pyplot as plt #Add new Long. 12.08.24
import time #Add new Long. 12.08.24



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/CPSC', help='Directory for data dir')
    parser.add_argument('--test-dir', type=str, default='', help='Directory for test dataset')
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--num-classes', type=int, default=int, help='Num of diagnostic classes')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=40, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume') #Modified. Long. 31.Jul.24, original: False
    parser.add_argument('--use-gpu', default=False, action='store_true', help='Use GPU')
    parser.add_argument('--model-path', type=str, default='', help='Path to saved model')
    return parser.parse_args()


def train(dataloader, net, args, criterion, epoch, scheduler, optimizer, device):
    print('Training epoch %d:' % epoch)
    net.train()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        # print('Round:', idx, 'Label:', labels, 'Data:', data)
        data, labels = data.to(device), labels.to(device)        
        output = net(data)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    # scheduler.step()
    print('Loss: %.4f' % running_loss)
    

def evaluate(dataloader, net, args, criterion, device):
    print('Validating...')
    #get labels name from dataloader. Modified. Long. 11.Jul.24
    print('Labels:', dataloader.dataset.labels)    


    #Store result to file for presentation
    # Open a file to write the loss values
    log_file =  f'logs/resnet34_{database}_{args.leads}_{args.seed}_{args.num_classes}_log.txt'
    with open(log_file, 'w') as f:

        net.eval()
        running_loss = 0
        output_list, labels_list = [], []
        idx = 1
        for _, (data, labels) in enumerate(tqdm(dataloader)):
            data, labels = data.to(device), labels.to(device)
            # #print round id and label. Modified. Long. 11.Jul.24
            # print('Round:', idx, 'Label:', labels, 'Data:', data)
            # idx += 1
            # #end of modification
            output = net(data)
            loss = criterion(output, labels)
            running_loss += loss.item()
            output = torch.sigmoid(output)
            output_list.append(output.data.cpu().numpy())
            #print shape of output_list. Modified. Long. 11.Jul.24
            # print('Shape of output_list:', len(output_list))
            labels_list.append(labels.data.cpu().numpy())
            #print shape of labels_list. Modified. Long. 11.Jul.24
            # print('Shape of labels_list:', len(labels_list))

        print('Loss: %.4f' % running_loss)
        #Store values to log file. Newly added. Long 12.08.24
        f.write(f'Loss Epoch: {epoch} = {running_loss:.4f}\n')
        

        y_trues = np.vstack(labels_list)
        #store y_trues to a file. Modified. Long. 11.Jul.24
        #np.savetxt('y_trues_org.txt', y_trues, fmt='%d')

        #print shape of y_trues. Modified. Long. 11.Jul.24
        print('Shape of y_trues:', y_trues.shape)
        y_scores = np.vstack(output_list)
        #store y_scores to a file. Modified. Long. 11.Jul.24
        #np.savetxt('y_scores_org.txt', y_scores, fmt='%f')
        #print shape of y_scores. Modified. Long. 11.Jul.24
        print('Shape of y_scores:', y_scores.shape)

        f1s = cal_f1s(y_trues, y_scores)
        avg_f1 = np.mean(f1s)
        print('F1s:', f1s)
        print('Avg F1: %.4f' % avg_f1)
        #Store values to log file. Newly added. Long 12.08.24
        f.write(f'F1 Epoch {epoch}: ')        
        for val in f1s:
            f.write(f'{val:.4f} ')  # Format the number to 4 decimal places   
        f.write('\n')
        f.write(f'Avg F1 at Epoch {epoch} = {avg_f1:.4f}\n')
        
        #Original commented block

        # if args.phase == 'train' and avg_f1 > args.best_metric:
        #     args.best_metric = avg_f1
        #     torch.save(net.state_dict(), args.model_path)
        # else:
        #     aucs = cal_aucs(y_trues, y_scores)
        #     avg_auc = np.mean(aucs)
        #     print('AUCs:', aucs)
        #     print('Avg AUC: %.4f' % avg_auc)

        #Newly change. Long 12.08.24. Handle both Avg and avg AUC in validation
        #start. adding block
        if args.phase == 'train' and avg_f1 > args.best_metric:
            args.best_metric = avg_f1
            torch.save(net.state_dict(), args.model_path)
            aucs = cal_aucs(y_trues, y_scores)
            avg_auc = np.mean(aucs)
            print('AUCs:', aucs)
            print('Epoch with Best Avg F1 = %.4f' % avg_f1)
            print('Avg AUC: %.4f' % avg_auc)
        else:
            aucs = cal_aucs(y_trues, y_scores)
            avg_auc = np.mean(aucs)
            print('AUCs:', aucs)
            print('Avg AUC: %.4f' % avg_auc)

        #Store values to log file. Newly added. Long 12.08.24
        f.write(f'AUC Epoch {epoch}: ')        
        for val in aucs:
            f.write(f'{val:.4f} ')  # Format the number to 4 decimal places   
        f.write('\n')
        f.write(f'Avg AUC at Epoch {epoch} = {avg_auc:.4f}\n')
        #end. adding block

    return running_loss, avg_f1, avg_auc



if __name__ == "__main__":
    __spec__ = None #Added. Long. 31.Jul.24
    args = parse_args()
    args.best_metric = 0
    data_dir = os.path.normpath(args.data_dir)
    
    database = os.path.basename(data_dir)
    num_classes = args.num_classes

    # Introduce new test data
    test_dir = os.path.normpath(args.test_dir)
    test_label_csv = os.path.join(test_dir, f'labels_{args.num_classes}_classes.csv') #mod to handle missing class in test set

    if num_classes != 8:          
        classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
    else:
        classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'STD', 'STE'] #removed 'PVC' as it is not existed in the test set

    if not args.model_path:
        # args.model_path = f'models/resnet34_{database}_{args.leads}_{args.seed}.pth'
        args.model_path = f'models/resnet34_{database}_{args.leads}_{args.seed}_{args.num_classes}_classes.pth'

    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = 'cpu'
    
    if args.leads == 'all':
        leads = 'all'
        nleads = 12
    else:
        leads = args.leads.split(',')
        nleads = len(leads)
    
    label_csv = os.path.join(data_dir, f'labels_{args.num_classes}_classes.csv') #Modified. Long. 23.Mar.24, original: os.path.join(data_dir, 'labelx.csv')
       
    #if test_dir is not provided, use the same training data for validation and test
    #split those by fold number. Modified. Long. 30.Jul.24
    if args.test_dir == '':
        train_folds, val_folds = split_data(seed=args.seed)
        #remove test folds, as unused in training
        # test_dataset = ECGDataset('test', data_dir, label_csv, test_folds, leads)
        # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    else:
        train_folds, val_folds = split_data(seed=args.seed)
        # print('Validation folds:', val_folds)
        test_folds = np.arange(1, 11)
        test_dataset = ECGDataset('test', test_dir, test_label_csv, test_folds, leads)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    

    train_dataset = ECGDataset('train', data_dir, label_csv, train_folds, leads)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    val_dataset = ECGDataset('val', data_dir, label_csv, val_folds, leads)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
           
    net = resnet34(input_channels=nleads,num_classes=num_classes).to(device) #Modified. Long. 31.Jul.24, original: resnet50, without num_classes argument
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
    
    criterion = nn.BCEWithLogitsLoss()

    epoch_losses = []
    epoch_avg_F1 = []
    epoch_avg_AUC = []
    
    if args.phase == 'train':
        if args.resume:
            net.load_state_dict(torch.load(args.model_path, map_location=device))
        for epoch in range(args.epochs):
            train(train_loader, net, args, criterion, epoch, scheduler, optimizer, device)
            e_loss, e_avg_f1, e_avg_auc = evaluate(val_loader, net, args, criterion, device)
            # Append the loss for this epoch to the list
            epoch_losses.append(e_loss) #Add. Long 12.08.24
            # Append the avg F1 for this epoch to the list
            epoch_avg_F1.append(e_avg_f1) #Add. Long 12.08.24
            # Append the avg AUC for this epoch to the list
            epoch_avg_AUC.append(e_avg_auc) #Add. Long 12.08.24
    else:
        net.load_state_dict(torch.load(args.model_path, map_location=device))
        running_loss, avg_f1, avg_auc = evaluate(test_loader, net, args, criterion, device)
        # Append the loss for this epoch to the list
        epoch_losses.append(running_loss) #Add. Long 12.08.24
        # Append the avg F1 for this epoch to the list
        epoch_avg_F1.append(avg_f1) #Add. Long 12.08.24
        # Append the avg AUC for this epoch to the list
        epoch_avg_AUC.append(avg_auc) #Add. Long 12.08.24

    #Add new by Long. 12.08.24

    # Plot the loss/f1/auc over epochs
    plt.plot(range(0, args.epochs), epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.savefig(f'imgs/loss_changes_{args.phase}_{database}_{args.leads}_{args.seed}_{args.num_classes}_classes.png')

    # Clear the current figure
    time.sleep(1)
    plt.clf()
    plt.plot(range(0, args.epochs), epoch_avg_F1, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Avg F1')
    plt.title('Avg F1 Change over Epochs')
    plt.savefig(f'imgs/avg_F1_changes_{args.phase}_{database}_{args.leads}_{args.seed}_{args.num_classes}_classes.png')

    # Clear the current figure
    time.sleep(1)
    plt.clf()
    plt.plot(range(0, args.epochs), epoch_avg_AUC, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Avg AUC')
    plt.title('Avg AUC Change over Epochs')
    plt.savefig(f'imgs/avg_auc_changes_{args.phase}_{database}_{args.leads}_{args.seed}_{args.num_classes}_classes.png')


