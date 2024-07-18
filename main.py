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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/CPSC', help='Directory for data dir')
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--num-classes', type=int, default=int, help='Num of diagnostic classes')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=40, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use-gpu', default=False, action='store_true', help='Use GPU')
    parser.add_argument('--model-path', type=str, default='', help='Path to saved model')
    return parser.parse_args()


def train(dataloader, net, args, criterion, epoch, scheduler, optimizer, device):
    print('Training epoch %d:' % epoch)
    net.train()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
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
    if args.phase == 'train' and avg_f1 > args.best_metric:
        args.best_metric = avg_f1
        torch.save(net.state_dict(), args.model_path)
    else:
        aucs = cal_aucs(y_trues, y_scores)
        avg_auc = np.mean(aucs)
        print('AUCs:', aucs)
        print('Avg AUC: %.4f' % avg_auc)


if __name__ == "__main__":
    args = parse_args()
    args.best_metric = 0
    data_dir = os.path.normpath(args.data_dir)
    database = os.path.basename(data_dir)

    if not args.model_path:
        args.model_path = f'models/resnet34_{database}_{args.leads}_{args.seed}.pth'

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
    
    label_csv = os.path.join(data_dir, 'labels.csv') #Modified. Long. 23.Mar.24, original: os.path.join(data_dir, 'labelx.csv')
    # Introduce new test data
    test_data_dir ='data/test_dataset'
    test_label_csv = os.path.join(test_data_dir, 'labels.csv') #mod to handle missing class in test set
    
    # train_folds, val_folds, test_folds = split_data(seed=args.seed)
    train_folds, val_folds = split_data(seed=args.seed)
    test_folds = np.arange(1, 11)

    train_dataset = ECGDataset('train', data_dir, label_csv, train_folds, leads)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataset = ECGDataset('val', data_dir, label_csv, val_folds, leads)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    #Updae test dataset. Modified. Long. 11.Jul.24 Original: 
    
    # test_dataset = ECGDataset('test', data_dir, label_csv, test_folds, leads)
    test_dataset = ECGDataset('test', test_data_dir, test_label_csv, test_folds, leads)
    
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
    net = resnet34(input_channels=nleads).to(device) #Modified back. Long. 11.Jul.24, original: resnet50
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
    
    criterion = nn.BCEWithLogitsLoss()
    
    if args.phase == 'train':
        if args.resume:
            net.load_state_dict(torch.load(args.model_path, map_location=device))
        for epoch in range(args.epochs):
            train(train_loader, net, args, criterion, epoch, scheduler, optimizer, device)
            evaluate(val_loader, net, args, criterion, device)
    else:
        net.load_state_dict(torch.load(args.model_path, map_location=device))
        evaluate(test_loader, net, args, criterion, device)
