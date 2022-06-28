# part of codes are modified from https://github.com/JunjieHu/cs769-assignments/tree/main/assignment2

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import f1_score

import json
from random import shuffle
import math

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode_train", type=bool, default=False)
    parser.add_argument("--path_train_df", type=str, default="/home/jifangao/N2C2_track3/data/N2C2-Track3-May3/train.csv")
    parser.add_argument("--path_dev_df", type=str, default="/home/jifangao/N2C2_track3/data/N2C2-Track3-May3/dev.csv")
    parser.add_argument("--path_test_df", type=str, default="")
    parser.add_argument("--path_train_add", type=str, default="/home/jifangao/N2C2_track3/added_fts_tr.npy")
    parser.add_argument("--path_dev_add", type=str, default="/home/jifangao/N2C2_track3/added_fts_dev.npy")
    parser.add_argument("--path_test_add", type=str, default="")
    parser.add_argument("--path_model", type=str, default="model/")
    parser.add_argument("--pretrained_model", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

# fix the random seed
def seed_everything(seed=41):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class N2C2_track3_dataset(Dataset):
    """
    A custom Dataset Class to be used for the dataloader
    """
    def __init__(self, args):
        if args.mode_train == True:
            print("Dataset initiated! Will train and return results on test set!")
            if args.path_test_df is not '': # test data is available
                print("Will concatenate train and dev data to train")
                self.df_train = pd.concat([pd.read_csv(args.path_train_df), pd.read_csv(args.path_dev_df)],
                                          ignore_index=True)
                self.df_test = pd.read_csv(args.path_test_df)
                self.added_fts_tr = np.load(args.path_train_add)
                self.added_fts_te = np.load(args.path_test_add)
            else:
                print("Test data is not provided. Will use dev data to test!")
                self.df_train = pd.read_csv(args.path_train_df)
                self.df_test = pd.read_csv(args.path_dev_df)
                self.added_fts_tr = np.load(args.path_train_add)
                self.added_fts_te = np.load(args.path_dev_add)
        else:
            # dev mode
            print("Dataset initiated! Will train and return results on dev and test set!")
            if args.path_test_df is not '': # test data is available
                print("Will concatenate train and dev data to train")
                self.df_train = pd.read_csv(args.path_train_df)
                self.df_dev = pd.read_csv(args.path_dev_df)
                self.df_test = pd.read_csv(args.path_test_df)
                self.added_fts_tr = np.load(args.path_train_add)
                self.added_fts_dv = np.load(args.path_dev_add)
                self.added_fts_te = np.load(args.path_test_add)
            else:
                print("Test data is not provided. Will split train data as df_train and df_dev and use dev data to test!")
                self.df_train, self.df_dev, self.added_fts_tr, self.added_fts_dv = self.split_train_data(args)
                self.df_test = pd.read_csv(args.path_dev_df)
                self.added_fts_te = np.load(args.path_dev_add)
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model,
                                               model_max_length=args.model_max_length)
        self.dic_label_index = {'Direct': 0, 'Indirect': 1, 'Neither': 2, 'Not Relevant': 3}
        self.train_lst = None
        self.dev_lst = None
        self.test_lst = None


    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        return self.dataframe.iloc[idx]

    def split_train_data(self, args):
        # use 80% hadmins as training data and rest as dev data
        df_train = pd.read_csv(args.path_train_df)
        added_fts_tr = np.load(args.path_train_add)
        hid_tr = np.random.choice(df_train['HADM ID'].unique(),
                                  size=int(df_train['HADM ID'].unique().shape[0]*0.8), replace=False)
        idx_tr = df_train[df_train['HADM ID'].isin(hid_tr)].index
        idx_dev = df_train[~df_train['HADM ID'].isin(hid_tr)].index
        return df_train.iloc[idx_tr].reset_index(drop=True), df_train.iloc[idx_dev].reset_index(drop=True), added_fts_tr[idx_tr], added_fts_tr[idx_dev]

    def convert_to_list(self):
        # convert dataframe to a list of dictionaries after tokenizing and padding
        data_iter = [self.df_train, self.df_test] if args.mode_train else [self.df_train, self.df_test, self.df_dev]
        for i, df_data in enumerate(data_iter):
            lst_data = []
            for rid, row in df_data.iterrows():
                encoded_dict = self.tokenizer(row['Assessment'], row['Plan Subsection'], truncation=True)
                # hand-crafted features
                if i == 0:
                    encoded_dict['added_features'] = list(self.added_fts_tr[rid])
                if i == 1:
                    encoded_dict['added_features'] = list(self.added_fts_te[rid])
                if i == 2:
                    encoded_dict['added_features'] = list(self.added_fts_dv[rid])
                # padding
                encoded_dict['input_ids'] += [0]*(args.model_max_length - len(encoded_dict['input_ids']))
                encoded_dict['attention_mask'] += [0]*(args.model_max_length - len(encoded_dict['attention_mask']))
                encoded_dict['token_type_ids'] += [1]*(args.model_max_length - len(encoded_dict['token_type_ids']))
                # label
                encoded_dict['label'] = self.dic_label_index[row['Relation']]
                # append
                lst_data.append(encoded_dict)
            if i == 0:
                self.train_lst = lst_data
            if i == 1:
                self.test_lst = lst_data
            if i == 2:
                self.dev_lst = lst_data
        if args.mode_train:
            print("Training and test have {}, {} samples.".format(len(self.train_lst),
                                                                  len(self.test_lst)))
        else:
            print("Training, dev, and test have {}, {}, {} samples.".format(len(self.train_lst),
                                                                            len(self.dev_lst),
                                                                            len(self.test_lst)))

class Transformer_base(nn.Module):
    """
    Pretrained transformer + one dense layer
    """
    def __init__(self, args):
        super(Transformer_base, self).__init__()
        self.tfm = AutoModel.from_pretrained(args.pretrained_model, output_hidden_states=True)
        self.linear_1 = nn.Linear(768+3, 4)
        self.act_1 = nn.ReLU()
        self.drop_1 = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, mask_attention, token_type_ids, added_fts):
        # hidden states of pretrained transformer
        out = self.tfm(input_ids=x, attention_mask=mask_attention, token_type_ids=token_type_ids).last_hidden_state[:,0,:]
        # disposition
        out = self.linear_1(torch.cat([out, added_fts], dim=1))
        out = self.drop_1(out)
        out = self.softmax(out)
        # output
        return out

def data_iter(data, batch_size):
    """
    Randomly shuffle training data, and partition into batches.
    """
    # Shuffle training data.
    np.random.shuffle(data)

    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sents = [data[i * batch_size + b]['input_ids'] for b in range(cur_batch_size)]
        labels = [data[i * batch_size + b]['label'] for b in range(cur_batch_size)]
        mask_attention = [data[i * batch_size + b]['attention_mask'] for b in range(cur_batch_size)]
        token_type_ids = [data[i * batch_size + b]['token_type_ids'] for b in range(cur_batch_size)]
        added_fts = [data[i * batch_size + b]['added_features'] for b in range(cur_batch_size)]
        yield sents, labels, mask_attention, token_type_ids, added_fts

def model_eval(lst_data, model, dataiter):
    """
    Given a list of dictionary and a model, make predictions and return micro-F1
    """
    list_yTrue = []
    list_yPred = []
    model.eval()
    for sents, labels, mask_attention, token_type_ids, added_fts in data_iter(lst_data, batch_size=4):
        X = torch.LongTensor(sents).to(device)
        mask_attention = torch.LongTensor(mask_attention).to(device)
        token_type_ids = torch.LongTensor(token_type_ids).to(device)
        added_fts = torch.LongTensor(added_fts).to(device)
        # predict
        y_pred = list(torch.argmax(model(X, mask_attention, token_type_ids, added_fts), dim=1).cpu().detach().numpy())[:len(labels)]
        # append to list
        list_yTrue.extend(labels)
        list_yPred.extend(y_pred)
    return f1_score(list_yTrue, list_yPred, average='micro')

if __name__ == "__main__":
    args = get_args()
    # Pytorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    # set random seeds
    seed_everything(args.seed)
    # load and process data
    dataset = N2C2_track3_dataset(args)
    dataset.convert_to_list()
    # initialize model
    model = Transformer_base(args).to(device)
    # set the model in 'train' mode and send it to the device
    model.train().to(device)
    # initialize the Adam optimizer (used for training/updating the model)
    optimizer = optim.AdamW(params=model.parameters(), lr=args.learning_rate)
    # loss functiom
    criterion = torch.nn.CrossEntropyLoss()
    # model path
    model_path = f"{args.path_model}tfm_added_{args.epoch}ep_{args.learning_rate}lr.pt"
    # training
    best_dev = 0
    for ep in range(args.epoch):
        model.train()
        train_loss = 0
        num_batches = 0
        for sents, labels, mask_attention, token_type_ids, added_fts in data_iter(dataset.train_lst, batch_size=args.batch_size):
            # convert to tensors
            X = torch.LongTensor(sents).to(device)
            mask_attention = torch.LongTensor(mask_attention).to(device)
            token_type_ids = torch.LongTensor(token_type_ids).to(device)
            added_fts = torch.LongTensor(added_fts).to(device)
            y = torch.LongTensor(labels).to(device)
            # set the gradients to zero
            optimizer.zero_grad()
            # predict
            y_pred = model(X, mask_attention, token_type_ids, added_fts)
            # BP
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            # add to total loss
            train_loss += loss.item()
            num_batches += 1
        # loss
        train_loss = train_loss / num_batches
        # print model performance in this epoch
        train_f1 = model_eval(dataset.train_lst, model, data_iter)
        print(f"Epoch {ep}\nTraining loss: {train_loss}\tTraining F1: {train_f1}")
        if not args.mode_train:
            dev_f1 = model_eval(dataset.dev_lst, model, data_iter)
            print(f"Dev F1: {dev_f1}")
            if dev_f1 > best_dev:
                torch.save(model.state_dict(), model_path)
    if args.mode_train:
        torch.save(model.state_dict(), model_path)

    # performance on test set
    print(f"\nTest F1: {model_eval(dataset.test_lst, model, data_iter)}")
