# python model_dev.py \
# --epoch=2 \
# --learning_rate=2e-5 \
# --pretrained_model=/home/jifangao/N2C2_track3/downloaded_models/PubmedBERTbase-MimicBig-EntityBERT \
# --local_model=True \
# --mode_train=True


import random
import pickle
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
    parser.add_argument("--path_test_df", type=str, default="/home/jifangao/N2C2_track3/data/n2c2_track3_test/n2c2_track3_test.csv")
    parser.add_argument("--path_train_add", type=str, default="/home/jifangao/N2C2_track3/added_fts_tr_1010.npy")
    parser.add_argument("--path_dev_add", type=str, default="/home/jifangao/N2C2_track3/added_fts_dev_1010.npy")
    parser.add_argument("--path_test_add", type=str, default="/home/jifangao/N2C2_track3/added_fts_te_1010.npy")
    parser.add_argument("--path_train_metamap", type=str, default="/home/jifangao/N2C2_track3/ner_metamap_training_1010.pickle")
    parser.add_argument("--path_dev_metamap", type=str, default="/home/jifangao/N2C2_track3/ner_metamap_dev_1010.pickle")
    parser.add_argument("--path_test_metamap", type=str, default="/home/jifangao/N2C2_track3/ner_metamap_test_1010.pickle")
    parser.add_argument("--path_model", type=str, default="model/")
    parser.add_argument("--path_dic_cui_chapter", type=str, default="/home/jifangao/N2C2_track3/dic_cui_chapter_1010.pickle")
    parser.add_argument("--pretrained_model", type=str, default="/home/jifangao/N2C2_track3/downloaded_models/PubmedBERTbase-MimicBig-EntityBERT")
    parser.add_argument("--local_model", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epoch", type=int, default=2)
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
                                                       model_max_length=args.model_max_length,
                                                       local_files_only=args.local_model)
        self.dic_label_index = {'Direct': 0, 'Indirect': 1, 'Neither': 2, 'Not Relevant': 3}
        self.idx_start_train = self.collect_start_ids(self.df_train)
        self.idx_start_test = self.collect_start_ids(self.df_test)
        if args.mode_train != True:
            self.idx_start_dev = self.collect_start_ids(self.df_dev)
        self.train_lst = None
        self.dev_lst = None
        self.test_lst = None


    def __len__(self):
        return self.df_train.shape[0], self.df_dev.shape[0], self.df_test.shape[0]

    def collect_start_ids(self, df):
        # return row ids of samples that are the first section of corresponding notes
        df_copy = df.copy()
        df = df.copy()
        df['rank_total'] = range(df.shape[0])
        df['rank'] = df.groupby('HADM ID')['rank_total'].rank(method='dense').astype(int)
        return list(df[df['rank']==1].index)

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
                encoded_dict['row_id'] = rid
                # hand-crafted features
                if i == 0:
                    encoded_dict['added_features'] = [1 if rid in self.idx_start_train else 0] + list(self.added_fts_tr[rid])
                if i == 1:
                    encoded_dict['added_features'] = [1 if rid in self.idx_start_test else 0] + list(self.added_fts_te[rid])
                if i == 2:
                    encoded_dict['added_features'] = [1 if rid in self.idx_start_dev else 0] + list(self.added_fts_dv[rid])
                # padding
                encoded_dict['cum_pred'] = [0, 0, 0, 0]
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
        self.tfm = AutoModel.from_pretrained(args.pretrained_model, output_hidden_states=True, local_files_only=args.local_model)
        self.linear_1 = nn.Linear(768+1+3+4, 4)
        self.act_1 = nn.ReLU()
        self.drop_1 = nn.Dropout(args.dropout)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, mask_attention, token_type_ids, added_fts, cum_pred):
        # hidden states of pretrained transformer
        out = self.tfm(input_ids=x, attention_mask=mask_attention, token_type_ids=token_type_ids).last_hidden_state[:,0,:]
        # disposition
        out = self.linear_1(torch.cat([out, added_fts, cum_pred], dim=1))
        out = self.drop_1(out)
        out = self.softmax(out)
        # output
        return out

def data_iter(data, batch_size, is_shuffle=True):
    """
    Randomly shuffle training data, and partition into batches.
    """
    # Shuffle training data.
    if is_shuffle:
        np.random.shuffle(data)

    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sents = [data[i * batch_size + b]['input_ids'] for b in range(cur_batch_size)]
        rids = [data[i * batch_size + b]['row_id'] for b in range(cur_batch_size)]
        labels = [data[i * batch_size + b]['label'] for b in range(cur_batch_size)]
        mask_attention = [data[i * batch_size + b]['attention_mask'] for b in range(cur_batch_size)]
        token_type_ids = [data[i * batch_size + b]['token_type_ids'] for b in range(cur_batch_size)]
        added_fts = [data[i * batch_size + b]['added_features'] for b in range(cur_batch_size)]
        cum_pred = [data[i * batch_size + b]['cum_pred'] for b in range(cur_batch_size)]
        yield sents, labels, mask_attention, token_type_ids, added_fts, cum_pred

def model_eval(lst_data, model, dataiter, mode='train'):
    """
    Given a list of dictionary and a model, make predictions and return micro-F1
    """
    list_yTrue = []
    list_yPred = []
    list_probPred = []
    model.eval()
    if mode == 'train':
        for sents, labels, mask_attention, token_type_ids, added_fts, cum_pred in data_iter(lst_data, batch_size=4, is_shuffle=False):
            X = torch.LongTensor(sents).to(device)
            token_type_ids = torch.LongTensor(token_type_ids).to(device)
            mask_attention = torch.LongTensor(mask_attention).to(device)
            added_fts = torch.FloatTensor(added_fts).to(device)
            cum_pred = torch.LongTensor(cum_pred).to(device)
            # predict
            y_pred_ary = torch.exp(model(X, mask_attention, token_type_ids, added_fts, cum_pred)).detach().cpu().numpy()
            y_pred = np.argmax(y_pred_ary, axis=1)
            # append to list
            list_probPred.extend(y_pred_ary)
            list_yTrue.extend(labels)
            list_yPred.extend(y_pred)
        return f1_score(list_yTrue, list_yPred, average='macro'), list_yTrue, list_yPred, list_probPred
    else:
        for sents, labels, mask_attention, token_type_ids, added_fts, _ in data_iter(lst_data, batch_size=1, is_shuffle=False):
            X = torch.LongTensor(sents).to(device)
            token_type_ids = torch.LongTensor(token_type_ids).to(device)
            mask_attention = torch.LongTensor(mask_attention).to(device)
            added_fts = torch.FloatTensor(added_fts).to(device)
            if added_fts[0][0] == 1:
                cum_pred = [[0, 0, 0, 0]]
                cum_pred = torch.LongTensor(cum_pred).to(device)
            # predict
            y_pred_ary = torch.exp(model(X, mask_attention, token_type_ids, added_fts, cum_pred)).detach().cpu().numpy()
            y_pred = np.argmax(y_pred_ary, axis=1)
            # add to cumulative predictions
            cum_pred[0][y_pred] += 1
            # append to list
            list_probPred.extend(y_pred_ary)
            list_yTrue.extend(labels)
            list_yPred.extend(y_pred)
        return f1_score(list_yTrue, list_yPred, average='macro'), list_yTrue, list_yPred, list_probPred

if __name__ == "__main__":
    args = get_args()
    # Pytorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open('tuning_model2.txt', 'a') as f:
        f.write(f"args: {vars(args)}\n")
    print("Device:", device)
    # set random seeds
    seed_everything(args.seed)
    # load and process data
    dataset = N2C2_track3_dataset(args)
    dataset.convert_to_list()
    # use ground truth in the 1st epoch
    for rid, sample in enumerate(dataset.train_lst):
        if rid+1 != len(dataset.train_lst) and rid+1 not in dataset.idx_start_train:
            dataset.train_lst[rid+1]['cum_pred'] = dataset.train_lst[rid]['cum_pred'].copy()
            dataset.train_lst[rid+1]['cum_pred'][sample['label']] += 1
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
    # training and evaluation
    best_dev = 0
    for ep in range(args.epoch):
        model.train()
        train_loss = 0
        num_batches = 0
        for sents, labels, mask_attention, token_type_ids, added_fts, cum_pred in data_iter(dataset.train_lst, batch_size=args.batch_size):
            # convert to tensors
            X = torch.LongTensor(sents).to(device)
            mask_attention = torch.LongTensor(mask_attention).to(device)
            token_type_ids = torch.LongTensor(token_type_ids).to(device)
            added_fts = torch.FloatTensor(added_fts).to(device)
            cum_pred = torch.LongTensor(cum_pred).to(device)
            y = torch.LongTensor(labels).to(device)
            # set the gradients to zero
            optimizer.zero_grad()
            # predict
            y_pred = model(X, mask_attention, token_type_ids, added_fts, cum_pred)
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
        train_f1, list_yTrue_train, list_yPred_train, list_probPred_train = model_eval(dataset.train_lst, model, data_iter)
        print(f"Epoch {ep}\nTraining loss: {train_loss}\tTraining F1: {train_f1}")
        # performance on dev set
        dev_f1, _, _, res_prob_dev = model_eval(dataset.dev_lst, model, data_iter, 'test')
        print(f"\nDev F1: {dev_f1}")
        # performance on dev set
        test_f1, _, _, res_prob_test = model_eval(dataset.test_lst, model, data_iter, 'test')
        print(f"\nTest F1: {test_f1}\n\n")

        # use cumulated predictions in the following epochs
        for rid, sample in enumerate(dataset.train_lst):
            if rid+1 != len(dataset.train_lst) and rid+1 not in dataset.idx_start_train:
                dataset.train_lst[rid+1]['cum_pred'] = dataset.train_lst[rid]['cum_pred'].copy()
                dataset.train_lst[rid+1]['cum_pred'][list_yPred_train[rid]] += 1

        if dev_f1 > best_dev:
            best_pred_dev, best_pred_test = res_prob_dev, res_prob_test

        with open('tuning_model2.txt', 'a') as f:
            f.write(f"\nDev F1: {dev_f1}")
            f.write(f"\nTest F1: {test_f1}\n\n")

np.save('pred_model_2_dev.npy', np.array(best_pred_dev))
np.save('pred_model_2_test.npy', np.array(best_pred_test))
torch.save(model.state_dict(), "model_2.pt")
