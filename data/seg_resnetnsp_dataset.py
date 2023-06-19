import os
import json
import pickle
import logging
import copy
import random
from itertools import chain
from tqdm import tqdm

import numpy as np

import torch
import torch.utils.data
from torch.utils.data import Dataset

def tokenize(obj,tokenizer):
    if isinstance(obj, str): # 对 string 格式的文本 tokenize
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict): # 对字典格式的文本 tokenize -> key:tokenized value
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj) # 其他情况

def get_dataset(tokenizer, data_file):
    dialog_data = json.load(open(data_file, 'r'))
    dialog_list = []
    vid_set = set()
    for vid, dialogs in tqdm(dialog_data.items(), desc='Process Data'): # dict {}
        dialog = [tokenize(d, tokenizer) for d in dialogs['dialog']] # [[dialog i id list], [], ...]
        vid_set.add(vid) # vid set
        scene = dialogs['scene']
        session = dialogs['session']
        item = {'vid':vid, 'dialog':dialog, 'scene':scene, 'session':session}
        assert len(dialog) == len(session)
        assert len(dialog) == len(scene)
        dialog_list.append(item)
    return dialog_list


class DataSet(Dataset):
    def __init__(self, dialogs, tokenizer, features=None):
        self.dialogs = dialogs # dialog_list
        self.features = features # all_feature
        self.tokenizer = tokenizer
        

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, index):
        vid = self.dialogs[index]['vid'] # 'vid'
        dialog_lst = self.dialogs[index]['dialog'] # [[u_1],[u_2],...[u_n]] 
        scene_lst = self.dialogs[index]['scene'] # [1, 0, ...]
        session_lst = self.dialogs[index]['session'] # [1, 0, ...]
        assert len(session_lst) == len(dialog_lst)
        
        if self.features is None:
            cls, sep = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
            sequence = []
            sequence_type = []
            session_label = []
            session_index = []
            seg_cnt = 0
            ses_wd_sz = 4
            vid_lst = []
            utter_lst = []
            # cal. max utterance length
            max_dialog_length = (500-len(dialog_lst)-1) // (ses_wd_sz*2+2)       
            for i in range(len(dialog_lst)):
                utter_idx_lst = i + np.arange(-ses_wd_sz, ses_wd_sz+1)
                utter_idx_lst = np.clip(utter_idx_lst, 0, len(dialog_lst)-1)
                tmp_sequence = [cls]
                tmp_sequence_type = [0]
                tmp_index = []
                tmp_cnt = 1
                for ii in utter_idx_lst:
                    tmp_sequence += dialog_lst[ii][:max_dialog_length] + [sep]
                    tmp_sequence_type += [ii%2] * (len(dialog_lst[ii][:max_dialog_length])+1)
                    tmp_index.append(tmp_cnt+(len(dialog_lst[ii][:max_dialog_length])+1))
                    tmp_cnt += (len(dialog_lst[ii][:max_dialog_length])+1)
                    
                sequence.append(torch.Tensor(tmp_sequence).long())
                sequence_type.append(torch.Tensor(tmp_sequence_type).long())
                
                # cid_start = sum([ii for ii in tmp_index[:8]])
                session_index.append(torch.Tensor([ii for ii in range(tmp_index[ses_wd_sz-1], tmp_index[ses_wd_sz])]).long())
                session_label += [session_lst[i]]
                vid_lst.append(vid)
                utter_lst.append(i)

            dialog = sequence
            dialog_type = sequence_type
            session_label = session_label
            session_index = session_index
            assert len(session_label) == len(dialog)

            feature = None
            feature_type = None
            scene_label = None
            scene_index = None
    
        else:
            clip_feature = np.load('inputs/features/resnet/{}/{}.npy'.format(vid.split('_clip')[0], vid))
            clip_feature_cnt = self.features[vid]
            feature = []
            feature_type = []
            scene_label = []
            scene_index = []
            seg_cnt = 0
            fea_wd_sz = 4

            clip_index = []
            start = 0
            max_feature_length = (500 - len(clip_feature_cnt)-1) // (fea_wd_sz * 2 + 1)
            for i in range(len(clip_feature_cnt)):
                # # sample 3 frames for each utterance
                # if clip_feature_cnt[i] < 3:
                #     sample_index = [start, start, start]
                # else:
                #     sample_index = [start, start+clip_feature_cnt[i]//2, start+clip_feature_cnt[i]-1]
                # sample 1
                if clip_feature_cnt[i] < 3:
                    sample_index = [start]
                else:
                    sample_index = [start+clip_feature_cnt[i]//2]
                clip_index.append(sample_index)
                start += clip_feature_cnt[i]

            vid_lst = []
            utter_lst = []
            start = 0
            feature_pt = torch.from_numpy(clip_feature)
            for i in range(len(clip_feature_cnt)):
                utter_idx_lst = i + np.arange(-fea_wd_sz, fea_wd_sz+1)
                utter_idx_lst = np.clip(utter_idx_lst, 0, len(clip_feature_cnt)-1)
                final_index_lst = []
                type_index_lst = []
                for ii in utter_idx_lst:
                    final_index_lst += clip_index[ii]
                    type_index_lst.append(clip_index[ii])
                feature += feature_pt[final_index_lst].unsqueeze(0)
                scene_label  += [scene_lst[i]]
                fea_type = []
                for ii in range(len(type_index_lst)):
                    fea_type += [ii%2]*len(type_index_lst[ii])
                feature_type.append(fea_type)
                cid_start = sum([len(ii) for ii in type_index_lst[:fea_wd_sz]])
                scene_index.append(torch.Tensor([ii for ii in range(cid_start, cid_start+len(type_index_lst[fea_wd_sz]))]).long())
                vid_lst.append(vid)
                utter_lst.append(i)

            assert len(scene_label) == len(feature)
            dialog = None
            dialog_type = None
            session_label = None
            session_index = None
            
        return dialog, dialog_type, session_label, session_index,\
             feature, feature_type, scene_label, scene_index, utter_lst, vid_lst


def collate_fn(batch, pad_token, features=None):

    def padding(seq, pad_token, limit=510, embed=False):
        max_len = max([i.size(0) for i in seq])
        assert max_len < limit
        if embed:
            result = torch.ones((len(seq), max_len, seq[0].size(-1))).float() * pad_token
        else:
            result = torch.ones((len(seq), max_len)).long() * pad_token
        for i in range(len(seq)):
            result[i, :seq[i].size(0)] = seq[i][-min(limit, seq[i].size(0)):]
        return result

    dialog_lst, dialog_type_lst, session_label_lst, session_index_lst = [], [], [], []
    feature_lst, feature_type_lst, scene_label_lst, scene_index_lst, utter_lst, vid_lst = [], [], [], [], [], []

    if features is None:
        for i in batch:
            dialog_lst += i[0]
            dialog_type_lst += i[1]
            session_label_lst += i[2]
            session_index_lst += i[3]
            utter_lst += i[8]
            vid_lst += i[9]
        dialog_ids = padding(dialog_lst, pad_token)
        dialog_type_ids = padding(dialog_type_lst, pad_token)
        session_label_ids = torch.Tensor(session_label_lst).long()
        dialog_mask = dialog_ids != pad_token
        feature_ids = None,
        feature_type_ids = None,
        scene_label_ids = None,
        feature_mask = None
    else:
        for i in batch:
            feature_lst += i[4]
            feature_type_lst += i[5]
            scene_label_lst += i[6]
            scene_index_lst += i[7]
            utter_lst += i[8]
            vid_lst += i[9]
        feature_ids = padding(feature_lst, 0, embed=True)
        feature_type_ids = padding([torch.Tensor(fea).long() for fea in feature_type_lst], pad_token)
        scene_label_ids = torch.Tensor(scene_label_lst).long()
        feature_mask = torch.sum(feature_ids, dim=2) != 0
        
        dialog_ids = None
        dialog_type_ids = None
        session_label_ids = None
        dialog_mask = None
        
    return dialog_ids, dialog_type_ids, dialog_mask, session_label_ids, session_index_lst,\
        feature_ids, feature_type_ids, feature_mask, scene_label_ids, scene_index_lst, utter_lst, vid_lst

