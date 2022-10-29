#!/usr/bin/env python 
# -*- coding:utf-8 -*-
'''40942'''

import json
import os

def step_1_process_orig_file(origin_file_path):
    ent_idx_list = []
    rela_idx_list = []

    # Load file
    with open(origin_file_path, 'r') as f:
        triplet_list = f.read().split('\n')
        triplet_num = int(triplet_list[0])
        triplet_list = triplet_list[1:-1]
        assert triplet_num == len(triplet_list)

    # Rectify them
    rectified_list = []
    for item_idx in range(triplet_num):
        this_triplet = triplet_list[item_idx]
        head_idx, tail_idx, rela_idx = this_triplet.split(' ')
        head_idx = int(head_idx)
        tail_idx=int(tail_idx)
        rela_idx = int(rela_idx)

        ent_idx_list.append(head_idx); ent_idx_list.append(tail_idx)
        rela_idx_list.append(rela_idx)
        rectified_list.append([head_idx, tail_idx, rela_idx])

    # Statistic
    print(f'Entity Idx: Max:{max(ent_idx_list)} / Min:{min(ent_idx_list)}')
    print(f'Relation Idx: Max:{max(rela_idx_list)} / Min:{min(rela_idx_list)}')

    return rectified_list


def step_2_process_text_file(text_file_path):
    # Load file
    with open(text_file_path, 'r') as f:
        triplet_list = f.read().split('\n')
        triplet_list = triplet_list[:-1]

    # Rectify them
    rectified_list = []
    for item_idx in range(len(triplet_list)):
        this_triplet = triplet_list[item_idx]
        head_text, tail_text, rela_text = this_triplet.split('\t')
        rectified_list.append([head_text, tail_text, rela_text])

    return rectified_list


if __name__ == '__main__':
    dataset_name = 'WN18'
    dataset_path = f'/home/shaoqian/workplace_cl/datasets_knowledge_embedding/{dataset_name}/'
    origin_path = dataset_path+'original/'
    text_path = dataset_path+'text/'

    # Process Original file
    train_orig = origin_path + 'train.txt'
    valid_orig = origin_path + 'valid.txt'
    test_orig = origin_path + 'test.txt'

    orin_result_list = []
    for file in [train_orig, valid_orig, test_orig]:
        orin_result_list.append(step_1_process_orig_file(file))

    # Process Text file
    train_text = text_path + 'train.txt'
    valid_text = text_path + 'valid.txt'
    test_text = text_path + 'test.txt'

    text_result_list = []
    for file in [train_text, valid_text, test_text]:
        text_result_list.append(step_2_process_text_file(file))


    # Unify all the file into a json file, for showing text in JonesE
    train_orin = orin_result_list[0]
    train_text = text_result_list[0]
    train_orin_text_dict = {}
    for count in range(len(train_orin)):
        train_orin_text_dict[str(train_orin[count])] = train_text[count]
    with open(f'./benchmarks/{dataset_name}/train_text_cl.json', 'w') as f:
        json.dump(train_orin_text_dict, f)

    valid_orin = orin_result_list[1]
    valid_text = text_result_list[1]
    valid_orin_text_dict = {}
    for count in range(len(valid_orin)):
        valid_orin_text_dict[str(valid_orin[count])] = valid_text[count]
    with open(f'./benchmarks/{dataset_name}/valid_text_cl.json', 'w') as f:
        json.dump(valid_orin_text_dict, f)

    test_orin = orin_result_list[2]
    test_text = text_result_list[2]
    test_orin_text_dict = {}
    for count in range(len(test_orin)):
        test_orin_text_dict[str(test_orin[count])] = test_text[count]
    with open(f'./benchmarks/{dataset_name}/test_text_cl.json', 'w') as f:
        json.dump(test_orin_text_dict, f)



