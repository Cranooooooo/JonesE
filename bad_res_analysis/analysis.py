#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import json


def rela_statis(res):
    bad_rela_count = {}
    for key in res.keys():
        bad_rela = res[key][-1][1]
        if bad_rela not in bad_rela_count.keys(): bad_rela_count[bad_rela] = 1
        else: bad_rela_count[bad_rela] += 1

    for item in bad_rela_count.items():
        print(item)

def head_statis(res):
    bad_head_count = {}
    for key in res.keys():
        bad_head = res[key][-1][0]
        if bad_head not in bad_head_count.keys(): bad_head_count[bad_head] = 1
        else: bad_head_count[bad_head] += 1

    # for item in bad_head_count.items():
    print(max(bad_head_count.values()))

def tail_statis(res):
    bad_tail_count = {}
    for key in res.keys():
        bad_tail = res[key][-1][-1]
        if bad_tail not in bad_tail_count.keys(): bad_tail_count[bad_tail] = 1
        else: bad_tail_count[bad_tail] += 1

    # for item in bad_tail_count.items():
    print(max(bad_tail_count.values()))


if __name__ == '__main__':
    for model_name in ['DualE', 'JonesE_only_X']:
        print(f'\n ========== {model_name} ========== ')
        # 0. Set Parameters
        res_file = f'./{model_name}_WN18_Valid.json'
        with open(res_file, 'r') as f: res = json.load(f)

        # 1. Bad Rela Statistic
        rela_statis(res)
        head_statis(res)
        tail_statis(res)