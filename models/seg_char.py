#!/usr/bin/python
#coding = utf-8
import sys
import re
special_char={' ', '\t', '\n','',''}


def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
        return True
    else:
        return False


def seg_char(line):
    line = line.decode('utf-8')
    seg = ""
    seg_list = []
    for j in range(len(line)):
        if line[j] in special_char:
            if len(seg):
                seg_list.append(seg)
            seg = ""
            continue;
        if not is_chinese(line[j]):
            seg = seg+line[j]
            continue;
        if len(seg):
            seg_list.append(seg)
            seg = ""
        seg_list.append(line[j])
    if len(seg):
        seg_list.append(seg)
    return seg_list

