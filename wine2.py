#!/usr/bin/env python
# coding: utf-8

# ## 吴楠楠  3220190895
# ## 数据集：wine
# ### 导入必须的包

# In[1]:


import numpy as np
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
import warnings
from collections import Counter
from math import isnan


# ### 1 处理数据
# #### 1.1 读取数据集并用最高频率填补缺失值。（该部分已在作业一中完成，因此直接读取填补后的csv文件）。
# #### 1.2 该数据集中price和points属于数值属性，因此需要分段。以五数概括为标准，将price分为high_price、above_price、below_price和low_price，离群点的数据则设置为so_high_price。同理，将points分为high_point、above_point、below_point和low_point。

# In[2]:


warnings.filterwarnings("ignore")
fill_df = pd.read_csv("D:/python/data/result.csv",index_col=0)
length = len(fill_df)
col = 'price'
with open("D:/python/data/result.csv", 'r',encoding='UTF-8') as file:
    next(file)
    reader = csv.reader(file)
    num = [row[5] for row in reader]
num = list(map(float, num))
for i in range(length):
    if num[i] > 71:
        fill_df[col][i] = 'so_high_price'
    elif num[i] > 38:
        fill_df[col][i] = 'high_price' 
    elif num[i] > 22:
        fill_df[col][i] = 'above_price'
    elif num[i] >16:
        fill_df[col][i] = 'below_price'
    else:
        fill_df[col][i] = 'low_price' 
fill_df.head()


# In[3]:


with open("D:/python/data/result.csv", 'r',encoding='UTF-8') as file:
    next(file)
    reader = csv.reader(file)
    num = [row[4] for row in reader]
num = list(map(float, num))
col = 'points'
for i in range(length):
    if num[i] > 90:
        fill_df[col][i] = 'high_point'
    elif num[i] > 88:
         fill_df[col][i] = 'above_point' 
    elif num[i] > 86:
        fill_df[col][i] = 'below_point'
    else:
        fill_df[col][i] = 'low_point'
fill_df.head()


# #### 1.3计算可得填补后的数据缺失值为0。

# In[4]:


def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))
missing_data(fill_df)


# #### 1.4 导入数据集，转换成适合进行关联规则挖掘的形式。删除掉一些不感兴趣的列，选取的属性列为country、designation、points、price和province。

# In[5]:


def load_data_set():
    census = fill_df
    length = len(census)
    data = []
    for i in range(length):
        data.append([census.loc[i]['country'],
                     census.loc[i]['designation'],
                     census.loc[i]['points'],
                     census.loc[i]['price'],
                     census.loc[i]['province']
                    ])
    data_set = {}
    for transaction in data:
        if frozenset(transaction) in data_set:
            data_set[frozenset(transaction)] += 1
        else:
            data_set[frozenset(transaction)] = 1
    return data_set


# ### 2 挖掘关联规则
# #### 2.1 找出频繁模式，设置支持度阈值为0.1。
# #### 2.2 导出关联规则以及其支持度和置信度，设置置信度阈值为0.3。
# #### 2.3 关联规则评价，采用Lift和Jaccard两种指标。

# In[6]:


class FPTree:
    def __init__(self, name, count, parent):
        self.name = name
        self.count = count
        self.link = None
        self.parent = parent
        self.children = {}

    def increase(self, count):
        self.count += count

    def dispaly(self, tab = 1):
        print('    ' * tab, self.name, ':', self.count)
        for child in self.children.values():
            child.dispaly(tab + 1)

def create_tree(data_set, min_sup):
    header_table = {}
    for transaction in data_set:
        for item in transaction:
            header_table[item] = header_table.get(item, 0) + data_set[transaction]
    for item in list(header_table):
        if header_table[item] < min_sup:
            del header_table[item]
    sup_item_set = set(header_table.keys())
    if len(sup_item_set) == 0:
        return None, None
    for item in header_table:
        header_table[item] = [header_table[item], None]
    fp_tree = FPTree('NULL', 0, None)
    for transaction, count in data_set.items():
        local_transaction = {}
        for item in transaction:
            if item in sup_item_set:
                local_transaction[item] = header_table[item][0]
        if len(local_transaction) > 0:
            ordered_items = [a[0] for a in sorted(local_transaction.items(),key=lambda b: b[1], reverse=True)]
            update_tree(ordered_items, count, fp_tree, header_table)
    return fp_tree, header_table

def update_tree(items, count, fp_tree, header_table):
    if items[0] in fp_tree.children:
        fp_tree.children[items[0]].increase(count)
    else:
        fp_tree.children[items[0]] = FPTree(items[0], count, fp_tree)
        if header_table[items[0]][1] is None:
            header_table[items[0]][1]  = fp_tree.children[items[0]]
        else:
            update_header(header_table[items[0]][1],fp_tree.children[items[0]])
    if len(items) > 1:
        update_tree(items[1:], count, fp_tree.children[items[0]], header_table)

def update_header(old_node, new_node):
    while old_node.link is not None:
        old_node = old_node.link
    old_node.link = new_node

def ascent_tree(leaf_node, prefix_path):
    if leaf_node.parent is not None:
        prefix_path.append(leaf_node.name)
        ascent_tree(leaf_node.parent, prefix_path)

def find_conditional_basis(tree_node):
    conditional_basis = {}
    sup = 0
    while tree_node is not None:
        prefix_path = []
        ascent_tree(tree_node, prefix_path)
        if len(prefix_path) > 1:
            conditional_basis[frozenset(prefix_path[1:])] = tree_node.count
        sup += tree_node.count
        tree_node = tree_node.link
    return conditional_basis, sup

def mine_tree(header_table, min_sup, l, l_list, sup_dict):
    ordered_header_table = [a[0] for a in sorted(header_table.items(),key=lambda b: str(b[1]))]
    for item in ordered_header_table:
        local_l = l.copy()
        local_l.add(item)
        l_list.append(frozenset(local_l))
        conditional_basis, sup = find_conditional_basis(header_table[item][1])
        sup_dict[frozenset(local_l)] = sup
        conditional_tree, conditional_header = create_tree(conditional_basis, min_sup)
        if conditional_header is not None:
            mine_tree(conditional_header, min_sup, local_l, l_list, sup_dict)

def apriori_gen(lk_1, k):
    ck = []
    for i in range(len(lk_1)):
        for j in range(i + 1, len(lk_1)):
            l1 = list(lk_1[i])[:k - 2]
            l2 = list(lk_1[j])[:k - 2]
            l1.sort()
            l2.sort()
            if l1 == l2:
                ck.append(lk_1[i] | lk_1[j])
    return ck

def generate_rules(l_list, sup_dict, min_conf,len1):
    rules_list = []
    for j in range(len(l_list)):
        item_list = [frozenset([item]) for item in l_list[j]]
        for i in range(1, len(item_list)):
            for left in item_list:
                conf = sup_dict[l_list[j]] / sup_dict[left]
                sup = sup_dict[l_list[j]] / len1
                sup1 = sup_dict[l_list[j] - left] / len1
                Lift = conf/sup1
                Jacard = sup_dict[l_list[j]] / (sup_dict[l_list[j] - left]+sup_dict[left]-sup_dict[l_list[j]])
                if conf >= min_conf:
                    rules_list.append((set(left), set(l_list[j] - left), round(conf, 3),round(sup,3)))
                    print('关联规则：{} ----->{}， 置信度：{}，支持度：{}'.format(set(left), set(l_list[j] - left), round(conf, 3),round(sup,3)))
                    print('规则评价：lift系数:{},  Jacard系数:{}'.format(round(Lift,3),round(Jacard,3)))
            if i + 1 < len(item_list):
                item_list = apriori_gen(item_list, i + 1)
    return rules_list

def main():
    data_set = load_data_set()
    len1 = len(fill_df)
    min_support = 0.1
    min_confidence = 0.3
    num_t = sum(data_set.values())
    fp_tree, header_table = create_tree(data_set, min_support*num_t)
    l_list = []
    sup_dict = {}
    mine_tree(header_table, min_support*num_t, set([]), l_list, sup_dict)
    for i in range(len(l_list)):
        print('频繁项集：{}， 支持度：{}'.format(l_list[i],round(sup_dict[l_list[i]]/len1,3)))
    print('频繁项集数目：{}'.format(len(l_list)))
    print()
    rules_list = generate_rules(l_list, sup_dict, min_confidence,len1)
    print('关联规则数目：{}'.format(len(rules_list)))

def function(min_suppot, min_confidence):
    data_set = load_data_set()
    num_t = sum(data_set.values())
    fp_tree, header_table = create_tree(data_set, min_support*num_t)
    l_list = []
    sup_dict = {}
    mine_tree(header_table, min_support*num_t, set([]), l_list, sup_dict)
    rules_list = generate_rules(l_list, sup_dict, min_confidence)
    return len(l_list), len(rules_list)

if __name__ == '__main__':
    main() 


# ### 3 挖掘结果分析
# #### 共挖掘出关联规则28条，删除一些无意义的规则，选取5条评价分数以及置信度较高的进行分析。
# #### 3.1 {'low_price'} ----->{'low_point'}，从这条可以看出较低价格的葡萄酒，评分一般较低。
# #### 3.2 {'above_price'} ----->{'US'}，价格中等偏上的葡萄酒大概率产自美国。
# #### 3.3 {'US', 'California'} ----->{'Reserve'}，来自美国加利福尼亚州的葡萄酒，大多会标注Reserve（珍藏）。
# #### 3.4 {'US'} ----->{'California'}，美国葡萄酒的主产地位于加利福尼亚州。
# #### 3.5 {'California'} ----->{'low_point'}，加利福尼亚州产出的葡萄酒大多分数较低。
