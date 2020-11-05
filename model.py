# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:41:40 2020

@author: 79072
"""
if __name__ =="__main__":
    import sys
    sys.path.append("..")
    from main import CombinationForecasting
    import argparse
    import time
    from cvxopt import matrix, solvers
    import pandas as pd
    import numpy as np
    ##---------------读取数据
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--input',required=True, help="输入数据文件的位置")
    ap.add_argument('-s',default='no',help='是否为诱导模型')
    ap.add_argument('-o','--output',default='output/result.txt', help="输出数据文件的位置")
    args = ap.parse_args()
    path = args.input
    data = pd.read_excel(path)
    ### 传入的数据中，e*列判断为error,a*列判断为诱导值列（一般是精度）
    errorCol = []
    for i in data.columns[1:]:
        if 'e' in i:
            errorCol.append(i)
    induceCol = []
    for i in data.columns[1:]:
        if 'a' in i:
            induceCol.append(i)
    error = data[errorCol]
    induceData = data[induceCol]
    ###---------------求解四种模型    
    types = ['mse','abse','bias','range']
    dic = {'mse':'误差平方和最小进行优化',
           'abse':'绝对误差和最小进行优化',
           'bias':'最大偏差最小进行优化',
           'range':'误差极差最小进行优化'}
    cf = CombinationForecasting()
    if args.s == 'no':
        with open(args.output, mode = 'w+') as fp:
            for t in types:
                res = cf.run(error, errorType = t)
                w = res["w"]
                Nowtime = time.strftime('%Y-%m-%d  %H:%M:%S',time.localtime(time.time()))
                fp.write(f'-----------{Nowtime}----------\n')
                fp.write('------------普通模型-----------\n')
                fp.write(f'-----------{dic[t]}----------\n')
                print(f'权重为:{w}')
                fp.write(f'权重为:{w}\n')
                print(f'最优值为:{res["primal_objective"]}')
                fp.write(f'最优值为:{res["primal_objective"]}\n')
                print(f'误差信息矩阵为:\n{res["errorMatrix"]}')
                fp.write(f'误差信息矩阵为:\n{res["errorMatrix"]}\n')

    else:
        result = cf.induce(error, induceData)
        with open(args.output, mode = 'w+') as fp:
            for t in types:
                res = cf.run(result, errorType = t)
                w = res["w"]
                Nowtime = time.strftime('%Y-%m-%d  %H:%M:%S',time.localtime(time.time()))
                fp.write(f'-----------{Nowtime}----------\n')
                fp.write('------------诱导模型-----------\n')
                fp.write(f'-----------{dic[t]}----------\n')
                print(f'权重为:{w}')
                fp.write(f'权重为:{w}\n')
                print(f'最优值为:{res["primal_objective"]}')
                fp.write(f'最优值为:{res["primal_objective"]}\n')
                print(f'误差信息矩阵为:\n{res["errorMatrix"]}')
                fp.write(f'误差信息矩阵为:\n{res["errorMatrix"]}\n')
                print(f'权重矩阵为:\n{cf.restoreW(w, error, induceData)}')
                fp.write(f'权重矩阵为:\n{cf.restoreW(w, error, induceData)}\n')
    