# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 09:55:01 2020

@author: 79072
"""
from cvxopt import matrix, solvers
import pandas as pd
import numpy as np

class CombinationForecasting():
    '''
    组合预测模型
        普通组合预测：
            1.run() -->得到权重和最优值
            2.组合预测
        诱导组合预测：
            1.induce()  -->得到诱导矩阵
            2.run()  -->得到权重和最优值
            3.restoreW() -->得到还原后的权重矩阵
            4.对还原后的权重矩阵进行处理，得到用于预测的权重
            5.组合预测
    '''
    def induce(self, error, induceData):
        '''
        通过诱导值对误差值进行排序，请将两组数据顺序对应
        error --> result
        Args:
            error: 误差值
            induceData: 诱导值
        '''
        T = induceData.shape[0]
        Nc = induceData.shape[1]
        s1 = np.argsort(-induceData)
        result = []
        for i in range(T):
            eCol = error.iloc[i, s1.iloc[i, :]]
            result.append(eCol.tolist())

        return pd.DataFrame(result)
    
    def restoreW(self,w,result, induceData):
        '''
        诱导之后的权重矩阵(用于结果展示)
        result --> error
        '''
        T = induceData.shape[0]
        Nc = induceData.shape[1]
        s1 = np.argsort(-induceData)
        s2 = np.argsort(s1)
        Wmatrix = []
        for i in range(T):
            wCol = w[s2.iloc[i, :]]
            Wmatrix.append(wCol.tolist())
        Wmatrix = pd.DataFrame(Wmatrix,columns = induceData.columns)
        return Wmatrix
        
    def run(self,data, Nonnegative = True,errorType = 'mse',rounds = 4):
            '''
            组合预测model
            Args:
                data: 传入的误差数据，[e1, e2,..]
                Nonnegative: 是否有非负约束，默认有,只对mse优化模型有效
                errorType:
                ['mse','abse','bias','range']
                    mse:误差平方和最小
                    abse:绝对误差和最小
                    bias:最大偏差最小
                    range:误差极差最小
                rounds:保留小数的位数，默认4位
            return:
                返回各个预测方法的权重，w = [w1, w2, w3, ....]
            '''
            T = data.shape[0]
            Nc = data.shape[1]
            total = T * 2 + Nc
            errorMatrix = np.dot(np.transpose(data),data)
            if errorType == 'mse':
                ## 误差平方和最小
                ## 误差信息矩阵E
                E = np.dot(np.transpose(data),data)
                ## 优化目标
                Q = 2*matrix(np.array(E.tolist()))
                p = matrix([0.0 for i in range(Nc)])
                ## 非负约束
                G = matrix(np.negative(np.identity(Nc)).tolist())
                h = p
                ## 和为1
                A = matrix([1.0 for i in range(Nc)],(1, Nc))
                b = matrix(1.0)
                print('********误差平方和最小进行优化************')

                if Nonnegative==False:
                    sol=solvers.qp(Q, p,A=A,b = b)
                else:
                    sol=solvers.qp(Q, p,G=G,h=h,A=A,b= b)
                w = sol['x']
                primal_objective = sol['primal objective']

            elif errorType=='abse':
                ### 绝对误差和最小
                ## 优化目标
                c = matrix([1.0 for i in range(2*T)] + [0.0 for i in range(Nc)])
                ## 限制条件
                A1 = np.concatenate((np.negative(np.identity(T)),np.identity(T),data),axis = 1).tolist()
                A2 = [0.0 for i in range(2*T)] + [1.0 for i in range(Nc)]
                A1.append(A2)
                A = np.transpose(np.array(A1))
                A = matrix(A.tolist())
                b = matrix([0.0 for i in range(T)] + [1])
                ## 非负约束
                G = matrix(np.negative(np.identity(total)).tolist())
                h = matrix([0.0 for i in range(total)])
                print('**************绝对误差和最小进行优化********************')
                sol = solvers.lp(c,G, h, A, b)
                w = sol['x'][-Nc:]
                primal_objective = sol['primal objective']

            elif errorType == 'bias':
                ## 优化目标
                c = matrix([0.0 for i in range(total)] + [1.0])
                ## 限制条件
                A1 = np.concatenate((np.negative(np.identity(T)),np.identity(T),data,np.zeros(shape=(T,1))),axis = 1).tolist()
                A2 = [0.0 for i in range(2*T)] + [1.0 for i in range(Nc)] + [0]
                A1.append(A2)
                A = np.transpose(np.array(A1))
                A = matrix(A.tolist())
                b = matrix([0.0 for i in range(T)] + [1])
                ## 非负约束
                G1 = np.negative(np.identity(total + 1))
                # print(G1)
                G2 = np.concatenate((np.identity(T),np.identity(T),np.zeros(shape=(T,Nc)),np.negative(np.ones(shape=(T,1)))),axis = 1)
                G = np.transpose(np.concatenate((G1, G2)))
                G = matrix(G.tolist())
                h = matrix([0.0 for i in range(T +total + 1)])
                print('********************最大偏差最小进行优化************************')

                sol = solvers.lp(c,G, h, A, b)
                w = sol['x'][-Nc - 1: -1]
                primal_objective = sol['primal objective']

            elif errorType == 'range':
                ## 优化目标
                c = matrix([0.0 for i in range(total)] + [1.0, -1.0])
                ## 限制条件
                A = matrix([0.0 for i in range(2*T)]+[1.0 for i in range(Nc)] + [0.0,0.0],(1,total+2))
                b = matrix([1.0])
                ## 非负约束
                G1 = np.concatenate((np.negative(np.identity(total)),np.zeros(shape = (total, 2))),axis = 1)
                # print(G1)
                G2 = np.concatenate((np.zeros(shape = (T,2*T)),data,np.negative(np.ones(shape=(T,1))),np.zeros(shape=(T,1))),axis = 1)
                G3 = np.concatenate((np.zeros(shape = (T,2*T)),np.negative(data), np.zeros(shape=(T,1)),np.ones(shape=(T,1))),axis = 1)
                G = np.transpose(np.concatenate((G1, G2, G3)))
                G = matrix(G.tolist(),(2*T +total,total + 2))
                h = matrix([0.0 for i in range(2*T +total)])
                print('**************误差极差最小进行优化*********************')

                sol = solvers.lp(c,G, h, A, b)
                w = sol['x'][-Nc-2:-2]
                primal_objective = sol['primal objective']
                
            else:
                print("无该方法")
                
            return {'w':np.round(list(w), rounds),'primal_objective':round(primal_objective,rounds),'errorMatrix':errorMatrix}
        
if __name__ =="__main__":
    import argparse
    import time
    from cvxopt import matrix, solvers
    import pandas as pd
    import numpy as np
    ##---------------读取数据
    data = pd.read_excel(r'example/example2.xlsx')
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
    cf = CombinationForecasting()
    result = cf.induce(error, induceData)
    for t in types:
        res = cf.run(result, errorType = t)
        w = res["w"]
        print(f'权重为:{w}')
        print(f'最优值为:{res["primal_objective"]}')
        print(f'误差信息矩阵为:\n{res["errorMatrix"]}')
        print(f'权重矩阵为:\n{cf.restoreW(w, error, induceData)}')
  
    
    
    
    