import numpy as np
import pandas as pd
import copy
import math

def sortnumber(D):
    numbse=numbve=numbvi=0          #初始化
    for i in range(len(D)):
        if D[i][4] == "Iris-setosa":
            numbse = numbse + 1
        elif D[i][4] == "Iris-versicolor":
            numbve = numbve + 1
        elif D[i][4] == "Iris-virginica":
            numbvi = numbvi + 1
    numb=[numbse,numbve,numbvi]
    return numb

def find_minimal_index(D):
    min_elem=D[0]
    count=0
    min_index=count
    for elem in D[1:]:
        count+=1
        if elem<min_elem:
            elem,min_elem=min_elem,elem
            min_index=count
    return min_index

def sortD(D,X):
    for i in range(len(D)):
        index=find_minimal_index(D[i:,X]) #找到未排序的最小值的索引
        m=copy.deepcopy(D[index+i])  #深拷贝
        D[index+i]=D[i]           #与最小值交换
        D[i]=m
    return D


def ENAttribute(D, X, allc):
    d = copy.deepcopy(D)     #深拷贝
    d=sortD(d, X)        #按X列属性排序
    M = []     # 存放所有分割点的集合
    N = []     # 存放每个分割点DN的样本数据
    nse = nve = nvi = 0     #初始化
    for i in range(len(D) - 1):
        if d[i][4] == allc[0]:
            nse += 1
        elif d[i][4] == allc[1]:
            nve += 1
        elif d[i][4] == allc[2]:
            nvi += 1
        if d[i + 1][X] != d[i][X]:
            v = (d[i + 1][X] + d[i][X]) / 2  #取切分点
            M.append(v)        #存入列表
            Nv = [nse, nve, nvi]
            N.append(Nv)
    if d[-1][4] == allc[0]:
        nse += 1
    elif d[-1][4] == allc[1]:
        nve += 1
    elif d[-1][4] == allc[2]:
        nvi += 1
    n = [nse, nve, nvi]    #样本每个标签个数
    HD=0    #划分前样本的熵
    for i in range(len(n)):
        if n[i]!=0:
            HD=HD-(n[i]/len(d))*math.log(n[i]/len(d),2)
    bestv = 0
    score = 0
    sum = []   #存放每个分割点DY数据总数
    sumn=[]   #存放每个分割点DN数据总数
    for i in range(len(N)):
        s = 0
        for j in range(len(allc)):
            s = s + N[i][j]
        sum.append(s)
    for i in range(len(N)):
        sumo = len(d)- sum[i]
        sumn.append(sumo)
    for i in range(len(M)):
        Py = []     #每一类在DY中的概率
        Pn = []    #每一类在DN中的概率
        Hy = 0     #DY子集的熵
        Hn = 0     #DN子集的熵
        for j in range(len(allc)):
            Pcy = N[i][j] / sum[i]
            Pcn = (n[j] - N[i][j]) / sumn[i]
            Py.append(Pcy)
            Pn.append(Pcn)
        for k in range(len(Py)):
            if Py[k]!=0:
                Hy = Hy - Py[k] * math.log(Py[k], 2)
            if Pn[k]!=0:
                Hn = Hn - Pn[k] * math.log(Pn[k], 2)
        H = (sum[i] / len(d)) * Hy + (sumn[i] / len(d)) * Hn #分裂点的熵
        score1=HD-H      #信息增益
        if score1 > score:
            score = score1
            bestv = M[i]
    return bestv, score

jiedian = 0
#retujiedian=0
def decisiontree(D, minp, pure):
    allc=['Iris-setosa','Iris-versicolor','Iris-virginica']  #标签列表
    n=len(D)                #样本大小
    num=sortnumber(D)        #标签个数
    pu=max(num)/n            #样本纯度

    if n<=minp or pu>=pure:       #终止条件
        i=np.argmax(num/(n*np.ones((1,len(num)))))   #获取最多标签的下标
        c=allc[i]                    #结点标签
        print("叶子结点，其标签为："+str(c)+",纯度为："+str(pu)+",大小为："+str(n))
        return
    else:
        global jiedian         #结点序号
        jiedian = jiedian + 1
    scoreinit=0              #最大的信息增益
    attribute=0              #记录最佳分割点的属性
    bestv=0                  #最佳分割点
    for i in range(D.shape[1]-1):
        v,score=ENAttribute(D,i,allc)   #找某属性最佳分割点和其信息增益
        if score>scoreinit:
            splitpinit = []     #DY
            other = []          #DN
            scoreinit=score
            attribute=i+1
            bestv=v
            for j in range(len(D)):
                if D[j][i]<=bestv:      #根据分割点划分数据
                    splitpinit.append(D[j])
                else:
                    other.append(D[j])
    jb=copy.deepcopy(jiedian)         #存当前结点序号
    print("第"+str(jb)+"个结点：以“属性" +str(attribute)+"<=" + str(bestv)+" ”作为决策,信息增益为："+str(scoreinit))
    DY=np.array(splitpinit)          #变为数组形式
    DN=np.array(other)
    print("结点"+str(jb)+"的左子树为：")
    decisiontree(DY,minp,pure)        #递归
    print("结点"+str(jb)+"的右子树为：")
    decisiontree(DN,minp,pure)

D=pd.read_csv('C:/Users\XJW\Desktop\iris.txt',header=None)  #读取数据
D=np.array(D)
decisiontree(D,5,0.95)
