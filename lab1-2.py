import numpy as np
import pandas as pd

D=pd.read_csv('C:/Users\XJW\Desktop\iris.txt',header=None)
D=np.array(D)
data=D[:,0:4]
#print(data)
k=[]
for i in range(len(data)):
    rowk=[]
    for j in range(len(data)):
        tempk=np.dot(data[i],data[j])**2
        rowk.append(tempk)
    k.append(rowk)
k=np.array(k)
print("齐次二次核矩阵为：")
print(k)
print("-------------------------------------------------------------")

N=np.eye(len(k))-np.ones(len(k))/len(k)
center=np.dot(np.dot(N,k),N)
print("中心化的核矩阵为：")
print(center)
print("-------------------------------------------------------------")

sum=0
for i in range(len(k)):
    for j in range(len(k)):
        sum+=k[i][j]
mu=sum/len(k)**2
normal=[]
for i in range(len(k)):
    normalk=[]
    for j in range(len(k)):
        m=k[i][j]/np.sqrt(k[i][i]*k[j][j])
        normalk.append(m)
    normal.append(normalk)
normal=np.array(normal)
print("标准化的核矩阵为：")
print(normal)
print("-------------------------------------------------------------")

space=[]
col=data.shape[1]
for i in range(len(data)):
    rowsp=[]
    for j in range(col):
        sp=data[i][j]**2
        rowsp.append(sp)
    for k in range(col-1):
        for m in range(col-k-1):
            j=k+m+1
            sp=2**0.5*data[i][k]*data[i][j]
            rowsp.append(sp)
    space.append(rowsp)
space=np.array(space)
print("映射到特征空间的点：")
print(space)
print("-------------------------------------------------------------")

sk=[]
for i in range(len(space)):
    rowk=[]
    for j in range(len(space)):
        tempk=np.dot(space[i].T,space[j])
        rowk.append(tempk)
    sk.append(rowk)
sk=np.array(sk)
#print(sk)
#print("-------------------------------------------------------------")

mu=[]
col=space.shape[1]
for i in range(col):
    m=np.mean(space[:,i])
    mu.append(m)
center_s=[]
for i in range(len(space)):
    rowcenter=[]
    for j in range(col):
        centers=space[i][j]-mu[j]
        rowcenter.append(centers)
    center_s.append(rowcenter)
center_s=np.array(center_s)
print("中心化后的特征空间的点：")
print(center_s)
print("-------------------------------------------------------------")

center_sk=[]
for i in range(len(center_s)):
    rowk=[]
    for j in range(len(center_s)):
        tempk=np.dot(center_s[i],center_s[j].T)
        rowk.append(tempk)
    center_sk.append(rowk)
center_sk=np.array(center_sk)
print("中心化后的特征空间的点求得的核矩阵为：")
print(center_sk)
print("-------------------------------------------------------------")

normal_s=[]
for i in range(len(space)):
    temp=space[i]/np.linalg.norm(space[i])
    normal_s.append(temp)
normal_s=np.array(normal_s)
print("标准化后的特征空间的点：")
print(normal_s)
print("-------------------------------------------------------------")


normal_sk=[]
for i in range(len(normal_s)):
    rownormal=[]
    for j in range(len(normal_s)):
        temp=np.dot(normal_s[i],normal_s[j].T)
        rownormal.append(temp)
    normal_sk.append(rownormal)
normal_sk=np.array(normal_sk)
print("标准化后的特征空间的点求的核矩阵为：")
print(normal_sk)

