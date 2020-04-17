import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc",size=10)   #字体

D=pd.read_csv('C:/Users\XJW\Desktop\magic04.data',header=None)   #读取数据
#print(D)
#print(np.cov(D[0],D[5]))
'''d=np.array(D).T[0:8]
d1=d.T
print(d1)
d1=d1.astype('float32')
q = np.cov(d1,rowvar=False)
print(q)'''

print("均值向量为：")
p=[]
#m=np.cov(D[0],D[1])
#print(m[0][1])
for i in range(10):
    x=D[i]
    m=np.mean(x)           #求每个属性的均值
    p.append(m)      #将每个属性均值加入列表中
#print(p)
print(np.array(p).T)

print("**********************************************************************************")
Z=[]
for i in range(10):
    Z.append([])
    for j in range(len(D)):
        #Z.append([])
        n=D[i][j]-p[i]        #中心化
        Z[i].append(n)       #将数据加入到列表中
#print(Z)
ZT=np.transpose(Z)            #转置
Y1=np.dot(np.array(Z),np.array(ZT))       #内积
Y=Y1/len(D)                  #进行标准化
print("中心化矩阵内积求协方差矩阵：")
print(Y)

print("**********************************************************************************")
y=np.zeros(shape=(10,10))        #生成都是0的数组
for i in range(len(D)):
    z=[]
    for j in range(10):
        n = D[j][i] - p[j]       #中心化
        z.append(n)
    z1=np.array(z)
    zT=np.transpose(z1)
    y1=np.outer(zT,z1)           #每一个属性的值做外积
    #print(zT)
    y=y+y1                      #累加
Y1=y/len(D)                     #进行标准化
print("中心化数据点外积求协方差矩阵：")
print(Y1)

print("**********************************************************************************")
#求属性1和属性2相关性
#cosc=np.dot(D[0],D[1])/(np.linalg.norm(D[0])*np.linalg.norm(D[1]))
#print(cosc)
m=D[0]-p[0]  #中心化属性1数据
n=D[1]-p[1]
cosco=np.dot(m,n)/(np.linalg.norm(m)*np.linalg.norm(n))
print("属性1和属性2的相关性为："+str(cosco))

#绘制属性1和属性2散点图
plt.scatter(D[0],D[1])
plt.title('属性1和属性2的散点图',fontproperties=font)
plt.xlabel('x:attribute1')
plt.ylabel('y:attribute2')
plt.show()

#绘制属性1的概率密度函数图
x=D[0]           #取属性1
mu=np.mean(x)    #均值
st=np.std(x)     #标准差
x1=np.arange(mu-3*st,mu+3*st,0.1)    #横坐标范围
y=1 / (np.sqrt(2 * math.pi) * st)*(np.exp(-(x1 - mu) ** 2.0 / (2 * st*st)))    #正态分布函数
plt.plot(x1,y)
plt.title('属性1的概率密度函数',fontproperties=font)
plt.show()


#求最大方差和属性
def maxvar(D):
    n=np.var(D[0])
    for i in range(10):
        m=D[i]    #第i列属性的数据
        n1=np.var(m)     #计算方差
        if n1>n:    #找最大值
            n=n1
            num=i+1    #下标
    print("属性"+str(num)+"方差最大，为："+str(n))


#求最小方差和属性
def minvar(D):
    n=np.var(D[0])
    for i in range(10):
        m=D[i]
        n1=np.var(m)
        if n1<n:
            n=n1
            num=i+1
    print("属性"+str(num)+"方差最小，为："+str(n))

#求最大协方差
def maxcov(D):
    s=np.cov(D[0],D[1])
    s1=s[0][1]
    for i in range(9):
        for j in range(9-i):    #和除本列以外的其他属性进行协方差计算
            n=i+j+1
            q=np.cov(D[i],D[n])  #协方差矩阵
            q1=q[0][1]    #取协方差
            if q1>s1:    #找最大值
                s1=q1
                num1=i+1   #下标
                num2=n+1
    print("第"+str(num1)+"列和第"+str(num2)+"列协方差最大,为:"+str(s1))


#求最小协方差
def mincov(D):
    s=np.cov(D[0],D[1])
    s1=s[0][1]
    for i in range(9):
        for j in range(9-i):
            n=i+j+1
            q=np.cov(D[i],D[n])
            q1=q[0][1]
            if q1<s1:
                s1=q1
                num1 = i + 1
                num2 = n + 1
    print("第"+str(num1)+"列和第"+str(num2)+"列协方差最小,为:"+str(s1))

maxvar(D)
minvar(D)
maxcov(D)
mincov(D)

