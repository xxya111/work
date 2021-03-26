import math
import pandas as pd
import numpy as np
import copy

#计算高斯核
def kernel(x,y,h,d):
    kernel = np.exp(-(np.linalg.norm(x - y) / h) ** 2/ 2) / (2. * np.pi)**(d/2)
    return kernel

#找密度吸引子
def findattarctor(x,data,h,e):
    m1=n1=0
    for i in range(len(data)):
        m=kernel(x,data[i],h,data.shape[1])*data[i]
        n=kernel(x,data[i],h,data.shape[1])*np.ones(data.shape[1]) #变成数组进行计算
        m1=m1+m
        n1=n1+n
    y=m1/n1     #X[t+1]
    d=np.linalg.norm(x-y)      #||x[t]-x[t-1]||
    if d>e:
        findattarctor(y, data, h, e)      #迭代找到符合条件的X[t]
    else:
        y = y.tolist()
    return y    #返回其密度吸引子


#计算密度吸引子概率密度的值
def faction(x,data,h,d):
    m=0
    for i in range(len(data)):
        n = kernel(x,data[i],h,d)
        m=m+n     #累加
    f=m/(len(data)*h**d)
    return f


def CA(A,data):    #寻找符合条件的密度吸引子
    k=5   #最邻近点的数目
    d=0.2  #半径
    x=[]
    for i in range(len(A)):
        #x.append([])
#        x1=A[i]
#        y1=D[1][x1]
        count = 0    #计数，密度吸引子的密度
        for j in range(len(data)):
#            x2=D[0][j]
#            y2=D[1][j]
            dis=np.linalg.norm(A[i]-data[j])  #距离
            if dis<=d:
                count=count+1
        if count>=k:      #除去噪声点
            x.append(A[i])   #符合条件的密度吸引子加入到新的队列中
    return x

def reach(x,C):      #找密度可达的密度吸引子集群
    d=0.2
    same=[]     #存放密度可达的密度吸引子集群
    dif=[]
    if len(x)!=1 and len(x)!=0 :      #如果密度吸引子个数大于1
        for i in range(1,len(x)):
            dis=math.sqrt((x[0][0]-x[i][0])**2+(x[0][1]-x[i][1])**2)    #计算第一个密度吸引子和其他密度吸引子的距离
            if dis<=d:        #如果可达，加入到same列表中
                same.append(x[i])
            else:             #如果不可达，加入到另一个列表
                dif.append(x[i])
        same.append(x[0])    #最后向密度可达的此集群里加入第一个密度吸引子
        same1=copy.deepcopy(same)     #深拷贝
        dif1=copy.deepcopy(dif)
        for i in range(len(same1)-1):       #遍历除去第一个密度吸引子的列表，判断与dif中剩余的密度吸引子是否密度可达
            for j in range(len(dif1)):
                dis =math.sqrt((same1[i][0]-dif1[j][0])**2+(same1[i][1]-dif1[j][1])**2)     #距离
                if dis<=d:             #如果可达，则与same列表内其余密度吸引子也可达
                    if dif1[j] not in same:    #添加到same列表中
                        same.append(dif1[j])
                    if dif1[j] in dif:      #从dif列表中删除
                        dif.remove(dif1[j])
    elif len(x)==1:       #如果只有一个密度吸引子，则自己为一个集群
        same=x
    elif len(x)==0:      #没有元素则返回
        return

    C.append(same)   #得到一个集群存入到一个列表中
    reach(dif,C)    #对剩余的元素进行迭代，继续找密度可达的密度吸引子集群，直到dif为空
    return C

def denclue(data,h,min,e):        #h:窗口带宽  min：最小密度阈值  e：容差
    A=[]        #存放密度吸引子
    R={}        #存放密度吸引子和其点集的字典
    d=data.shape[1]
    for i in range(len(data)):  #依次找data的密度吸引子
        x=findattarctor(data[i],data,h,e)
        fx=faction(x,data,h,d)
        if fx>=min:   #如果密度吸引子概率大于阈值，存入
            if ",".join('%s' %id for id in x) in R:      #字典的键不能是列表，转为字符串
                r = R[",".join('%s' % id for id in x)]    #密度吸引子在字典内，取出其值，将新的点加入列表，再存入字典内
                r.append(data[i].tolist())
                R[",".join('%s' % id for id in x)] = r
            else:                                    #密度吸引子不在字典内，添加键值对存储密度吸引子和点集
                R[",".join('%s' %id for id in x)]=[data[i].tolist()]
            x=x.tolist()       #数组转为列表存入列表中
            if x not in A:     #如果密度吸引子不在列表内，则存入，避免重复
                A.append(x)

    a=CA(A,data)    #除去噪声点
    C=[]
    c=reach(a,C)    #密度可达的密度吸引子集群
    C1=np.array(c)  #转为数组形式
#    climb=[]
#    climbp=[]
    '''for i in range(len(C1)):
        for j in range(len(C1[i])):
            climb=climbp+R[C1[i][j]]
        climbp.append(climb)'''
    print("集群的数量为："+str(len(C1)))
    for i in range(len(C1)):
        numb=0
        for j in range(len(C1[i])):
            num=len(R[",".join('%s' %id for id in C1[i][j])])
            numb=numb+num
        print("第"+str(i+1)+"个集群大小为："+str(numb))
        #print("第" + str(i + 1) + "个集群大小为：" + str(len(R[C1[i]])))
    print("所有的密度吸引子为：")
    ''' Rlist=list(R.values())
    for i in range(len(Rlist)):
        for j in range(len(Rlist[i])):
    print(Rlist)'''
    FinalR={}   #合并点集后的字典
    for i in range(len(C1)):
        sonc=[]
        for j in range(len(C1[i])):     #遍历每个集群内的密度吸引子
            ele=R[",".join('%s' % id for id in C1[i][j])]
            for k in range(len(ele)):
                sonc.append(ele[k])
        FinalR[",".join('%s' % id for id in C1[i][0])]=sonc

    for i in range(len(C1)):
        print("第"+str(i+1)+"个：")
        print(str(C1[i]))
        print("其簇中的点集为：" )
        print(np.array(FinalR[",".join('%s' % id for id in C1[i][0])]))
        '''for j in range(len(C1[i])):
            print(R[",".join('%s' % id for id in C[i][j])])'''

           # print(R[",".join('%s' % id for id in C[i][j])])
        print("---------------------------------------")
       # print("其簇中的点集为：" + ",".join(R[",".join('%s' % id for id in A[i])]))

#计算聚类纯度
    cluster=[]
    all=0
    maxa=0
    for i in range(len(C1)):         #字典的值存到列表中
        cluster1=FinalR[",".join('%s' % id for id in C[i][0])]
        cluster.append(cluster1)
    for i in range(len(cluster)):      #求总聚类个数
        all=all+len(cluster[i])
    for i in range(len(cluster)):   #遍历每一个簇
        setosa = versicolor = virginica =0     #计数，每一个标签的个数
        for j in range(len(cluster[i])):      #遍历每一个簇内的点
            seq=data.tolist().index(cluster[i][j])   #获取在数据列表中的索引值
            if D[4][seq]=="Iris-setosa":
                setosa+=1
            elif D[4][seq]=="Iris-versicolor":
                versicolor+=1
            elif D[4][seq]=="Iris-virginica":
                virginica+=1
        maxn=max(setosa,versicolor,virginica)     #每一类中最多的标签的个数
        maxa=maxa+maxn     #所有类的最多的标签个数的和
    purity=maxa/all
    print("聚类纯度为："+str(purity))



#denclue(data,0.3,0.25,0.0001)
#输入参数
h=float(input("请输入带宽："))
m=float(input("请输入最小密度阈值："))
e=float(input("请输入容差："))
denclue(data,h,m,e)
