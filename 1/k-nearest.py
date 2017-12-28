import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

K_RANGE=30
df = pd.read_csv("iris.csv")
data = df.values
numberOfData=data.shape[0]

def getDistance(vector_A,vector_B):
    distance_list = []
    for i in range(4):
        a = vector_A[i]
        b = vector_B[i]
        square = pow((a-b),2)
        distance_list.append(square)
    distance = sum(distance_list)**(1/2)
    return distance

#return the most pattern of k data
#data_set must be ordered
def getMostPattern(data_set,k):
    class_lists = ["Iris-setosa","Iris-versicolor","Iris-virginica"]
    pattern=[0,0,0]
    for i in range (k):
        if data_set[i][4]==class_lists[0]:
            pattern[0]=pattern[0]+1
        elif data_set[i][4]==class_lists[1]:
            pattern[1]=pattern[1]+1
        elif data_set[i][4]==class_lists[2]:
            pattern[2]=pattern[2]+1
        else:
            print("Bug")
    pattern=np.array(pattern)
    answer = class_lists[pattern.argmax()]
    return answer

#return pattern using K neaest method
#data_set must be 5*n matrix
def getPattern(point,data_set,k):
    out_data = np.array(data_set)
    distance=[]
    for i in range(data_set.shape[0]):
        distance.append(getDistance(point,out_data[i]))
    distance = np.reshape(distance, (1, len(distance)))
    out_data=np.concatenate((out_data,distance.T),axis=1)
    #rearrange data_set according to the distance  ascending
    out_data = out_data[out_data[:,-1].argsort()]
    answer = getMostPattern(out_data,k)
    if point[4]==answer:
        return 1

    else:
        return 0

def leaveOneOut(data_set):
    identification_rates=[]
    for k in range(K_RANGE-1):
        k = k+1
        summation = 0
        for i in range(numberOfData):
            point = data_set[i]
            temp_data_set=data_set.copy()
            others = np.delete(temp_data_set,i,axis=0)
            if getPattern(point,others,k)==1:
                summation=summation+1
        rate =summation/numberOfData
        identification_rates.append(rate)
    x = np.arange(1, K_RANGE, 1)
    y = identification_rates
    print("最適なkの値は",x[np.argmax(identification_rates)])
    plt.plot(x, y)
    plt.xlabel("k")
    plt.show()

if __name__ == '__main__':
    leaveOneOut(data)
