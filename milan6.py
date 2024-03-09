import csv
import numpy as np
import datetime
import os
def maxsec(map):
    loc = np.argmax(map)
    print(loc)
    x , y = loc % 100 , loc // 100
    loca = (x,y)
    area = 100 * y + x
    result = area , loca
    return result
#2weeks(1680,2016)  all (7440,8928)
#def process(term):
#    if term == "2weeks":
#        start,mid,end = 0,1680,2016
#    if term == "2week_2":
#        start,mid,end = 2016,3696,4032
#    if term == "2week_3":
#        start,mid,end = 4032,5712,6048
#    if term == "2week_4":
#        start,mid,end = 6048,7728,8064
#    if term == "november":
#        start,mid,end = 0,3600,4320
#    if term == "december":
#        start,mid,end = 4320,8040,8784
#    if term == "all":
#        start,mid,end = 0,7440,8928
#    result = start , mid , end
#    print(result)
#    return result

def load_data(date = datetime.date(2013,11,1)):
    print("load from scratch!!")
    delta = datetime.timedelta(days = 1)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    data = []
    arealist = []
    location = []
    count = 0
    
    for i in range(62):
        #print("/" * 25 , i , "/" * 25)
        for j in range(144):
            with open("./dataset/" + str(date) + "-" + str(j) + ".csv" , "r") as csvfile:
                map = [[[0]*100]*100]
                map = list(csv.reader(csvfile , delimiter = ","))
                map = np.array(map ,dtype = np.int64)
                data.append(map)
                if count != 0:
                    print(count," : " ,end = "")
                    result = maxsec(map)
                    arealist.append(result[0])
                    location.append(result[1])
            count += 1
        date += delta
    
    arealist.append(5563)
    location.append((63,44))
    data = np.array(data)
    location = np.array(location)
    np.savez("data.npz",data,location)

#for i in range(100):
#    for j in range(100):

#    print(data , data.shape)
#    print(arealist , len(arealist))

#    term = input("please enter term : ")
#    term = term.strip()
#    start ,mid, end = process(term)
    #2weeks(1680,2016)  all (7440,8928)
#    x_train = data[start:mid]
#    y_train = arealist[start:mid]
#    x_test = data[mid:end]
#    y_test = arealist[mid:end]
#    x_train = np.array(x_train)
#    y_train = np.array(y_train)
#    x_test = np.array(x_test)
#    y_test = np.array(y_test)
#np.savez("log_area_"+ term +".npz" , x_train , y_train , x_test , y_test)


date = datetime.date(2013,11,1)
#(x_train , y_train) , (x_test , y_test) = load_data(date)
#if np.load("data.npz") is None:
#load_data(date)

datas = np.load("data.npz")
data = datas.files
map = datas["arr_0"]
location = datas["arr_1"]
place = np.zeros((101,101))
trajectory = np.zeros((10,10))
flow = np.zeros((10,10))
record = []
#for i in range(288,432):
#    trajectory[location[i][0]//10][location[i][1]//10] += 1
##print(i,location[i],location[i][0]//10 , location[i][1]//10 , trajectory[location[i][0]//10][location[i][1]//10])
##    print(i," : ",location[i][0] , location[i][1])
##print(trajectory[location[0][0]//10][location[0][1]//10])
start = 0;
end = 144;
for k in range(62):
    flow = np.zeros((10,10))
    for i in range(start,end):
        trajectory[location[i][0]//10][location[i][1]//10] = i
        flow[location[i][0]//10][location[i][1]//10] += 1
#        print("-"*50)
#        print(i)
    print(k,"\n")
    print(trajectory,"\n")
    start = end
    end += 144
    print(flow)
    print("/"*50 , "\n")
#print("\n\n")
#for i in range(144):
#    print(location[i])




#print("x_train" , x_train , "shape : " , x_train.shape)
#print("y_train" , y_train , "shape : " , y_train.shape)
#print("x_test" , x_test , "shape : " , x_test.shape)
#print("y_test" , y_test , "shape : " , y_test.shape)

