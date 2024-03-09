import numpy as np
import csv
import datetime
def value(list,i) : 
    list[i] = list[i].strip()
    if list[i] is "":
        list[i] = list[i].replace("" , "0")
    return float(list[i])    
def process(date):
    with open("./org_data/sms-call-internet-mi-"+ str(date) + ".txt", "r") as res:
        datas = res.readlines()
    flag = 0
    form = [[0] * 144 for _ in range(10001)]
    sms_ins = np.array([[0.0] * 144 for _ in range(10001)],dtype = float)
    sms_outs = np.array([[0.0] * 144 for _ in range(10001)],dtype = float)
    call_ins = np.array([[0.0] * 144 for _ in range(10001)],dtype = float)
    call_outs = np.array([[0.0] * 144 for _ in range(10001)],dtype = float)
    inter_acts = np.array([[0.0] * 144 for _ in range(10001)],dtype = float)
    map = np.array([[[0.0] * 100] * 100]*5,dtype = float)
    for data in datas:      
        list = data.split('\t')
        if flag == 0:
            start = int(list[1][0:8])
            flag = 1
        sqrd_id = int(list[0])
        time = int(list[1][0:8])
        time_sec = (time-start)//6
        con_code = int(list[2])
        #print(list)
        sms_in = value(list,3)
        sms_out =  value(list,4)
        call_in =  value(list,5)
        call_out = value(list,6)
        inter_act = value(list,7)
        cdr = sms_in + sms_out + call_in + call_out + inter_act
        try:
            #form[sqrd_id][time_sec] += cdr
            sms_ins[sqrd_id][time_sec] += sms_in
            sms_outs[sqrd_id][time_sec] += sms_out
            call_ins[sqrd_id][time_sec] += call_in
            call_outs[sqrd_id][time_sec] += call_out
            inter_acts[sqrd_id][time_sec] += inter_act
        except:
            print(date,sqrd_id,time_sec,time)
           # print(form)
            break    
        

    #print(form) 
    

    for i in range(144):
        #path = "./dataset/" + str(date) + "-" + str(i) +".csv"
        path1 = "./dataset/sms_in/" + str(date) + "-" + str(i) +".csv"
        path2 = "./dataset/sms_out/" + str(date) + "-" + str(i) +".csv"
        path3 = "./dataset/call_in/" + str(date) + "-" + str(i) +".csv"
        path4 = "./dataset/call_out/" + str(date) + "-" + str(i) +".csv"
        path5 = "./dataset/inter_act/" + str(date) + "-" + str(i) +".csv"
        out1 = open(path1,"w+")
        out2 = open(path2,"w+")
        out3 = open(path3,"w+")
        out4 = open(path4,"w+")
        out5 = open(path5,"w+")
        for sqrd_id in range(1,10001):
            x = sqrd_id % 100
            y = sqrd_id // 100
#            print("map",type(map[0][99-y][x-1]))
#            print("sms",type(sms_ins[sqrd_id][i]))
            #map[99-y][x-1] = form[sqrd_id][i]
            map[0][99-y][x-1] = sms_ins[sqrd_id][i]
            map[1][99-y][x-1] = sms_outs[sqrd_id][i]
            map[2][99-y][x-1] = call_ins[sqrd_id][i]
            map[3][99-y][x-1] = call_outs[sqrd_id][i]
            map[4][99-y][x-1] = inter_acts[sqrd_id][i]
        mywriter = csv.writer(out1 , delimiter = ",")
        mywriter.writerows(map[0])
        mywriter = csv.writer(out2 , delimiter = ",")
        mywriter.writerows(map[1])
        mywriter = csv.writer(out3 , delimiter = ",")
        mywriter.writerows(map[2])
        mywriter = csv.writer(out4 , delimiter = ",")
        mywriter.writerows(map[3])
        mywriter = csv.writer(out5 , delimiter = ",")
        mywriter.writerows(map[4])
        
date = datetime.date(2013,11,1)
delta = datetime.timedelta(days = 1)
for i in range(62) :
    process(date)
    date += delta

    






# 1,1   98,0   2,1   98,1
# 1,0   99,0   2,0   99,1   3,0  99,2
       

