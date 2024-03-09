import numpy as np
import csv
import datetime
date = "2013-11-01"
for i in range(144):
    form = [[0]* 144 for _ in range(10001)]  
    with open("./dataset/"+ date + "-" + str(i) + ".csv") as csvfile :
        inline = []
        outline = []
        datas = list(csv.reader(csvfile,delimiter = ","))
        for data in datas:
            for da in data:
                da = int(da)
                inline.append(da)
            outline.append(inline)      
        maps = np.array(outline)       
        #print(maps)         
        print("max" , np.max(maps) , "pos" , np.argmax(maps))



# 1,1   98,0   2,1   98,1
# 1,0   99,0   2,0   99,1   3,0  99,2
       

