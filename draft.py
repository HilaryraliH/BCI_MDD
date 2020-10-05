import numpy as np
import csv
# f = open(u'sample.csv','r')
# reader = csv.reader(f)

# print(type(reader))
# for i in reader:
#     #print(type(i))
#     print(i)

# ['id', 'label']
# ['1', '1']
import csv
f2 = open('demo1.csv','w',newline='')
result = csv.writer(f2)
result.writerow(['id','label'])
result.writerows([['1','3'],[2,4]])
f2.close()







