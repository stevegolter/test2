from pandas import Series
from pandas import DataFrame
from pandas import TimeGrouper
import matplotlib.pyplot as plt
import pandas as pd
import pymysql
import mysql.connector
from sqlalchemy import create_engine
from scipy.misc import imread
import matplotlib.cbook as cbook
import numpy as np
from matplotlib import style
#style.use("ggplot")
from sklearn.cluster import KMeans

df= []

engine = create_engine("mysql+mysqlconnector://traffic_counter:tr@ff1c@10.60.10.13:3306/traffic")
#between '2018-06-29 13:36:00' and '2018-06-29 13:36:05'
#df = pd.read_sql_query("SELECT * FROM traffic_log WHERE location = '10th and Ford' order by date_time", engine)
df = pd.read_sql_query("SELECT * FROM traffic_log WHERE date_time between '2018-07-10 17:07:00' and '2018-07-10 17:27:00' AND location = '10th and Ford' order by date_time", engine)
x = df['dir_x']
y = df['dir_y']
date = df[b'date_time']
#print(df['dir_x'])
#print(df.head(20))

#print(df.columns.tolist())

X = np.array(list(zip(x, y)))
#print('X', X)
kmeans = KMeans(n_clusters=8)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)

colors = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'w.']

for i in range(len(X)):
    #print("coordinate:",X[i], "label:", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 1)

	
plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)
plt.annotate('Group 1 \n % of Total Traffic: 13.3%', xy=(centroids[1, 0],centroids[1, 1]), xytext=(550,450),arrowprops=dict(facecolor='black', shrink=0.05),)
#plt.show()


	


#plt.scatter(x, y, s=0.01)
plt.xlim(0, 550)
plt.ylim(165, 550)
plt.title('Traffic Direction')
plt.xlabel('dir_x')
plt.ylabel('dir_y')

datafile = cbook.get_sample_data('image.jpg')
img = imread(datafile)
plt.imshow(img, extent=[0, 500, 165, 600])
plt.plot([52.094,427.792], [191.491,301.781])

plt.show()


