import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.stats import mode
import seaborn as sns
df = pd.read_csv("prediction_obama.csv")
size = df.shape[0]
#print size
print df['pred'].value_counts()
j = 0
l = []
for i in range(20,size,20):
	d = df.iloc[j:i]
	j = i
	a = d['pred'].value_counts()
	#print a.index[0]
	#l.append(a.iget(0))
	l.append(a.index[0])

print len(l)
x = list(range(0, len(l)))
y = l
plt.plot(x,y,'*')
plt.ylim(0,8)
plt.xlabel(" frames(20) ")
plt.title("emotions per 20 frames")
plt.ylabel("emotion")
plt.savefig('obama.png')
plt.show()

# bar plot for total percentage of airline sentiment
labels = 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'
sizes = [1548,0,4,355,557,4,80,0]
plt.bar([0,1,2,3,4,5,6,7],sizes)
plt.xlabel("Facial expresion")
plt.ylabel("total number expression ")
plt.title("putin vs trump")
plt.xticks([0,1,2,3,4,5,6,7],['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],rotation=45)
plt.savefig('trump vs putin.png')
plt.show()

 
# bar plot for total percentage of airline sentiment
labels = 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'
sizes = [1730,0,0,795,886,0,39,0]
plt.bar([0,1,2,3,4,5,6,7],sizes)
plt.xlabel("Facial expresion")
plt.ylabel("total number expression ")
plt.title("obama vs trump")
plt.xticks([0,1,2,3,4,5,6,7],['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],rotation=45)
plt.savefig('obama vs trump.png')
plt.show()

###########
'''labels = 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'
sizes = [1548,0,4,355,557,4,80,0]
plt.bar([0,1,2,3,4,5,6,7],sizes)
plt.xlabel("Facial expresion")
plt.ylabel("total number expression ")
#plt.title("putin")
plt.xticks([0,1,2,3,4,5,6,7],['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],rotation=45)
plt.show()'''