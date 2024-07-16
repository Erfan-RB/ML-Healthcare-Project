#Prediction of Wine type 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep =';')
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep =';')

red.head()
white.tail()

red.sample(5)
white.describe()

pd.isnull(red)

# Create Histogram
fig, ax = plt.subplots(1, 2)
ax[0].hist(red.alcohol, 10, facecolor = 'red',
           alpha = 0.5, label = "Red wine")
ax[1].hist(white.alcohol, 10, facecolor ='white',
           ec ="black", lw = 0.5, alpha = 0.5,
           label ="White wine")

fig.subplots_adjust(left = 0, right = 1, bottom = 0, 
               top = 0.5, hspace = 0.05, wspace = 1)
 
ax[0].set_ylim([0, 1000])
ax[0].set_xlabel("Alcohol in % Vol")
ax[0].set_ylabel("Frequency")
ax[1].set_ylim([0, 1000])
ax[1].set_xlabel("Alcohol in % Vol")
ax[1].set_ylabel("Frequency")
 
fig.suptitle("Distribution of Alcohol in % Vol")
plt.show()

red["type"] = 1
white["type"] = 0
wines = red.append(white, ignore_index = True)

from sklearn.model_selection import train_test_split
X = wines.ix[:, 0:11]
y = np.ravel(wines.type)
X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size = 0.34, random_state = 45)

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(12, activation = 'relu', input_shape=(11, )))
model.add(Dense(9, activation ='relu'))
model.add(Dense(1, activation ='sigmoid'))
model.output_shape
model.summary()
model.get_config()
model.get_weights()
model.compile(loss ='binary_crossentropy', 
  optimizer ='adam', metrics =['accuracy'])

# Training Model
model.fit(X_train, y_train, epochs = 3,
           batch_size = 1, verbose = 1)
  
# Predicting the Value
y_pred = model.predict(X_test)
print(y_pred)
