import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Convolution2D
from keras.optimizers import Adam
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model, preprocessing
from sklearn.decomposition import PCA   


scaling = True
principal = True
create_model = True
training = True


# Read Data    
train_lab = pd.read_hdf("data/train_labeled.h5")
train_unlab = pd.read_hdf("data/train_unlabeled.h5")
test = pd.read_hdf("data/test.h5")
x_test= test.values
train_lab = train_lab.values
x_train_unlab = train_unlab.values
y_train_lab = train_lab[0:, 0]
x_train_lab = train_lab[0:, 1:]

# Preprocess Data (Scaling)
if scaling is True:
	# Scale unlabeled data
    scaler_unlab = preprocessing.StandardScaler().fit(x_train_unlab)
    x_train_unlab = scaler_unlab.transform(x_train_unlab)
    # Scale labeled data
    scaler_lab = preprocessing.StandardScaler().fit(x_train_lab)
    x_train_lab = scaler_lab.transform(x_train_lab)
    # Scale test data
    scaler_test = preprocessing.StandardScaler().fit(x_test)
    x_test = scaler_test.transform(x_test)

if principal is True:
	# Fit PCA transformation to x_train_unlab
    pca = PCA()
    pca.fit(x_train_unlab)
    # Transform x_train_lab and x_test using previous PCA transformation
    x_train_lab = pca.transform(x_train_lab)
    x_test = pca.transform(x_test)

if create_model is True:
	model = Sequential()
	model.add(Dense(1000, activation='relu', input_dim=x_train_lab.shape[1]))
	model.add(Dropout(0.5))
	#model.add(Dense(800, activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

if training is True:

	x_train, x_test, y_train, y_test = train_test_split(x_train_lab, y_train_lab, test_size = 0.2, random_state=0)
	y_test = keras.utils.to_categorical(y_test, 10)
	y_train = keras.utils.to_categorical(y_train, 10)	
	model.fit(x_train, y_train, epochs=200, batch_size=128)
	score = model.evaluate(x_test, y_test, batch_size=128)
	results = model.predict_classes(x_test)
	np.savetxt("data/submit2.csv", results, delimiter=',')
