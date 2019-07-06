# Import libraries
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split

# Import data
dataset = pd.read_csv('HexoANN/hexo.csv')
X = dataset.iloc[:, 0:6]
y = dataset.iloc[:, 6]

preprocess = make_column_transformer(
        (StandardScaler(), ['Open', 'High', 'Low', 'Close',
                            'Adj Close', 'Volume']))
X = preprocess.fit_transform(X)

# Split in train/test
y = y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ANN

# Initiation
classifier = Sequential()

# Ajout couche entrée et cachée

# Couche cachée
classifier.add(Dense(units=20, kernel_initializer='uniform',
                     activation='relu', input_dim=6))

# 6- Moyenne entre les entrés et sorties


# Ajout seconde couche cachée
classifier.add(Dense(units=20, kernel_initializer='uniform',
                     activation='relu'))

# Ajout de la couche de sortie
classifier.add(Dense(units=1, kernel_initializer='uniform',
                     activation='relu')) # si plus d'une categorie : softmax

# Compiler le réseau de neurone

classifier.compile(optimizer="adam", loss="mean_squared_error",
                   metrics=['msle'])

# Entrainer

classifier.fit(X_train, y_train, batch_size=2, epochs=500)

# Résultat

result = classifier.evaluate(X_test, y_test)

# Test nouvelle donnée

Xnew = pd.DataFrame(data={
         'Open': [6.71],
         'High': [6.91],
         'Low': [6.69],
         'Close': [6.88],
         'Adj Close': [6.88],
         'Volume': [807211]})

Xnew = preprocess.transform(Xnew)
newPrediction = classifier.predict(Xnew)
