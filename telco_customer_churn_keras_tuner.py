from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense,Input

df = pd.read_csv('/content/telcochurndata.csv')

df.head()

df.dtypes

df.isna().sum()

df.columns

X = df.drop('Churn',axis=1)
y = df['Churn']

X = df.drop(['customerID', 'Churn'], axis=1)
X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
X = X.dropna()
y = y[X.index]

# Select categorical columns excluding 'Churn' and 'customerID' which are already handled
categorical_cols = X.select_dtypes(include='object').columns

# One-hot encode categorical columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)


scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y.apply(lambda x: 1 if x == 'Yes' else 0), test_size=0.2, random_state=42)

inputs = Input(shape=(X_train.shape[1],))
l1 = Dense(8,activation='tanh')(inputs)
l2 = Dense(10,activation='tanh')(l1)
l3 = Dense(10,activation='tanh')(l2)
outputs = Dense(1,activation='sigmoid')(l3)

model.summary()

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=100,batch_size=5,verbose=1)

scores = model.evaluate(X_test,y_test)
print("Accuracy is : ",scores[1])



"""Now, let's use Keras Tuner to find better hyperparameters for our model.

First, we need to install Keras Tuner.
"""

!pip install keras_tuner

"""Next, we define a function to build the model, including the hyperparameters we want to tune."""

from keras_tuner import RandomSearch

def build_model(hp):
    inputs = Input(shape=(X_train.shape[1],))
    x = inputs
    for i in range(hp.Int('num_layers', 1, 3)):
        x = Dense(units=hp.Int('units_' + str(i), min_value=8, max_value=32, step=8),
                  activation='tanh')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=hp.Choice('optimizer', ['adam', 'rmsprop']),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

"""Now, we can instantiate a tuner and run the search. We will use `RandomSearch` for this example."""

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,  # Number of models to train
    executions_per_trial=1, # Number of models to train per trial
    directory='my_dir',
    project_name='telco_churn_tuning')

"""Let's check the search space summary."""

tuner.search_space_summary()

"""Now, run the hyperparameter search. This will take some time depending on the number of trials and epochs."""

tuner.search(X_train, y_train,
             epochs=50,
             validation_data=(X_test, y_test))

"""After the search is complete, we can get the best hyperparameters and the best model."""

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The optimal number of layers is {best_hps.get('num_layers')}.
The optimal optimizer is {best_hps.get('optimizer')}.
""")

for i in range(best_hps.get('num_layers')):
    print(f"The optimal number of units in layer {i} is {best_hps.get('units_' + str(i))}")

best_model = tuner.get_best_models(num_models=1)[0]

"""Finally, evaluate the best model on the test data."""

loss, accuracy = best_model.evaluate(X_test, y_test)
print(f"Accuracy of the best model: {accuracy}")

"""Let's visualize the architecture of the best model."""

tf.keras.utils.plot_model(best_model, to_file='best_model_architecture.png', show_shapes=True, dpi=96, show_layer_activations=True, show_trainable=True)
