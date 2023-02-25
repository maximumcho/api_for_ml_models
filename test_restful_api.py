import numpy as np
import requests
import json

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()

#print("iris: " + str(iris))

# Split into train and test sets using the same random state
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=12)

# Serialize the data into json and send the request to the model
xTest = json.dumps(X_test.tolist());
payload = {'data': xTest}
# Fix the single quote bug
payload = json.dumps(payload)

#print(payload)

# Set request's header.
headers = {'Content-type': 'application/json'}
y_predict = requests.post('http://127.0.0.1:5000/iris', headers=headers, data=payload)

# Make array from the list
y_predict = np.array(y_predict.json())
print(y_predict)