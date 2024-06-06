# TEST PERTAMA (train classifier)
import sys

sys.path.append("pipeline/loanclassifier")
from Model import Model

model = Model()

import numpy as np

from train_classifier import load_data

data, X_train, y_train, X_test, y_test = load_data()
proba = model.predict(X_test)

pred = np.argmax(proba, axis=1)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred)

# TEST KEDUA (detector)
import sys

sys.path.append("pipeline/outliersdetector")
from Detector import Detector

detector = Detector()

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from alibi_detect.utils.data import create_outlier_batch
from sklearn.metrics import confusion_matrix, f1_score

np.random.seed(1)
outlier_batch = create_outlier_batch(
    data.data, data.target, n_samples=1000, perc_outlier=10
)
X_outlier, y_outlier = outlier_batch.data.astype("float"), outlier_batch.target
y_pred = detector.predict(X_outlier)
labels = outlier_batch.target_names
f1 = f1_score(y_outlier, y_pred)
print("F1 score: {}".format(f1))
cm = confusion_matrix(y_outlier, y_pred)
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
sns.heatmap(df_cm, annot=True, cbar=True, linewidths=0.5)
plt.show()

# TEST KETIGA (deploy terpisah)
import json

from seldon_core.seldon_client import SeldonClient
from seldon_core.utils import get_data_from_proto

from train_classifier import load_data
data, X_train, y_train, X_test, y_test = load_data()

to_explain = X_test[:3]
print(to_explain)

sc = SeldonClient(
    gateway="istio-ingressgateway",
    deployment_name="loanclassifier",
    gateway_endpoint="localhost:1234",
    payload_type="ndarray",
    namespace="istio-system",
    transport="rest",
)

prediction = sc.predict(data=to_explain)
get_data_from_proto(prediction.response)
print(get_data_from_proto)