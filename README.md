# jika menjalankan di lokal

1. buat virtual environment 

2. install semua library

pip install -r requirements.txt

3. jalankan program train_classifier

python train_classifier.py

4. pastikan di pipeline/loanclassifier telah ada preprocessor.dill & model.dill yang merupakan file hasil train diatas

5. pastikan di file pipeline/loanclassifier/Model.py akan membaca file preprocessor.dill & model.dill yang terdapat di pipeline/loanclassifier

6. lakukan test dengan code

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

7. latih outlier detector

python train_detector.py

8. pastikan di pipeline/outliersdetector sudah tergenerate 2 file yaitu preprocessor.dill & model.dill

9. di file pipeline/outliersdetector/Detector.py pastikan membaca file preprocessor.dill & model.dill yang terdapat di direktori pipeline/outliersdetector

10. jalankan test dengan code

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

# build image

jalankan semua perintah di Makefile pertama mulai dari base, loanclassifier, outliersdetect, dan combiner

# deploy dengan kubernetes

## install istio

1. download, kita gunakan versi 1.22.1

https://github.com/istio/istio/releases/download/1.22.1/istioctl-1.22.1-win.zip

untuk melihat versi lainnya

https://github.com/istio/istio/releases

2. ekstrak dan masukkan file nya (file istioctl-{versi}-win) kedalam path environment variable (lebih tepatnya system variable)

3. cek versi Istio

istioctl version

sumber Langkah 1, 2, 3

https://medium.com/@amrilhakimsihotang/cara-install-istio-di-windows-11-291d2967803c

4. install yang versi demo 

istioctl install --set profile=demo

https://dev.to/wn/getting-started-with-docker-kubernetes-and-istio-on-windows-3b2m

5. buat namespace baru, pada namespace ini kita install Istio

                kubectl create namespace istio-system

                kubectl label namespace istio-system istio-injection=enabled --overwrite


6. cek namespace yang telah dipasang Istio (lewat label istio-injection)

                kubectl get namespace -L istio-injection

7. buat gateway (ingress), file nya lihat di deploy/ingress.yaml

8. deploy dengan Kubernetes, ingress nya

                kubectl apply -f deploy/ingress.yaml

9. cek ingress

                kubectl get svc -n istio-system

## install seldon

                helm install seldon-core seldon-core-operator --repo https://storage.googleapis.com/seldon-charts --set usageMetrics.enabled=true --namespace istio-system --set istio.enabled=true

## deploy classifier & outlier detector secara terpisah ke kubernetes

                kubectl apply -f deploy/loanclassifier.yaml

                kubectl apply -f deploy/outliersdetector.yaml

lakukan port-forward, untuk service ambassador printahnya :

                kubectl port-forward svc/ambassador 8003:80

test dengan code python

import json

from seldon_core.seldon_client import SeldonClient
from seldon_core.utils import get_data_from_proto
to_explain = X_test[:3]
print(to_explain)

sc = SeldonClient(
    gateway="ambassador",
    deployment_name="loanclassifier",
    gateway_endpoint="localhost:8003",
    payload_type="ndarray",
    namespace="seldon",
    transport="rest",
)

prediction = sc.predict(data=to_explain)
get_data_from_proto(prediction.response)

sc = SeldonClient(
    gateway="ambassador",
    deployment_name="outliersdetector",
    gateway_endpoint="localhost:8003",
    payload_type="ndarray",
    namespace="seldon",
    transport="rest",
)

prediction = sc.predict(data=to_explain)
get_data_from_proto(prediction.response)

## deploy dengan combiner

                kubectl apply -f pipeline/combiner.yaml

jika belum lakukan port-forward

lalu test dengan code python

sc = SeldonClient(
    gateway="ambassador",
    deployment_name="loanclassifier-combined",
    gateway_endpoint="localhost:8003",
    payload_type="ndarray",
    namespace="seldon",
    transport="rest",
)

prediction = sc.predict(data=to_explain)
output = get_data_from_proto(prediction.response)
print(prediction.response)

print(output["loanclassifier"])

print(output["outliersdetector"])