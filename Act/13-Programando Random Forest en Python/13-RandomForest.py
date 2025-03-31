#Automatic creation of an virtual environment to run the script and intall the libraries
import subprocess
import os
import venv
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
env_name = os.path.join(script_dir, "VirtualEnv")
if os.path.exists(os.path.join(script_dir, "VirtualEnv")):
    #Checks if the VirtualEnv is activated (This is the path to the Python installation currently in use. If the virtual environment is active, sys.prefix will point to the virtual environment directory, while sys.base_prefix points to the global Python installation.)
    if sys.prefix == sys.base_prefix:
        print("Activating the Virtual Environment...")
        python_exe = os.path.join(env_name, "Scripts", "python")
        subprocess.run([python_exe, __file__])
else:
    print("Installing the Required Libraries on a New Virtual Environment")
    venv.create(env_name, with_pip=True)

    # Step 2: Install the libraries
    libraries = ["scikit-learn", "matplotlib","seaborn","pandas","numpy"]
    for lib in libraries:
        subprocess.run([os.path.join(env_name, "Scripts", "pip"), "install", lib], check=True)
    
    #Re-Run the script with the Virtual Env Activated
    python_exe = os.path.join(env_name, "Scripts", "python")
    subprocess.run([python_exe, __file__])

#Random Forest
#Importacion de Librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from matplotlib import rcParams

from collections import Counter

#set up graphic style in this case I am using the color scheme from xkcd.com
rcParams['figure.figsize'] = 14, 8.7 # Golden Mean
LABELS = ["Normal","Fraud"]
df = pd.read_csv("creditcard.csv")
pd.set_option('display.max_columns', None) #Muestra todas las columnas del dataSet

print("Primeros 5 Registros")
print(df.head()) 
print("\n\nForma del dataset")
print(df.shape)

print("Desbalanceo en el DataSet")
print(pd.value_counts(df['Class'], sort = True))

normal_df = df[df.Class == 0] #registros normales
fraud_df = df[df.Class == 1] #casos de fraude

#Creacion del DataSet
y = df['Class']
X = df.drop('Class', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

#Cracion de una Matriz de Confusion
def mostrar_resultados(y_test, pred_y):
    conf_matrix = confusion_matrix(y_test, pred_y)
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    print (classification_report(y_test, pred_y))

#Ejecucion del Modelo con Regresion Logistica para Comparacion
def run_model_balanced(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(C=1.0,penalty='l2',random_state=1,solver="newton-cg",class_weight="balanced")
    clf.fit(X_train, y_train)
    return clf

#Modelo Regresion Logistica
model = run_model_balanced(X_train, X_test, y_train, y_test)

#Mostrar Resultados
pred_y = model.predict(X_test)
print("\n\nResultados Regresion Logistica")
mostrar_resultados(y_test, pred_y)



#Creacion del RandomForest1
# Crear el modelo con 100 arboles
model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,verbose=2,
                               max_features = 'sqrt',
                               n_jobs=-1) #USA TODOS LOS NUCLEOS DISPONIBLES
# entrenar!
model.fit(X_train, y_train)

#Mostrar Resultados
pred_y = model.predict(X_test)
print("\n\nResultados Random Forest 1")
mostrar_resultados(y_test, pred_y)


# Creacion del RandomForest2 (Cambio de Parametros)
model = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                               max_features = 'sqrt', verbose=2, max_depth=6,
                               oob_score=True, random_state=50,
                               n_jobs=-1) #USA TODOS LOS NUCLEOS DISPONIBLES
# a entrenar
model.fit(X_train, y_train)

#Mostrar Resultados 
pred_y = model.predict(X_test)
print("\n\nResultados Random Forest 2")
mostrar_resultados(y_test, pred_y)


# Calculate roc auc
roc_value = roc_auc_score(y_test, pred_y)
print("\n\nROC AUC Value:")
print(roc_value)

input("Presiona Cualquier Tecla para Continuar...")