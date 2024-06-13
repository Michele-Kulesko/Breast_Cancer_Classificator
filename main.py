import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold, GridSearchCV, cross_val_score, KFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, classification_report, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from urllib.request import urlopen
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis #, LocalOutlierFactor
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree, datasets, svm, metrics
import graphviz
import tensorflow as tf
from tensorflow import keras

from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.figure_factory as ff
from matplotlib.colors import ListedColormap

#from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')
pd.set_option('display.max_columns', 500)


from google.colab import drive
drive.mount('/content/drive')

data = pd.read_csv('data.csv')

data_violin = data

data.head()
data.info()

data = data.drop(['Unnamed: 32','id'],axis = 1)
data["diagnosis"] = data["diagnosis"].map({"M":1, "B":0})

data_save=data

def num_missing(x):
  return sum(x.isnull())

print("Missing values per column:")
print(data.apply(num_missing, axis=0)) #axis=0 defines that function is to be applied on each column

data.describe().T

data = data_save
plt.figure(figsize = (15,10)) #crea una figura con le dimensioni di 15x10 pollici utilizzando la funzione "figure"
sns.jointplot(data.radius_mean,data.area_mean,kind="reg") #la funzione "jointplot" della libreria viene utilizzata per creare un grafico di dispersione tra le colonne "radius_mean" e "area_mean"
plt.show()

plt.figure(figsize = (15,10))
sns.jointplot(data.area_mean,data.fractal_dimension_se,kind="reg")
plt.show()

#generiamo un grafico a dispersione con le sole colonne "mean"
cols = ['diagnosis',
        'radius_mean',
        'texture_mean',
        'perimeter_mean',
        'area_mean',
        'smoothness_mean',
        'compactness_mean',
        'concavity_mean',
        'concave points_mean',
        'symmetry_mean',
        'fractal_dimension_mean']

sns.pairplot(data=data[cols], hue='diagnosis', palette='rocket')

data.corr()

f,ax=plt.subplots(figsize = (20,20)) #creiamo una figura e un set di assi, stabilendo i parametri con figsize
sns.heatmap(data.corr(),annot= True,linewidths=0.5,fmt = ".2f",ax=ax) #costruiamo un heatmap
plt.xticks(rotation=90) #permette di ruotare le etichette dell'asse x per renderle legibili
plt.yticks(rotation=0)
plt.title('Correlation Map')
plt.show()

#Prima, elimino tutte le colonne "worst" perchè correlano tutte coi "mean"
cols = ['radius_worst',
        'texture_worst',
        'perimeter_worst',
        'area_worst',
        'smoothness_worst',
        'compactness_worst',
        'concavity_worst',
        'concave points_worst',
        'symmetry_worst',
        'fractal_dimension_worst']
data = data.drop(cols, axis=1)

#dopo, elimino tutte le colonne relative agli attributi perimetro e area perchè ovviamente sono correlati
cols = ['perimeter_mean',
        'perimeter_se',
        'area_mean',
        'area_se']
data = data.drop(cols, axis=1)

#alla fine, droppo tutte le colonne relative agli attributi di concavità e punti concavi perchè correlati con un altra che abbiamo lasciato
cols = ['concavity_mean',
        'concavity_se',
        'concave points_mean',
        'concave points_se']
data = data.drop(cols, axis=1)

data_final = data
#verifico le colonne rimanenti
data.columns

#Disegno nuovamente un heatmap, contenente la nuova matrice di correlazione
f,ax=plt.subplots(figsize = (10,10))
sns.heatmap(data.corr(),annot= True,linewidths=0.5,fmt = ".2f",ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Map')
plt.show()

col = data_violin.columns       #mettiamo in col le colonne del df
y = data_violin.diagnosis        #variabile che contiene i valori della colonna diagnosis
list = ['Unnamed: 32','id','diagnosis']         #array con colonne da eliminare
x = data_violin.drop(list,axis = 1 )   #nuovo df chiamato x contenere tutte le colonne tranne quelle messe in list
ax = sns.countplot(y,label="Count")    #qui con countploot della libreria seaborn creiamo un gafico che mostra il conteggio delle variabili e lo assegniamo ad ax
B, M = y.value_counts()               #con tale funzione conteggiamo e riportiamo nelle variabili B e M i valori di ogni variabile presente
                                    #il risultato sarà in modo che il primo elemento sia quello più frequente. Esclude i valori NA di default.
print('Number of Benign: ',B)
print('Number of Malignant : ',M)

lista = ["radius_mean", "texture_mean", 'smoothness_mean',
       'compactness_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'smoothness_se', 'compactness_se',
       'symmetry_se', 'fractal_dimension_se']


for i in lista:
  m = plt.hist(data[data["diagnosis"] == 1][i], bins=30, fc=(1,0,0,0.5), label="Tumori Maligni")
  b = plt.hist(data[data["diagnosis"] == 0][i], bins=30, fc=(0,1,0,0.5), label="Tumori Benigni")
  plt.legend()
  plt.xlabel(f"{i}")
  plt.ylabel("Frequenza")
  plt.title("Istogramma del {} del nucleo di cellule tumorali".format(i))
  plt.show()
  frequent_malignant_radius_mean = m[0].max()

for i in lista:
  data[i].plot.kde(title = i)
  plt.show()

for i in lista:
  data[i][data["diagnosis"] == 1].plot.kde(title = i, c = "r")
  data[i][data["diagnosis"] == 0].plot.kde(title = i, c = "g")
  plt.show()

y = data["diagnosis"]
x = data.drop(["diagnosis"],axis=1)

# la PCA ha bisogno della standardizzazione dei dati
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca = PCA(n_components=5) #numero di parametri da aggiungere dopo la riduzione della dimensionalità
pca.fit(x_scaled) #utilizzo il metodo fit per addestrare il modello di PCA sui dati x_scaled

# Calcolo dei valori di significatività delle componenti principali
variance_ratio = pca.explained_variance_ratio_

# Visualizzazione dei valori di significatività in un grafico a barre
plt.bar(range(len(variance_ratio)), variance_ratio) #funzione bar per creare il grafico a barre
plt.xlabel('Componenti Principali')
plt.ylabel('Percentuale di varianza spiegata')

# Aggiunta dell'etichetta con i valori di significatività nel grafico
for i, val in enumerate(variance_ratio):
    plt.text(i, val, str(round(val*100, 2)) + '%') #la funzione text utilizzata per aggiungere le etichette con i valori di significatività

plt.show()

# Costruzione PCA
pca = PCA(n_components = 3)
pca.fit(x_scaled) #addestro il modello PCA sui dati standardizzati
X_reduced_pca = pca.transform(x_scaled) #trasforma i dati originali in un insieme di dati ridotti

pca_data = pd.DataFrame(X_reduced_pca,columns=["p1","p2","p3"]) #creiamo un nuovo dataframe con i dati ridotti e assegnamo le varie etichette alle colonne
pca_data["diagnosis"] = y

hue = pca_data["diagnosis"]
data = [
    go.Scatter3d(
        x=pca_data.p1,
        y=pca_data.p2,
        z=pca_data.p3,
        mode='markers',
        marker=dict(
            size=4,
            color=hue,
            symbol="circle",
            line=dict(width=2)
        )
    )
]

layout = go.Layout(title="PCA",
                   scene=dict(
                       xaxis=dict(title="p1"),
                       yaxis=dict(title="p2"),
                       zaxis=dict(title="p3")
                   ),
                   hovermode="closest")

fig = go.Figure(data=data, layout=layout)

# aggiunta animazione per rotazione
fig.update_layout(scene=dict(camera=dict(up=dict(x=0, y=0, z=1),
                                         center=dict(x=0, y=0, z=0),
                                         eye=dict(x=2, y=2, z=0.1))),
                  updatemenus=[dict(type='buttons',
                                    showactive=False,
                                    buttons=[dict(label='Rotate',
                                                  method='animate',
                                                  args=[None,
                                                        dict(frame=dict(duration=50, redraw=True),
                                                             fromcurrent=True,
                                                             transition=dict(duration=0))
                                                       ]
                                                 )
                                            ]
                                   )
                             ])

pyo.iplot(fig)


data=data_save
y = data["diagnosis"]
x = data.drop(["diagnosis"],axis=1)

# la PCA ha bisogno della standardizzazione dei dati
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Costruzione PCA
pca = PCA(n_components = 2)
pca.fit(x_scaled)
X_reduced_pca = pca.transform(x_scaled)

pca_data = pd.DataFrame(X_reduced_pca,columns=["p1","p2"])
pca_data["diagnosis"] = y
hue =pca_data["diagnosis"]
data = [go.Scatter(x = pca_data.p1,
                   y = pca_data.p2,
                   mode = 'markers',
                   marker=dict(
                           size=12,
                           color=hue,
                           symbol="pentagon",
                           line=dict(width=2)
                           ))]

layout = go.Layout(title="PCA",
                   xaxis=dict(title="p1"),
                   yaxis=dict(title="p2"),
                   hovermode="closest")
fig = go.Figure(data=data,layout=layout)
pyo.iplot(fig)


data = data_save

data_bening = data[data["diagnosis"] == 0]
data_bening.drop('diagnosis', inplace=True, axis=1)
list_column = data_bening.columns

for element in list_column:
  print("\n" + element.upper())
  data_bening = data[data["diagnosis"] == 0]
  desc = data_bening[element].describe()
  Q1 = desc[4]
  Q3 = desc[6]
  IQR = Q3-Q1
  lower_bound = Q1 - 1.5*IQR
  upper_bound = Q3 + 1.5*IQR
  print("\nPer i tumori benigni, ogni valore fuori da questo range è un outlier: (", lower_bound ,",", upper_bound,")")
  data_bening[data_bening[element] < lower_bound][element]
  print("Outliers: ", data_bening[(data_bening[element] < lower_bound) | (data_bening[element] > upper_bound)][element].values)

data_malignant = data[data["diagnosis"] == 0]
data_malignant.drop('diagnosis', inplace=True, axis=1)
data_malignant.head()
list_column = data_malignant.columns

for element in list_column:
  print("\n" + element.upper())
  data_malignat = data[data["diagnosis"] == 0]
  desc = data_malignant[element].describe()
  Q1 = desc[4]
  Q3 = desc[6]
  IQR = Q3-Q1
  lower_bound = Q1 - 1.5*IQR
  upper_bound = Q3 + 1.5*IQR
  print("\nPer i tumori maligni, ogni valore fuori da questo range è un outlier: (", lower_bound ,",", upper_bound,")")
  data_malignant[data_malignant[element] < lower_bound][element]
  print("Outliers: ", data_malignant[(data_malignant[element] < lower_bound) | (data_malignant[element] > upper_bound)][element].values)

from sklearn.cluster import DBSCAN

# Apply DBSCAN to the PCA data
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X_reduced_pca)

# Get the outliers identified by DBSCAN
outliers = pca_data[dbscan.labels_ == -1]

# Visualize the PCA with outliers highlighted
data = [go.Scatter(x = pca_data.p1,
                   y = pca_data.p2,
                   mode = 'markers',
                   marker=dict(
                           size=12,
                           color=hue,
                           symbol="pentagon",
                           line=dict(width=2)
                           )),
        go.Scatter(x = outliers.p1,
                   y = outliers.p2,
                   mode = 'markers',
                   marker=dict(
                           size=3,
                           color='red',
                           symbol="circle",
                           line=dict(width=2)
                           ))]

layout = go.Layout(title="PCA with Outliers Detected by DBSCAN",
                   xaxis=dict(title="p1"),
                   yaxis=dict(title="p2"),
                   hovermode="closest")

fig = go.Figure(data=data,layout=layout)
pyo.iplot(fig)


data = data_final

data_bening = data[data["diagnosis"] == 0]
data_bening.drop('diagnosis', inplace=True, axis=1)
list_column = data_bening.columns

for element in list_column:
  melted_data = pd.melt(data,id_vars = "diagnosis",value_vars = element)
  plt.figure(figsize = (7,7))
  sns.boxplot(x = "variable" , y = "value", hue="diagnosis",data = melted_data)
  plt.show()

y = data["diagnosis"]
x = data.drop(["diagnosis"],axis=1)

data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y, data_n_2], axis=1)

data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data, split=True, inner="quart")
plt.xticks(rotation=90)

data = data_final

from scipy.stats import ttest_ind

# Dividi il dataframe in due gruppi sulla base della variabile dicotomica (0 o 1)
gruppo_0 = data.loc[data['diagnosis'] == 0]
gruppo_1 = data.loc[data['diagnosis'] == 1]

# Seleziona le colonne del dataframe che non contengono valori dicotomici
colonne = [col for col in data.columns if col != 'diagnosis']

# Calcola la dimensione dell'effetto di Cohen per ogni colonna del dataframe
for col in colonne:
    diff_media = gruppo_0[col].mean() - gruppo_1[col].mean()
    std_com = (gruppo_0[col].std() + gruppo_1[col].std()) / 2
    d = abs(diff_media / std_com)
    print("Dimensione dell'effetto di Cohen per la feature", col, ":", d)

data = data_final

X = data.drop(['diagnosis'],axis=1)
y = data['diagnosis']

# Dividi i dati in training e test set
X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print(f"Train set:\n{y_train.value_counts()}")

print(f"Test set:\n{y_test.value_counts()}")

# Applica la standardizzazione ai dati (Feature scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crea il modello di KNN
knn = KNeighborsClassifier()

# Definisci una griglia di valori per i parametri da testare
param_grid = {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

# Crea un oggetto GridSearchCV per la ricerca dei migliori iperparametri
grid_search = GridSearchCV(knn, param_grid, cv=5)

# Esegui la ricerca dei migliori iperparametri sui dati di addestramento
grid_search.fit(X_train_scaled, y_  train)

# Stampa i migliori iperparametri trovati
print("Iperparametri ottimali:\n", grid_search.best_params_)

# Crea il modello di KNN con i migliori iperparametri trovati
knn = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto')

# Addestramento del modello
model4 = knn.fit(X_train_scaled, y_train)

# Previsione delle etichette delle classi per i dati di test
prediction4 = model4.predict(X_test_scaled)

cm4 = confusion_matrix(y_test, prediction4)
sns.heatmap(cm4,annot=True)

TP=cm4[0][0]
TN=cm4[1][1]
FN=cm4[1][0]
FP=cm4[0][1]
acc4 = (TP+TN)/(TP+TN+FN+FP)
print('Testing Accuracy:',acc4)

# Stampa il classification report
print(classification_report(y_test, prediction4)) #prediction?

# Definisci i possibili valori degli iperparametri da testare
param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [0.1, 1, 10, 100],
              'kernel': ['linear', 'rbf', 'poly']}

# Crea un oggetto GridSearchCV per la ricerca degli iperparametri
grid_search = GridSearchCV(SVC(), param_grid, cv=5)

# Addestra il modello sul train set
grid_search.fit(X_train_scaled, y_train)

# Stampa i migliori iperparametri trovati
print("Iperparametri ottimali:\n", grid_search.best_params_)

# Crea il modello di SVM con i migliori iperparametri
svm = SVC(C=1, gamma=0.1, kernel='rbf')

# Addestrare il modello utilizzando il set di addestramento
model5 = svm.fit(X_train_scaled, y_train)

# Valutare le prestazioni del modello utilizzando il set di test
prediction5 = model5.predict(X_test_scaled)

cm5 = confusion_matrix(y_test, prediction5)
sns.heatmap(cm5,annot=True)

TP=cm5[0][0]
TN=cm5[1][1]
FN=cm5[1][0]
FP=cm5[0][1]
acc5 = (TP+TN)/(TP+TN+FN+FP)
print('Testing Accuracy:',acc5)

# Stampa la tabella di valutazione
print(classification_report(y_test, prediction5))

# definire i parametri da esplorare nella grid search
params = {'penalty': ['l1', 'l2'], 'C': [0.1, 1, 10], 'tol': [1e-3, 1e-4],
          'solver': ['lbfgs', 'liblinear', 'saga']}

# creare un'istanza del modello di regressione logistica
logreg=LogisticRegression(max_iter = 1000)

# creare un'istanza di GridSearchCV con il modello e i parametri
grid_search = GridSearchCV(logreg, params, cv=5)

# eseguire la grid search sul set di dati di addestramento
grid_search.fit(X_train_scaled, y_train)

# stampare i parametri ottimali e lo score di validazione incrociata
print("I parametri ottimali sono:", grid_search.best_params_)
print("Lo score di validazione incrociata è:", grid_search.best_score_)

# istanzia un nuovo oggetto di LogisticRegression con i parametri ottimali
logreg = LogisticRegression(penalty='l2', C=10, solver='lbfgs', tol=0.001)

# addestra il modello sul set di addestramento
model1 = logreg.fit(X_train_scaled,y_train)
prediction1 = model1.predict(X_test_scaled)

from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(y_test,prediction1)
sns.heatmap(cm1,annot=True)

TP=cm1[0][0]
TN=cm1[1][1]
FN=cm1[1][0]
FP=cm1[0][1]
acc1 = (TP+TN)/(TP+TN+FN+FP)
print('Testing Accuracy:',acc1)

# Stampa il classification report
print(classification_report(y_test, prediction1))

# Crea un oggetto DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42,max_depth=3, min_samples_split=10)

# definisce la griglia di valori degli iperparametri
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# esegue la grid search
grid_search = GridSearchCV(dt, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# stampa i migliori iperparametri trovati
print(grid_search.best_params_)


# crea un nuovo oggetto DecisionTreeClassifier con i migliori iperparametri trovati
dt = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=2, min_samples_leaf=4)

# Addestra l'albero decisionale sul set di training
model2 = dt.fit(X_train_scaled, y_train)

# Effettua le previsioni sul set di test
prediction2 = model2.predict(X_test_scaled)

cm2 = confusion_matrix(y_test, prediction2)
sns.heatmap(cm2,annot=True)


TP=cm2[0][0]
TN=cm2[1][1]
FN=cm2[1][0]
FP=cm2[0][1]
acc2 = (TP+TN)/(TP+TN+FN+FP)
print('Testing Accuracy:',acc2)


# Stampa la tabella di valutazione
print(classification_report(y_test, prediction2))


# definisci i parametri da testare nella grid search
param_grid = {
    'n_estimators': [10, 50, 100, 500],
    'max_depth': [None, 2, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None]
}

# crea un'istanza di Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# crea un'istanza di Grid Search con cross-validation
grid_search = GridSearchCV(rf, param_grid=param_grid, cv=5, n_jobs=-1)

# esegui la grid search sui dati di addestramento
grid_search.fit(X_train_scaled, y_train)

# stampa i parametri ottimali trovati dalla grid search
print("Iperparametri ottimali:\n", grid_search.best_params_)


# Crea il modello di Random Forest con 100 alberi
rf = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, min_samples_leaf=2, max_features='auto', random_state=42)

# Addestra il modello sul train set
model3 = rf.fit(X_train_scaled, y_train)

# Fai le predizioni sul test set
prediction3 = model3.predict(X_test_scaled)

cm3 = confusion_matrix(y_test, prediction3)
sns.heatmap(cm3,annot=True)

TP=cm3[0][0]
TN=cm3[1][1]
FN=cm3[1][0]
FP=cm3[0][1]
acc3 = (TP+TN)/(TP+TN+FN+FP)
print('Testing Accuracy:',acc3)

# Stampa il classification report
print(classification_report(y_test, prediction3))

# Definisci il modello di Gradient Boosting
gb = GradientBoostingClassifier()

# Definisci i valori dei parametri da testare nella Grid Search
params = {'learning_rate': [0.05, 0.1, 0.2],
          'n_estimators': [50, 100, 200],
          'max_depth': [2, 3, 4]}

# Esegui la Grid Search
grid_search = GridSearchCV(gb, params, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

# Stampa i migliori parametri trovati
print("Iperparametri ottimali:\n", grid_search.best_params_)

# Crea il modello di Gradient Boosting con gli iperparametri ottimali
gb = GradientBoostingClassifier(learning_rate=0.2, max_depth=4, n_estimators=50, random_state=42)

# Addestra il modello sul train set
model7 = gb.fit(X_train_scaled, y_train)

# Fai le predizioni sul test set
prediction7 = model7.predict(X_test_scaled)

cm7 = confusion_matrix(y_test, prediction7)
sns.heatmap(cm7,annot=True)

TP=cm7[0][0]
TN=cm7[1][1]
FN=cm7[1][0]
FP=cm7[0][1]
acc7 = (TP+TN)/(TP+TN+FN+FP)
print('Testing Accuracy:',acc7)

# Stampa il classification report
print(classification_report(y_test, prediction7))


# Crea il modello di Naive Bayes Gaussiano
gnb = GaussianNB()

# Definisci la griglia di parametri da testare
param_grid = {
    'priors': [None, [0.25, 0.75], [0.4, 0.6]],
    'var_smoothing': [1e-9, 1e-7, 1e-5]
}

# Crea l'oggetto GridSearchCV
grid = GridSearchCV(gnb, param_grid, cv=5)

# Addestra il modello tramite la grid search
grid.fit(X_train_scaled, y_train)

# Stampa i migliori parametri trovati
print("Iperparametri ottimali:\n", grid.best_params_)


# Creazione del classificatore Naive Bayes
gnb = GaussianNB(priors=[0.4, 0.6], var_smoothing=1e-9)

# Addestramento del classificatore
model6 = gnb.fit(X_train_scaled, y_train)

# Valutazione del classificatore sul test set
prediction6 = model6.predict(X_test_scaled)

cm6 = confusion_matrix(y_test, prediction6)
sns.heatmap(cm6,annot=True)

TP=cm6[0][0]
TN=cm6[1][1]
FN=cm6[1][0]
FP=cm6[0][1]
acc6 = (TP+TN)/(TP+TN+FN+FP)
print('Testing Accuracy:',acc6)

# Stampa il classification report
print(classification_report(y_test, prediction6))

# funzione per creare il modello della rete neurale
def create_model(optimizer='adam', activation='relu', neurons=1):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train_scaled.shape[1], activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# creazione del modello KerasClassifier
model = KerasClassifier(build_fn=create_model, verbose=0)

# definizione dei parametri per la GridSearch
optimizers = ['adam', 'sgd']
activations = ['relu', 'sigmoid', 'tanh']
neurons = [5, 10, 15, 20]
epochs = [50, 100, 150]
batch_size = [16, 32, 64]
dropout_rate = [0.0, 0.2, 0.4, 0.6, 0.8]
learning_rate = [0.001, 0.01, 0.1]

param_grid = dict(optimizer=optimizers, activation=activations, neurons=neurons)

# definizione della GridSearch
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)

# addestramento della GridSearch sul train set
grid_result = grid.fit(X_train_scaled, y_train)

# stampa dei risultati
print("Iperparametri ottimali:\n", grid_result.best_params_)


# Crea il modello
model = Sequential()
model.add(Dense(20, input_dim=X_train_scaled.shape[1], activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# Compila il modello
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Addestra il modello
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test))

# Valuta il modello sul test set
test_loss, test_acc = model.evaluate(X_test_scaled, y_test)

# Calcola i valori di probabilità di classe positiva per la rete neurale (per la curva ROC)
y_score_nn = model.predict(X_test_scaled)


training_loss=history.history["loss"]
test_loss=history.history["val_loss"]
epoch_count=range(1,len(training_loss)+1)
plt.plot(epoch_count, training_loss, "r--")
plt.plot(epoch_count, test_loss, "b-")
plt.legend(["Training loss","Test loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
print(history.history.keys())


training_acc=history.history["accuracy"]
test_accu=history.history["val_accuracy"]
plt.plot(epoch_count, training_acc, "r--")
plt.plot(epoch_count, test_accu, "b-")
plt.legend(["Training accuracy","Test accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy Score")
plt.show()


print('Test accuracy:', test_acc)


# definisci una lista di tuple con il nome del modello e il modello stesso
models = [('Modello Logit', model1), ('Albero decisionale', model2), ('Random Forest', model3), ('KNN', model4), ('SVM', model5), ('Naive Bayes Gaussiano', model6), ('Gradient Boosting', model7)]

# crea una figura vuota per la curva ROC
fig, ax = plt.subplots()

# per ogni modello nella lista
for name, model in models:
    # addestra il modello sul set di addestramento
    model.fit(X_train_scaled, y_train)

    # controlla se il modello ha il metodo predict_proba()
    if hasattr(model, "predict_proba"):
        # ottieni i valori di probabilità di classe positiva per le istanze di test
        y_score = model.predict_proba(X_test_scaled)[:, 1]
    else:
        # altrimenti, ottieni i valori delle previsioni
        y_score = model.predict(X_test_scaled)

    # calcola la curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=1)

    # calcola l'area sotto la curva ROC (AUC)
    roc_auc = auc(fpr, tpr)

    # valuta il modello sul set di test
    accuracy = model.score(X_test_scaled, y_test)

    # aggiungi la curva ROC al grafico
    ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# Calcola la curva ROC per la rete neurale
fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_test, y_score_nn, pos_label=1)

# calcola l'area sotto la curva ROC
roc_auc = auc(fpr_nn, tpr_nn)

# plotta la curva ROC
plt.plot(fpr, tpr, label=f"Neural Network (AUC = {roc_auc:.2f})", linewidth=2)

# aggiungi la legenda
plt.legend(loc="lower right")

# aggiungi la linea di riferimento per il caso casuale
ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')

# personalizza il grafico
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc='lower right')

# mostra il grafico
plt.show()


# Grafico a linee con le accuracy dei modelli
x_labels = ['LOG_REG', 'DT','RF', 'KNN', 'SVM', 'GNB', 'GB', 'NN']
y_values = [acc1,acc2,acc3,acc4,acc5,acc6,acc7,test_acc]

acc_modelli = dict(zip(x_labels, y_values))
acc_modelli = dict(sorted(acc_modelli.items(), key=lambda item: item[1], reverse=False))
for a,b in acc_modelli.items():
    plt.text(a, b, str(round(b,3)), fontsize=12, color='dodgerblue', horizontalalignment='right', verticalalignment='bottom')

plt.plot(acc_modelli.keys(), acc_modelli.values(), marker='.', markerfacecolor='dodgerblue', markersize=12, linewidth=4)
plt.xlabel('Modelli')
plt.ylabel('Accuracy')
plt.title('Accuracy dei modelli')
plt.legend(['Modelli'], loc='lower right')
plt.show()




