import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

def find(vector, value):
    founded=False
    for i in range(len(vector)):
        if vector[i]['label']==value:
            vector[i]['count']+=1
            founded=True
            break
    if founded==False:
        vector.append({'label':value,'count':1})

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

dataset = pd.read_csv('./Data/seed.csv')
#Imprimimos las primeras 5 instancias del conjunto de datos
print(dataset.head())
#Obtenemos la informacion general de cada atributo
attributes_info=dataset.describe(include = "all")
print("INFORMACIÓN DE LOS ATRIBUTOS")
for instance in attributes_info:
    print('\t\t\t',instance)
    print(attributes_info[instance])

#Definimos las caracteristicas que son los primeros 7 atributos(columnas) y sus instancias
features = dataset.iloc[:, 0:7]
features_aux=features.values
#Imprimiendo las instancias de las variables de entrada
print(features)
#Definimos el atributo que es el último atributo
target = dataset.iloc[:, -1]
#Imprimiendo las instancias de la variable destino
print(target)

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10))
#Se normalizan los datos en el visualizador
visualizer.fit(features)
#Se muestran los datos en el visualizador del metodo del codo
visualizer.show()

kmeans = KMeans(n_clusters=3,
                  init='k-means++',
                  max_iter=300,
                  n_init=10,
                  random_state=0)

kmeans.fit(features)
cluster_labels = kmeans.fit_predict(features)
initial_centroids=kmeans.cluster_centers_
print("Centroides iniciales")
for instance in initial_centroids:
    print(instance)

silhouette_avg = metrics.silhouette_score(features, cluster_labels)
print ('El coeficiente de silueta del agrupamiento es = ', silhouette_avg)
purity = purity_score(target, cluster_labels)
print ('Pureza del clustering realizado = ', purity)

points_summarize=[]
for obj in range(len(cluster_labels)):
    print('Instancia '+(str(obj+1)), 'Cluster',cluster_labels[obj])
    find(points_summarize,cluster_labels[obj])
print("Contador de clusters",points_summarize)

plt.scatter(features_aux[cluster_labels==0,0],features_aux[cluster_labels==0,1],c='red',label='Grupo I')
#plt.scatter(features_aux[cluster_labels==1,0],features_aux[cluster_labels==1,1],c='blue',label='Grupo II')
#plt.scatter(features_aux[cluster_labels==2,0],features_aux[cluster_labels==2,1],c='green',label='Grupo III')
plt.scatter(initial_centroids[:,0],initial_centroids[:,1],c='yellow',label='Centroides')

plt.title("KMeans Clustering")
plt.xlabel("A")
plt.ylabel("B")
plt.legend()
plt.show()