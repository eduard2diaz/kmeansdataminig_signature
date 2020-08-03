import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

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
    print("Matriz de contingencia")
    print(contingency_matrix)
    numerador=np.sum(np.amax(contingency_matrix, axis=0))
    denominador=np.sum(contingency_matrix)
    #print('Numerado',numerador,'Denominador',denominador)
    return numerador / denominador

dataset = pd.read_csv('Data/semilla/seed.csv')
#Imprimimos las primeras 5 instancias del conjunto de datos
#print(dataset.head())
#Obtenemos la informacion general de cada atributo
"""attributes_info=dataset.describe(include = "all")
print("INFORMACIÓN DE LOS ATRIBUTOS")
for instance in attributes_info:
    print('\t\t\t',instance)
    print(attributes_info[instance])
"""
#Definimos las caracteristicas que son los primeros 7 atributos(columnas) y sus instancias
features = dataset.iloc[:, 0:7]
features_aux=features.values
#Imprimiendo las instancias de las variables de entrada
print(features)
#Definimos el atributo que es el último atributo
target = dataset.iloc[:, -1]
#Imprimiendo las instancias de la variable destino
print(target)

#Analisis de correlacion
correlation_matrix=features.corr()
print("Matriz de correlacion")
print(correlation_matrix)
df=pd.DataFrame(correlation_matrix)
df.to_csv('./Data/semilla/correlacion.csv', index=False)

#PCA
#dicha funcion scale lo que hace es centrar y escalar los datos
scaled_data=preprocessing.scale(features)

pca=PCA()
#Aqui se hacen los calculos de PCA
pca.fit(scaled_data)
#Aqui generamos las coordenadas para una grafica de PCA basado en los datos escalados
pca_data=pca.transform(scaled_data)
"""Cada nuevo atributo es una combinacion lineal de los atributos originales. PCA permite describir un conjunto
de datos en termino de nuevas variables no correlacionadas. Dichos componentes se ordenan por la cantidad
de varianza original que describen por la cantidad de varianza original que describen
"""
#calculamos el porcentaje de variacion de cada componente principal
per_var=np.round(pca.explained_variance_ratio_*100,decimals=1)
#creamos los labels, uno por cada compoenente principal
labels=['PC'+str(x) for x in range(1, len(per_var)+1)]
plt.bar(x=range(1,len(per_var)+1),height=per_var,tick_label=labels)
plt.ylabel('Porcentaje de varianza explicada')
plt.xlabel('Componente principal')
plt.title('Screen plot')
plt.show()
#imprimiendo el porcentaje de varianza explicada por cada componente
for x in range(1,len(per_var)):
    print("PC"+str(x), "Percentage", per_var[x-1])

#Planteamos los datos como la relacion lineal de solamente dos componentes
pca = PCA(n_components = 2)
dataset_questions_pca = pca.fit_transform(features)

#Analizamos la cantidad de cluster a partir de la informacion obtenida de la relacion lineal del pca
#Aplico el metodo del codo sobre el conjunto de datos
wcss = []
for i in range(1, 7):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(dataset_questions_pca)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 7), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Analizamos la cantidad de cluster a partir de la informacion original
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10))
#Se normalizan los datos en el visualizador
visualizer.fit(features)
#Se muestran los datos en el visualizador del metodo del codo
visualizer.show()
#Aplicamos K-Means sobre el conjunto de datos original para 3 clusters
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
    #print('Instancia '+(str(obj+1)), 'Cluster',cluster_labels[obj])
    find(points_summarize,cluster_labels[obj])
print("Contador de clusters",points_summarize)

plt.scatter(features_aux[cluster_labels==0,0],features_aux[cluster_labels==0,1],c='red',label='Cluster I')
plt.scatter(features_aux[cluster_labels==1,0],features_aux[cluster_labels==1,1],c='blue',label='Cluster II')
plt.scatter(features_aux[cluster_labels==2,0],features_aux[cluster_labels==2,1],c='green',label='Cluster III')
plt.scatter(initial_centroids[:,0],initial_centroids[:,1],c='yellow',label='Centroides')

plt.title("KMeans Clustering")
plt.xlabel("A")
plt.ylabel("B")
plt.legend()
plt.show()

#Aplico k-means sobre el conjunto brindado por pca
kmeans = KMeans(n_clusters = 3, init = 'k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(dataset_questions_pca)
initial_centroids=kmeans.cluster_centers_
points_summarize=[]
for obj in range(len(y_kmeans)):
    #print('Instancia '+(str(obj+1)), 'Cluster',y_kmeans[obj])
    find(points_summarize,y_kmeans[obj])
print("Contador de clusters",points_summarize)

distintos=0
for obj in range(len(y_kmeans)):
    if y_kmeans[obj]!=cluster_labels[obj]:
        distintos+=1
print("Total de elementos distintos de cada clustering",distintos);

print("Centroides iniciales")
for instance in initial_centroids:
    print(instance)

plt.scatter(dataset_questions_pca[y_kmeans == 0, 0], dataset_questions_pca[y_kmeans == 0, 1], c = 'red', label = 'Cluster 1')
plt.scatter(dataset_questions_pca[y_kmeans == 1, 0], dataset_questions_pca[y_kmeans == 1, 1], c = 'blue', label = 'Cluster 2')
plt.scatter(dataset_questions_pca[y_kmeans == 2, 0], dataset_questions_pca[y_kmeans == 2, 1], c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c = 'yellow', label = 'Centroides')
plt.title('Clusters of semillas')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()