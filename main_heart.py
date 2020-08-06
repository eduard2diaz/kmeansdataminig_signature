import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from fcmeans import FCM
from sklearn.datasets import make_blobs
from seaborn import scatterplot as scatter

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

dataset = pd.read_csv('Data/heart/heart.csv')
#Definimos las caracteristicas que son los primeros 7 atributos(columnas) y sus instancias
features = dataset.iloc[:, 0:12]
features_aux=features.values
#Imprimiendo las instancias de las variables de entrada
#print(features)
#Definimos el atributo que es el último atributo
target = dataset.iloc[:, -1]
#Imprimiendo las instancias de la variable destino
#print(target)
#Analisis de correlacion
correlation_matrix=features.corr()
#print("Matriz de correlacion")
#print(correlation_matrix)
df=pd.DataFrame(correlation_matrix)
df.to_csv('./Data/heart/correlacion.csv', index=False)
#PCA
#dicha funcion scale lo que hace es centrar y escalar los datos
scaled_data=preprocessing.scale(features)

pca=PCA()
#Aqui se hacen los calculos de PCA
pca.fit(scaled_data)
#Aqui generamos las coordenadas para una grafica de PCA basado en los datos escalados
pca_data=pca.transform(scaled_data)

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
for x in range(1,len(per_var)+1):
    print("PC"+str(x), "Percentage", per_var[x-1])


#Planteamos los datos como la relacion lineal de solamente dos componentes
pca = PCA(n_components = 2)
dataset_questions_pca = pca.fit_transform(scaled_data)

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


#Aplico k-means sobre el conjunto brindado por pca
kmeans = KMeans(n_clusters = 2, init = 'k-means++',max_iter=300,n_init=10,random_state=1)
y_kmeans = kmeans.fit_predict(dataset_questions_pca)
initial_centroids=kmeans.cluster_centers_
"""
print("Centroides iniciales")
for instance in initial_centroids:
    print(instance)
"""

silhouette_avg = metrics.silhouette_score(scaled_data, y_kmeans)
print ('El coeficiente de silueta del agrupamiento es = ', silhouette_avg)
purity = purity_score(target, y_kmeans)
print ('Pureza del clustering realizado = ', purity)

plt.scatter(dataset_questions_pca[y_kmeans == 0, 0], dataset_questions_pca[y_kmeans == 0, 1], c = 'red', label = 'Cluster 1')
plt.scatter(dataset_questions_pca[y_kmeans == 1, 0], dataset_questions_pca[y_kmeans == 1, 1], c = 'blue', label = 'Cluster 2')
plt.scatter(dataset_questions_pca[y_kmeans == 2, 0], dataset_questions_pca[y_kmeans == 2, 1], c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c = 'yellow', label = 'Centroides')
plt.title('Clusters of pacientes de insuficiencia cardiaca')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()

"""
points_summarize=[]
for obj in range(len(y_kmeans)):
    #print('Instancia '+(str(obj+1)), 'Cluster',cluster_labels[obj])
    find(points_summarize,y_kmeans[obj])
print("Contador de clusters",points_summarize)


# Realizando clustering a partir de c-means
# fit the fuzzy-c-means
fcm = FCM(n_clusters=2)
fcm.fit(dataset_questions_pca)
result=fcm.predict(dataset_questions_pca)
# outputs
fcm_centers = fcm.centers
fcm_labels  = fcm.u.argmax(axis=1)

silhouette_avg = metrics.silhouette_score(features, result)
print ('El coeficiente de silueta del agrupamiento es = ', silhouette_avg)
purity = purity_score(target, result)
print ('Pureza del clustering realizado = ', purity)
# plot result
f, axes = plt.subplots(1, 2, figsize=(11,5))
scatter(dataset_questions_pca[:,0], dataset_questions_pca[:,1], ax=axes[0])
scatter(dataset_questions_pca[:,0], dataset_questions_pca[:,1], ax=axes[1], hue=fcm_labels)
scatter(fcm_centers[:,0], fcm_centers[:,1], ax=axes[1],marker="s",s=200)
plt.show()
"""