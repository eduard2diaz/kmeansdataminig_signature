import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import collections
import scipy.cluster.hierarchy as sch
sns.set_style('whitegrid')

dataset = pd.read_csv('./Data/turkiye-student.csv')
#Imprimo las primeras 5 instancias del conjunto de datos
print(dataset.head())
#Obtenemos la informacion general de cada atributo
attributes_info=dataset.describe(include = "all")
print("INFORMACIÓN DE LOS ATRIBUTOS")
for instance in attributes_info:
    print('\t\t\t',instance)
    print(attributes_info[instance])

#Obtengo la informacion del cuestionario
dataset_questions = dataset.iloc[:,5:33]
#Obtengo la informacion de las primeras 5 instancias
print(dataset_questions.head())

#DUDA
pca = PCA(n_components = 2)
dataset_questions_pca = pca.fit_transform(dataset_questions)

#Aplico el metodo del codo sobre el conjunto de datos
wcss = []
for i in range(1, 7):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(dataset_questions_pca)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 7), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 3, init = 'k-means++')
y_kmeans = kmeans.fit_predict(dataset_questions_pca)
plt.scatter(dataset_questions_pca[y_kmeans == 0, 0], dataset_questions_pca[y_kmeans == 0, 1], s = 100, c = 'yellow', label = 'Cluster 1')
plt.scatter(dataset_questions_pca[y_kmeans == 1, 0], dataset_questions_pca[y_kmeans == 1, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(dataset_questions_pca[y_kmeans == 2, 0], dataset_questions_pca[y_kmeans == 2, 1], s = 100, c = 'red', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'blue', label = 'Centroids')
plt.title('Clusters of students')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()

print("Conteo", collections.Counter(y_kmeans))


dendrogram = sch.dendrogram(sch.linkage(dataset_questions_pca, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('questions')
plt.ylabel('Euclidean distances')
plt.show()

print("FINALIZO")