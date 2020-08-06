import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
import collections
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
import scipy.cluster.hierarchy as sch
sns.set_style('whitegrid')

def find(vector, value):
    founded=False
    for i in range(len(vector)):
        if vector[i]['label']==value:
            vector[i]['count']+=1
            founded=True
            break
    if founded==False:
        vector.append({'label':value,'count':1})

dataset = pd.read_csv('Data/student/turkiye-student.csv')
#Imprimo las primeras 5 instancias del conjunto de datos
#print(dataset.head())
#Obtenemos la informacion general de cada atributo
"""attributes_info=dataset.describe(include = "all")
print("INFORMACIÓN DE LOS ATRIBUTOS")
for instance in attributes_info:
    print('\t\t\t',instance)
    print(attributes_info[instance])"""

"""
Analizamos los cursos más respondidos por los estudiantes(EN ESTE CASO ES EL CURSO 3)
plt.figure(figsize=(20, 6))
sns.countplot(x='class', data=dataset)
plt.show()
"""

"""
Analizamos las respuestas dadas a cada prugunta por parte de los estudiantes
plt.figure(figsize=(20, 20))
sns.boxplot(data=dataset.iloc[:,5:33]);
plt.show()
"""

"""
# Calculate mean for each question response for all the classes, a fin de 
obtener las clases que mas prefieren los estudiantes
questionmeans = []
classlist = []
questions = []
totalplotdata = pd.DataFrame(list(zip(classlist, questions, questionmeans))
                             , columns=['class', 'questions', 'mean'])
for class_num in range(1, 13):
    class_data = dataset[(dataset["class"] == class_num)]

    questionmeans = []
    classlist = []
    questions = []

    for num in range(1, 13):
        questions.append(num)
    # Class related questions are from Q1 to Q12
    for col in range(5, 17):
        questionmeans.append(class_data.iloc[:, col].mean())
    classlist += 12 * [class_num]
    print(classlist)
    plotdata = pd.DataFrame(list(zip(classlist, questions, questionmeans))
                            , columns=['class', 'questions', 'mean'])
    totalplotdata = totalplotdata.append(plotdata, ignore_index=True)

plt.figure(figsize=(20, 10))
sns.pointplot(x="questions", y="mean", data=totalplotdata, hue="class")
plt.show()
"""

"""
# Calculate mean for each question response for all the classes, a fin de obtener cual es el instructor que mas prefieren
los estudiantes

questionmeans = []
inslist = []
questions = []
totalplotdata = pd.DataFrame(list(zip(inslist, questions, questionmeans))
                             , columns=['ins', 'questions', 'mean'])
for ins_num in range(1, 4):
    ins_data = dataset[(dataset["instr"] == ins_num)]
    questionmeans = []
    inslist = []
    questions = []

    for num in range(13, 29):
        questions.append(num)

    for col in range(17, 33):
        questionmeans.append(ins_data.iloc[:, col].mean())
    inslist += 16 * [ins_num]
    plotdata = pd.DataFrame(list(zip(inslist, questions, questionmeans))
                            , columns=['ins', 'questions', 'mean'])
    totalplotdata = totalplotdata.append(plotdata, ignore_index=True)

plt.figure(figsize=(20, 5))
sns.pointplot(x="questions", y="mean", data=totalplotdata, hue="ins")
plt.show()
"""

#Obtengo la informacion del cuestionario
dataset_questions = dataset.iloc[:,5:33]
#Obtengo la informacion de las primeras 5 instancias
#print(dataset_questions.head())

#Analisis de correlacion
correlation_matrix=dataset_questions.corr()
#print("Matriz de correlacion")
#print(correlation_matrix)
df=pd.DataFrame(correlation_matrix)
df.to_csv('./Data/student/correlacion.csv', index=False)

scaled_data=preprocessing.scale(dataset_questions)

pca = PCA(n_components = 2)
dataset_questions_pca = pca.fit_transform(scaled_data)

#Aplico el metodo del codo sobre el conjunto de datos
wcss = []
for i in range(1, 7):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(dataset_questions_pca)
    #Guardo la suma del error cuadrado
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 7), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
"""
for i in range(2, 7):
    print("VALOR DE K=",i)
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    cluster_labels=kmeans.fit_predict(dataset_questions_pca)
    silhouette_avg = metrics.silhouette_score(dataset_questions_pca, cluster_labels)
    print ('El coeficiente de silueta del agrupamiento es = ', silhouette_avg)
"""

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

"""
# Using the dendrogram to find the optimal number of clusters, se obtiene para k=2
dendrogram = sch.dendrogram(sch.linkage(dataset_questions_pca, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('questions')
plt.ylabel('Euclidean distances')
plt.show()


# Fitting Hierarchical Clustering to the dataset
hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(dataset_questions_pca)
X = dataset_questions_pca

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'yellow', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'red', label = 'Cluster 2')
plt.title('Clusters of STUDENTS')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()

print('Conteo clustering jerarquico',collections.Counter(y_hc))
"""