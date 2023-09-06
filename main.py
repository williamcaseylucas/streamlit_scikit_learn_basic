import streamlit as st
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import matplotlib.pyplot as plt

st.title("Streamlit example")
st.write("""
         # Explore different classifier
         Which one is the best?
         """)

# dataset_name = st.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine Dataset"))
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine Dataset", "Diabetes Dataset"))
# st.write("selected dataset:", dataset_name)
classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest", "Decision Tree"))

def get_dataset(dataset_name): 
  if dataset_name == 'Iris':
    data = datasets.load_iris()
  elif dataset_name == 'Breast Cancer':
    data = datasets.load_breast_cancer()
  elif dataset_name == 'Diabetes Dataset':
    data = datasets.load_diabetes()
  else:
    data = datasets.load_wine()
  X = data.data
  y = data.target
  return X, y

X, y = get_dataset(dataset_name)
st.write("Shape of dataset: ", X.shape)
st.write("Number of classes: ", len(np.unique(y)))

def add_parameter_ui(clf_name): 
  params = dict()
  if clf_name == 'KNN':
    K = st.sidebar.slider("K", 1, 15)
    params["K"] = K
  elif clf_name == 'SVM':
    C = st.sidebar.slider("C", 0.01, 10.0)
    params["C"] = C
  elif clf_name == 'Decision Tree':
    max_depth = st.sidebar.slider("Max Depth", 1, 10)
    random_state = st.sidebar.slider("Random State", 1, 10)
    params["max_depth"] = int(max_depth)
    params["random_state"] = int(random_state)
  else:
    max_depth = st.sidebar.slider("max_depth", 2, 15)
    n_estimators = st.sidebar.slider("n_estimators", 1, 50)
    params["max_depth"] = max_depth
    params["n_estimators"] = n_estimators
  return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
  clf = None
  if clf_name == 'KNN':
    clf = KNeighborsClassifier(n_neighbors=params["K"])
  elif clf_name == 'SVM':
    clf = SVC(C=params["C"])
  elif clf_name == 'Decision Tree':
    clf = DecisionTreeClassifier(max_depth = params['max_depth'], random_state = params['random_state'])
  else:
    clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=1234)
  return clf

clf = get_classifier(classifier_name, params)

# Classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"classifier = {classifier_name}")
st.write(f"accuracy = {acc}")

# Plot
# pca = PCA(2) # 2 components
# X_projected = pca.fit_transform(X)
# x1 = X_projected[:, 0]
# x2 = X_projected[:, 1]

pca = PCA(4) # 2 components
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]
x3 = X_projected[:, 2]
x4 = X_projected[:, 3]


fig = plt.figure()
# plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.scatter(x1, x2, alpha=0.8, c=y, cmap="viridis")
plt.scatter(x3, x4, alpha=0.8, c=y, cmap="viridis")
# plt.scatter(x1, x2, alpha=0.8, color='blue')
# plt.scatter(x3, x4, alpha=0.8, color='red')

plt.xlabel("Principal Component 1 and 3")
plt.ylabel("Principal Component 2 and 4")
plt.colorbar()

# plt.show()
st.pyplot(fig)

# TODO
# - add more parameters (sklearn)
# - add other classifiers
# - add feature scaling

import graphviz
if classifier_name == "Decision Tree":
  st.title("Decision Tree Visualization")
  graph = export_graphviz(clf, out_file=None, filled=True, special_characters=True)
  st.graphviz_chart(graph)

if classifier_name == "Random Forest":
    st.write("Random Forest consists of multiple Decision Trees.")
    st.write(f"Number of trees in the Random Forest: {len(clf.estimators_)}")
    
    num_columns = st.sidebar.slider('Select Number of Columns', 1, 5)
    
    # data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    col = st.columns(num_columns)

    for idx, tree in enumerate(clf.estimators_):
        with col[idx % num_columns]:
          graph = export_graphviz(tree, out_file=None, 
                                    filled=True, rounded=True, 
                                    special_characters=True)
          st.subheader(f"Decision Tree {idx + 1}")
          st.graphviz_chart(graph)