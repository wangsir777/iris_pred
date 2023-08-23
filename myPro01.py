import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("機器學習分類器")
#側邊欄
data = st.sidebar.selectbox(
    "### 請選擇資料集：",
    ['IRIS',"WINE",'CANCER']
)
classifier = st.sidebar.selectbox(
    "### 請選擇分類器：",
    ['SVM',"KNN",'RandomForest']
)
#下載資料集,並取出X與y
def loadData(sets):
    myData = None
    if sets=="IRIS":
        myData = datasets.load_iris()
    elif sets=="WINE":
        myData = datasets.load_wine()
    else:
        myData = datasets.load_breast_cancer()

    X = myData.data
    y = myData.target
    yName = myData.target_names
    return X, y, yName

X, y, yName = loadData(data)
st.write("#### 資料集的結構：", X.shape)
st.write("#### 資料集的分類數：", len(np.unique(y)))
st.write("#### 資料集的分類名稱：")
for i in yName:
    st.write("##### ",i)
st.write("#### 資料集前 5 筆：")
st.write(X[:5])

#定義模型參數
def model(m):
    p={}
    if m=='SVM':
        C = st.sidebar.slider("設定參數C的值：", 0.01, 10.0)
        p['C']=C
    elif m=="KNN":
        K = st.sidebar.slider("設定參數K的值：", 1, 10)
        p['K']=K
    else:
        N = st.sidebar.slider("設定樹的數量：", 10, 500)
        D = st.sidebar.slider("設定樹的分析層數：", 1, 100)
        p['N']=N
        p['D']=D
    return p

#建立模型
ps = model(classifier)
def myModel(clf, p):
    new_clf = None
    if clf=='SVM':
        new_clf = SVC(C=p["C"])
    elif clf=='KNN':
        new_clf = KNeighborsClassifier(n_neighbors=p["K"])
    else:
        new_clf = RandomForestClassifier(n_estimators=p["N"],
                                        max_depth=p["D"],
                                        random_state=123)
    return new_clf

clf = myModel(classifier, ps)

# 分割訓練,測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=123,
                                                    stratify=y)
# 進行訓練計算+預測
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#進行評分
acc = accuracy_score(y_test, y_pred)

st.write("### 分類準確度：", acc)

#降維
pca = PCA(2)
new_X = pca.fit_transform(X)

fig = plt.figure()
plt.scatter(new_X[:,0], new_X[:,1],c=y, alpha=0.7)

#plt.show()
st.pyplot(fig)
