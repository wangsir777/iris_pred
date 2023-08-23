import streamlit as st
import joblib

#載入model
svm_clf = joblib.load("svm_clf_model.joblib")
knn_clf = joblib.load("knn_clf_model.joblib")
rf_clf = joblib.load("rf_clf_model.joblib")

st.title("鳶尾花品種預測")

clf = st.sidebar.selectbox("#### 請選擇模型：", 
                           ["KNN","SVM","RandomForest"])

s1 = st.slider("花萼長度：",3.0, 8.0, 5.8)
s2 = st.slider("花萼寬度：",2.0, 5.0, 3.5)
s3 = st.slider("花瓣長度：",1.0, 7.0, 4.5)
s4 = st.slider("花瓣寬度：",0.1, 2.6, 1.2)

labels = ['setosa','versicolor','virginica']

if clf == "KNN":
    clf_model = knn_clf
elif clf == "SVM":
    clf_model = svm_clf
else:
    clf_model = rf_clf

if st.button("進行預測"):
    X = [[s1, s2, s3, s4]]
    y = clf_model.predict(X)
    st.write(y[0])
    st.write("### 預測品種是：", labels[y[0]])
