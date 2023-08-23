import streamlit as st
import numpy as np
import pandas as pd

#文字顯示
st.title("AAAAAAAA")
st.header("BBBBBBBBB")
st.subheader("CcCCCCCCCC")
st.write("# AAAAAAAA")
st.write("## BBBBBBBBB")
st.write("Hello1213131313132")

a = np.array([10,20,30])
st.write(a)

b = pd.DataFrame([[11,22],[33,44]])
st.write(b)

st.write(range(10))
c = 10
#print(c+=10)
st.write(c+10)

# 核取方塊Checkbox
st.write("### 核取方塊Checkbox")
re1 = st.checkbox("Dog")
if re1:
    st.info("你選中狗狗")
else:
    st.info("你沒有選中狗狗")

checks = st.columns(4)
with checks[0]:#針對欄index進行排列
    c1 = st.checkbox("A")
    if c1:
        st.write("C1")#或 st.info("C1")
with checks[1]:
    st.checkbox("B")   
with checks[2]:
    st.checkbox("C")
with checks[3]:
    st.checkbox("D")

# 選項按鈕RadioButton
st.write("### 選項按鈕RadioButton")
gender = st.radio("性別：", ("M", "F", "None"), index=1)
st.info(gender)

st.write("### 選項按鈕RadioButton 2+ 數字輸入框") 
col1, col2 = st.columns(2)
with col1:
    ra1 = st.number_input("請輸入任一整數")
with col2:
    #ra2 = st.number_input("請輸入任一整數")
    ra2 = 100

ra3 = st.radio("計算：", ("+", "-", "*", '/')) 
if ra3=='+':
    st.write("{}+{}={:.2f}".format(ra1, ra2, ra1+ra2))
elif ra3=='-':
    st.write("{}-{}={:.2f}".format(ra1, ra2, ra1-ra2))
elif ra3=='*':
    st.write("{}*{}={:.2f}".format(ra1, ra2, ra1*ra2))
elif ra3=='/':
    st.write("{}/{}={:.2f}".format(ra1, ra2, ra1/ra2))


st.write("### 滑桿Slider")
#slider = st.slider("請選擇數量：", 1.0, 20.0, step=0.1)
slider = st.slider("請選擇數值範圍：", 1.0, 20.0,(12.0, 18.0) ) 
st.info(slider)
st.write("{},{}".format(slider[0], slider[1]))


st.write("### 下拉選單SelectBox:單選")
select1 = st.selectbox("請選擇城市", ("台北",'台中','台南'), index=1)
#st.info(select1)
"選擇城市:", select1


st.write("### 下拉選單MultiSelect:複選")
select2 = st.multiselect("請選擇城市", ("台北",'台中','台南'))
st.info(select2)
st.write(select2)


st.write("### 顯示圖片")
st.image("aa.jpg")


st.write("### 上傳csv")
file = st.file_uploader("請選擇CSV檔")
if file is not None:
    df = pd.read_csv(file, header=None)
    st.dataframe(df)
    st.table(df.iloc[:, [1,2,5]])


st.write("### 隱藏欄位")
hidden = st.expander("點選后向下拉開")
hidden.write("1. ADASFJSLKJFKDSJGK")
hidden.write("2. sgdfsgjhksdfgdfkg")


st.write("### 側邊欄SideBar")
st.sidebar.text("側邊欄SideBar")

st.write("### 按鈕Button")
ok = st.button("確定執行")
if ok:
    st.write("#### OK")

