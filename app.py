from operator import index
import streamlit as st
import pandas as pd
import os
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup,compare_models,pull,save_model
with st.sidebar:
    st.title("AUTO ML")
    choice=st.radio("navigation",["Upload","Profiling","Model","Download"])
if os.path.exists("source.csv"):
    df=pd.read_csv("source.csv",index_col=None)

if choice=="Upload":
    st.title("upload your data for modeling!")
    file=st.file_uploader("upload your dataset ")
    if file:
        df=pd.read_csv(file,index_col=None)
        df.to_csv("source.csv",index=None)
        st.dataframe(df)
if choice=="Profiling":
    st.title("EXPLORATORY DATA ANALYSIS")
    if st.button("DO PROFILING"):
        profile_report=df.profile_report()
        st_profile_report(profile_report)
if choice=="Model":
    st.title("MODEL BUILDING")
    target=st.selectbox("select The target:",df.columns)
    if st.button("TRAIN MODEL"):
        setup(df,target=target)
        setup_df=pull()
        st.info("ML EXPERIMENT SETTINGS")
        st.dataframe(setup_df)
        best_model=compare_models(include=['lr','svm','dt'])
        compare_df=pull()
        st.info(" ML MODEL COMPARISON ")
        st.dataframe(compare_df)
        best_model
        save_model(best_model,"best_model")
if choice=="Download":
    with open("best_model.pkl",'rb') as f:
        st.download_button("DOWNLOAD THE MODEL FILE",f,"best_model.pkl")
