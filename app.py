from tkinter import Menu
from turtle import color
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

pipe_lr=joblib.load(open("models/emotion_classifier_pipe_lr2.pkl",'rb'))

def predict_emotion(text):
    results=pipe_lr.predict([text])
    return results[0]

def get_probability(text):
    prob_result=pipe_lr.predict_proba([text])
    return prob_result

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

def main():
    st.title("Emotion Classifier App")
    menu=["Home","Monitor","About"]
    choice=st.sidebar.selectbox("Menu",menu)
    if choice=='Home':
        st.subheader("Home- Emotion in Text")
        with st.form(key='ak47'):
            raw_text=st.text_area("Type Here")
            submit_text=st.form_submit_button(label="Submit")
        if submit_text:
            col1,col2=st.columns(2)
            prediction=predict_emotion(raw_text)
            probability=get_probability(raw_text)
            with col1:
                st.success("Original Text")
                st.write(raw_text)
                st.success("Prediction")
                st.write("{}: {}".format(prediction,emotions_emoji_dict[prediction]))
                st.write("Confidence:{}".format(round(np.max(probability),4)))
        
            with col2:
                st.success("Prediction Probability")
                #st.write(probability)
                probab_df=pd.DataFrame(probability,columns=pipe_lr.classes_)
                #st.write(probab_df.T)
                probab_df_clean=probab_df.T.reset_index()
                probab_df_clean.columns=["Emotion","Probability"]
                fig= alt.Chart(probab_df_clean).mark_bar().encode(x="Emotion",y='Probability',color='Emotion')
                st.altair_chart(fig,use_container_width=True)
    elif choice=='Monitor':
        st.subheader("Monitor App")
    
    else:
        st.subheader("About")
    
if __name__=='__main__':
    main()