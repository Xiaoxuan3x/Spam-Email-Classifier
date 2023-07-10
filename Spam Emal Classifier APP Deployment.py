import pandas as pd
import streamlit as st
import keras
from PIL import Image
from keras.utils import pad_sequences
import pickle
#from keras_preprocessing.sequence import pad_sequence


##load ann model
with open(r'D:\Users\V\Desktop\Bootcamp\RegClass\Spam_classifier_model.pkl', 'rb') as file:
    model_ann = pickle.load(file)

## load the copy of dataset
df=pd.read_csv('emails.csv')

## set page configuration
st.set_page_config(page_title='Email Classifier', layout='wide')

## add page title and content
st.title('Email Classifier Using Neural Network')
st.write('Please Enter an email to be classified:')

##add image
image=Image.open(r'D:\Users\V\Desktop\Bootcamp\RegClass\ann_model\spam-detector-online.png')
st.image(image,use_column_width=True)


## get user input
email_text=st.text_input("Email Text:")
## convert text to numerical values
word_index={word: index for index,word in enumerate(df.columns[:-1])}
numerical_email=[word_index[word] for word in email_text.lower().split() if word in word_index]


## pad the numerical emalis so that it can have a uniques shapes
padded_email=pad_sequences([numerical_email],maxlen=3000)

## make the prediction
if st.button("Predct"):
    prediction=model_ann.predict(padded_email)
##print the result
## set a threshold of 0.5
    if prediction>0.5:
        st.write('Spam')#1
    else:
        st.write('Not Spam！')#0




