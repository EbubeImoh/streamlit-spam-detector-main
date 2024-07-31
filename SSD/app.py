import packages.data_processor as dp
import streamlit as st 
import joblib
import os
os.chdir("C:\\Users\\amanr\\OneDrive\\Desktop\\streamlit-spam-detector-main\\SSD")

# Load the model
spam_clf = joblib.load(open('models/my_spam_model.pkl','rb'))

# Load vectorizer
vectorizer = joblib.load(open('vectors/my_vectorizer.pickle', 'rb'))

### MAIN FUNCTION ###
def main(title = "Text classification App for demo".upper()):
    st.markdown("<h1 style='text-align: center; font-size: 25px; color: blue;'>{}</h1>".format(title), 
    unsafe_allow_html=True)
    st.image("images\myimage.png",width=100)
    info = ''
    
    with st.expander("1. Ckeck if your text is a spam or not spam 😀"):
        text_message = st.text_input("Please enter your message")
        if st.button("Predict"):
            prediction = spam_clf.predict(vectorizer.transform([text_message]))

            if(prediction[0] == 0):
                info = 'NOT SPAM'

            else:
                info = 'SPAM'
            st.success('Prediction: {}'.format(info))

if __name__ == "__main__":
    main()