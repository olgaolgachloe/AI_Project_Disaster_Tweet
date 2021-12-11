import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re
import numpy as np
import string

model_path = "../model"

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

st.write("# Disaster Tweet Detection Engine")

twitter_text = st.text_input("Enter a tweet text for disaster detection",
                             placeholder="40 displaced by ocean township apartment fire new york")


def clean_input(text):
    # lower case
    text = text.lower()

    # remove url
    text = re.sub('https?://.+', '', text)

    # punctuation
    text = re.sub(r'\[.*?.\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)

    # remove ending […]
    text = re.sub('[…“”’]', '', text)

    # remove leading or ending space
    text = text.strip()
    return text


def model_detect(text):
    text = clean_input(text)
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    predictions = outputs.logits.detach().numpy()[0]
    probabilities = outputs.logits.softmax(dim=-1).tolist()
    prediction = np.argmax(predictions)
    probability = probabilities[0][prediction]
    return prediction, probability


if st.button('Detect'):
    if twitter_text.strip() == "":
        st.write("Please enter a valid input")
    else:
        prediction, probability = model_detect(twitter_text)
        prediction_label = "Yes, It's a real disaster tweet" if prediction == 1 \
            else "No, It's not a real disaster tweet"
        st.write({"prediction": prediction_label, "probability": probability})
