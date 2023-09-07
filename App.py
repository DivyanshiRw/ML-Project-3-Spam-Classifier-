import streamlit as st
import pickle
import string
from nltk.stem import PorterStemmer

def transform_text(Text):
    # Convert text to lowercase
    Text = Text.lower()
    
    # Tokenize the text
    Text = Text.split()  # can use split() for basic tokenization
    
    # Remove non-alphanumeric characters and stopwords, and perform stemming
    ps = PorterStemmer()
    stopwords = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                     "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
                     "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
                     "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                     "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because",
                     "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
                     "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
                     "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where",
                     "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
                     "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
                     "don", "should", "now"])
    
    # Initialize an empty list to store cleaned and stemmed words
    cleaned_words = []
    
    for word in Text:
        # Remove punctuation and check if the word is not a stopword
        word = word.translate(str.maketrans('', '', string.punctuation))
        if word not in string.punctuation and word not in stopwords:
            # Perform stemming
            stemmed_word = ps.stem(word)
            cleaned_words.append(stemmed_word)
    
    # Join the cleaned and stemmed words back into a sentence
    cleaned_text = " ".join(cleaned_words)
    
    return cleaned_text




tfidf=pickle.load(open("vectorizer.pkl", 'rb'))
model=pickle.load(open("model.pkl", 'rb'))

st.title("SPAM CLASSIFIER")

st.markdown('#### "Guarding Your Inbox: Separating Ham from Spam with Precision"')

st.write("This is a Spam Classifier Web App.\
             Enter your Email or SMS in the input box to check whether it is a Spam or not.")

input_text=st.text_area("Enter your Email or SMS ")

if st.button('Predict'):

    # Preprocess

    transformed= transform_text(input_text)

    # Vectorize
    vector_input = tfidf.transform([transformed])

    # Predict
    result = model.predict(vector_input)[0]

    # Display
    if result == 1:
        st.error("Spam")
    else:
        st.success("Not Spam")

