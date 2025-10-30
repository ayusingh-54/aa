import streamlit as st
import joblib
import os

MODEL_PATH = os.path.join('models', 'news_classifier.pkl')

st.set_page_config(page_title='Fake News Detector', layout='centered')

st.title('Fake News Detection')
st.markdown('Enter a news article (or a short text) and the model will predict whether it is likely `fake` or `true`.')

if not os.path.exists(MODEL_PATH):
    st.error(f'Model not found at {MODEL_PATH}.\n\nRun `python train_model.py` to train and create the model before using this app.')
else:
    model = joblib.load(MODEL_PATH)

    text = st.text_area('Enter text to classify', height=250)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button('Predict') and text.strip() != '':
            pred = model.predict([text])[0]
            st.write('### Prediction')
            st.success(f'Predicted: {pred}')

            # probabilities (if supported)
            try:
                proba = model.predict_proba([text])[0]
                classes = model.classes_
                proba_pairs = sorted(zip(classes, proba), key=lambda x: -x[1])
                st.write('### Probabilities')
                for cls, p in proba_pairs:
                    st.write(f'- {cls}: {p*100:.2f}%')
            except Exception:
                st.info('Model does not expose probabilities.')

    with col2:
        st.write('### Tips')
        st.write('- Provide at least one or two sentences for better predictions.')
        st.write('- If the model is missing, run the included `train_model.py` script to generate it.')

    st.write('---')
    st.write('Small note: this app uses a pipeline (CountVectorizer + Tfidf + LogisticRegression). For best results, retrain the model on your dataset if you change preprocessing.')
