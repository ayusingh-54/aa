# Fake News Detection — Streamlit app

This small project contains a script to train a text classification pipeline and a Streamlit app to run predictions locally.

Files added:

- `train_model.py` — trains a pipeline (CountVectorizer -> TfidfTransformer -> LogisticRegression) and saves it to `models/news_classifier.pkl`.
- `app.py` — Streamlit app that loads the saved pipeline and provides a text input and prediction output.
- `requirements.txt` — Python dependencies.

Quick start (Windows / PowerShell):

1. Create a virtual environment (recommended) and activate it:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Train the model (this will read `data/Fake.csv` and `data/True.csv`, preprocess, train and save the pipeline):

```powershell
python train_model.py
```

This creates `models/news_classifier.pkl`. If training runs successfully you'll see the validation accuracy printed.

4. Run the Streamlit app:

```powershell
streamlit run app.py
```

Then open the URL shown by Streamlit in your browser.

Notes and troubleshooting:

- If you see an error about NLTK stopwords, run `python -m nltk.downloader stopwords` (the training script will try to download them automatically).
- If you want a different model, edit `train_model.py` and change the estimator in the pipeline.
