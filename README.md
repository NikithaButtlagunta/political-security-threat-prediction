# Political Security Threat Prediction Framework

A research prototype that combines lexicon-based sentiment features with supervised machine-learning classifiers to study sentiment polarity in political text. The project follows the major-project report's proposed approach: Decision Tree, Naive Bayes, and Support Vector Machine, with Random Forest as an extension.

> **Research-use note:** This is a sentiment-analysis prototype, not an operational national-security threat detector. Sentiment alone cannot establish intent, credibility, or whether a real-world threat exists. Do not use outputs to make decisions about individuals or groups.

## Project scope

- Text preprocessing and a transparent, lightweight lexicon score.
- TF-IDF text features combined with the lexicon score.
- Evaluation of Decision Tree, Multinomial Naive Bayes, SVM, and Random Forest.
- Flask web page for training from a labelled CSV and testing text.
- Dataset upload stays local to the running app; no text is sent to a third-party API.

The report describes Decision Tree, Naive Bayes, and SVM, and notes Random Forest as an extension. Accuracy must be measured on the dataset you supply. No accuracy result is claimed by this repository in advance.

## Repository layout

```text
.
├── app.py
├── sentiment_model.py
├── requirements.txt
├── .gitignore
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── data/
    └── .gitkeep
```

## Requirements

- Python 3.10–3.12 recommended
- pip
- A labelled CSV dataset. The CSV must contain a text column and a target/label column.

A dataset mentioned in the report is the [Dataset of Indian Politics Tweets and Reactions on Kaggle](https://www.kaggle.com/datasets/adritpal08/dataset-of-indian-politics-tweets-and-reactions). Check its current terms and column names before use; this app lets you select columns after uploading.

## Run locally

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Use the app

1. Upload a labelled CSV file.
2. Choose the text and label columns.
3. Choose a classifier and train/evaluate it.
4. Review the held-out test metrics and confusion matrix.
5. Enter text to receive the selected model's predicted label and the lexicon score.

Use labels that genuinely represent your research target. If your dataset has sentiment labels such as `positive`, `neutral`, and `negative`, the model predicts those labels; it does **not** automatically convert them into verified security threats.

## Evaluation and limitations

- The app uses a stratified train/test split when possible; metrics are calculated on the held-out split.
- Results depend on the dataset, label quality, class balance, and chosen columns.
- Lexicon scoring is a small, explainable baseline and is not a validated domain-specific political-security lexicon.
- The app does not scrape social media, identify people, authenticate users, or produce real-world threat assessments.
- Avoid uploading personal or sensitive data. Use datasets for which you have appropriate rights and permissions.

## Methods

The text pipeline uses TF-IDF features plus one lexicon-score feature. Classifiers available are Decision Tree, Multinomial Naive Bayes, Linear SVM, and Random Forest. This is an implementable baseline inspired by the report, not a reproduction of unprovided original experiment code.

## Troubleshooting

- **CSV encoding error:** export the dataset as UTF-8 CSV.
- **Missing labels:** remove blank target rows and ensure there are at least two classes.
- **Very small class counts:** use more labelled examples per class so a stratified split can be made.
- **NLTK download prompts:** this implementation does not require NLTK downloads.

## Citation / project context

Project title: *Political Security Threat Prediction Framework Using Hybrid Lexicon-Based Approach and Machine Learning Technique* (B.Tech major project, 2023–2024, Geethanjali College of Engineering and Technology).

Dataset reference in the report: [Kaggle dataset](https://www.kaggle.com/datasets/adritpal08/dataset-of-indian-politics-tweets-and-reactions).

## License

No license is declared yet. Add a license only after confirming that all contributors agree to the chosen terms and that dataset terms are respected.
