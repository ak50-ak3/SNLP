
SNLP Resume Analyzer App

An interactive Streamlit-based NLP application for automated resume analysis.
It extracts skills from resumes using spaCy, classifies them using a Naive Bayes model,
and visualizes important keywords via Chi-square (χ²) n-gram analytics.


Features

Skill Extraction:
  Uses `spaCy` Named Entity Recognition (NER) and noun chunking to identify and clean relevant skills.

Resume Classification:
  Applies a TF-IDF + Multinomial Naive Bayes pipeline to classify resumes into job roles (e.g., Data Science, Java Developer, HR).

Explainability (χ² N-grams):
  Shows the most discriminative unigrams and bigrams per job category using Chi-square analysis.

Streamlit UI Tabs:

  1. Search by Skill* — Find resumes matching specific skills.
  2. Naive Bayes Classifier* — View accuracy, F1-score, and confusion matrix.
  3. N-gram Analytics* — Visualize top χ² features per category.


Name: `UpdatedResumeDataSet.csv`
Columns: `Resume`, `Category`
Size: ~962 resumes
Classes: Multi-domain — Data Science, HR, Java Developer, etc.



# Tech Stack

| Layer         | Tools / Libraries                        |
| ------------- | ---------------------------------------- |
| Language      | Python 3.10+                             |
| Web Framework | Streamlit                                |
| NLP           | spaCy                                    |
| ML            | Scikit-learn (Naive Bayes, TF-IDF, Chi²) |
| Visualization | Matplotlib                               |
| Data Handling | Pandas, NumPy                            |



# Model Details

Vectorizer: TF-IDF (1–2 grams, max 30,000 features)Classifier: Multinomial Naive Bayes
Performance:

Accuracy ≈ 96.9%
Weighted F1 ≈ 96.5%



# Insights

Highlights top skills and n-grams correlated with each role.
Enables recruiters to quickly identify resumes relevant to a job profile.
Demonstrates interpretability in NLP classification tasks.



# How to Run

bash
# Clone this repository
git clone https://github.com/<your-username>/SNLP-Resume-App.git
cd SNLP-Resume-App

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run resume_app.py




# Future Enhancements

Integration with deep learning models (LSTM, BERT).
Support for PDF/Docx resume uploads.
Visualization dashboard for role-skill overlap.



License

This project is open-source under the MIT License.




