# ResumeScreen: Smart Resume Screening App 💼🤖

**ResumeScreen** is an intelligent web application that automates the screening of resumes using machine learning. Designed for HR professionals and recruiters, this app cleans and processes resume files, extracts key features using TF-IDF vectorization, and classifies resumes into relevant job categories with high accuracy—all through an interactive Streamlit interface.

---

## Key Features

- **Automated Resume Classification:** Quickly categorize resumes into predefined job roles.
- **Robust Text Preprocessing:** Uses regular expressions to remove URLs, hashtags, mentions, special characters, and extra whitespace.
- **Efficient Feature Extraction:** Converts textual data into numerical features using TF-IDF.
- **Accurate Prediction:** Implements a K-Nearest Neighbors classifier in a OneVsRest framework, achieving high accuracy.
- **User-Friendly Interface:** Built with Streamlit for a seamless, interactive user experience.
- **Model Serialization:** Models and vectorizers are saved using Pickle for rapid deployment and scalability.

---

## Data Preprocessing & Model Training Pipeline

- **Data Loading & Exploration:**
  - Loaded the dataset (CSV) containing resumes and their corresponding categories from Kaggle.
  - Analyzed category distribution using Pandas and visualized data with Seaborn.
  
- **Text Cleaning:**
  - Employed regex-based cleaning to remove unwanted patterns (URLs, mentions, hashtags, punctuation, non-ASCII characters, etc.).
  
- **Encoding & Vectorization:**
  - Used `LabelEncoder` to transform job category labels into numerical values.
  - Utilized `TfidfVectorizer` (with English stop words) to convert cleaned resumes into numerical features.
  
- **Model Training:**
  - Split the dataset into training and testing sets (67:33 ratio).
  - Trained a OneVsRestClassifier with KNeighborsClassifier, achieving an accuracy of approximately **98.74%**.
  
- **Serialization:**
  - Serialized the trained TF-IDF vectorizer, classifier, and label encoder using Pickle for later use in the web app.

---

## Technologies Employed

- **Programming Language:** Python  
- **Web Framework:** Streamlit  
- **Machine Learning:** scikit-learn, NLTK  
- **Data Processing & Visualization:** Pandas, Matplotlib, Seaborn  
- **Model Persistence:** Pickle  

---

## How to Run the Application

1. **Clone the Repository:**
   - Run: `git clone https://github.com/your_username/ResumeScreen.git`
   
2. **Set Up Your Environment:**
   - **Windows:**  
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - **macOS/Linux:**  
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
     
3. **Install Dependencies:**
   - Upgrade pip and install required packages:
     ```bash
     pip install --upgrade pip
     pip install -r requirements.txt
     ```
     
4. **Run the Streamlit App:**
   - Start the app with:
     ```bash
     streamlit run app.py
     ```
   - Open your browser and navigate to [http://localhost:8501](http://localhost:8501)

---

## Project Structure

- **app.py:** Main Streamlit application.
- **clf.pkl:** Serialized classifier model.
- **tfidf.pkl:** Serialized TF-IDF vectorizer.
- **encoder.pkl:** Serialized label encoder.
- **requirements.txt:** List of project dependencies.
- **README.md:** Project documentation.

---

Enjoy using **ResumeScreen** and happy screening! 💼🚀
