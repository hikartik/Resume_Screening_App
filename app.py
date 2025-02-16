import streamlit as st
import pickle
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

tfidfd = TfidfVectorizer(stop_words='english')
nltk.download('punkt')
nltk.download('stopwords')


#loading models
clf=pickle.load(open('clf.pkl','rb'))
tfidfd=pickle.load(open('tfidf.pkl','rb'))



def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText
# web app
def main():
    st.title('Resume Screening App')
    upload_file=st.file_uploader('Upload Resume',type=['pdf','txt'])

    if upload_file is not None:
        try:
            resume_bytes=upload_file.read()
            resume_text=resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')
        cleaned_resume=cleanResume(resume_text)
        cleaned_resume=tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(cleaned_resume)[0]
        
        category_mapping={0: 'Advocate', 1: 'Arts', 2: 'Automation Testing', 3: 'Blockchain', 4: 'Business Analyst', 5: 'Civil Engineer', 6: 'Data Science', 7: 'Database', 8: 'DevOps Engineer', 9: 'DotNet Developer', 10: 'ETL Developer', 11: 'Electrical Engineering', 12: 'HR', 13: 'Hadoop', 14: 'Health and fitness', 15: 'Java Developer', 16: 'Mechanical Engineer', 17: 'Network Security Engineer', 18: 'Operations Manager', 19: 'PMO', 20: 'Python Developer', 21: 'SAP Developer', 22: 'Sales', 23: 'Testing', 24: 'Web Designing'}
        category_name=category_mapping.get(prediction_id,"Unknown")
        st.write("Predicted Category:",category_name)        

if __name__=="__main__":
    main()
