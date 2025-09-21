import streamlit as st
import pandas as pd
import google.generativeai as genai
import pdfplumber
import docx2txt
import os
import re
import json
import tempfile
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("spaCy English model not found. Please run: python -m spacy download en_core_web_sm")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Innomatics AI Hiring Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- Constants & Paths ---
DATA_DIR = "data"
JD_DIR = os.path.join(DATA_DIR, "JD")
RESUMES_DIR = os.path.join(DATA_DIR, "Resumes")
os.makedirs(JD_DIR, exist_ok=True)
os.makedirs(RESUMES_DIR, exist_ok=True)

# Common skills database for enhancement
TECH_SKILLS_DB = [
    "Python", "Java", "JavaScript", "SQL", "R", "C++", "C#", "PHP", "Swift", "Kotlin",
    "React", "Angular", "Vue", "Django", "Flask", "Spring", "Node.js", "Express",
    "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Pandas", "NumPy", "Matplotlib",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Jenkins", "Git", "CI/CD",
    "MySQL", "PostgreSQL", "MongoDB", "Redis", "Cassandra", "Hadoop", "Spark",
    "Tableau", "Power BI", "Excel", "SAS", "SPSS", "ML", "AI", "NLP", "Computer Vision"
]

# --- Text Extraction Functions ---
@st.cache_data
def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    try:
        with pdfplumber.open(file_path) as pdf:
            text = "".join(page.extract_text() or "" for page in pdf.pages)
        return text
    except Exception as e:
        st.error(f"Error reading PDF {os.path.basename(file_path)}: {e}")
        return None

@st.cache_data
def extract_text_from_docx(file_path):
    """Extracts text from a DOCX file."""
    try:
        text = docx2txt.process(file_path)
        return text
    except Exception as e:
        st.error(f"Error reading DOCX {os.path.basename(file_path)}: {e}")
        return None

def extract_text(file_path):
    """Extracts text from a file based on its extension."""
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    else:
        st.error(f"Unsupported file format: {os.path.basename(file_path)}")
        return None

# --- Text Processing Functions ---
def preprocess_text(text):
    """Preprocess text by removing extra spaces, special characters, and normalizing."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep some basic punctuation
    text = re.sub(r'[^\w\s.,!?;:]', ' ', text)
    
    return text.strip()

def extract_skills(text):
    """Extract skills from text using a combination of keyword matching and NLP."""
    text_lower = text.lower()
    found_skills = []
    
    # Keyword matching
    for skill in TECH_SKILLS_DB:
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
            found_skills.append(skill)
    
    # Use spaCy for additional entity recognition
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "TECH"] and ent.text not in found_skills:
            # Check if it might be a technical skill
            if any(char.isupper() for char in ent.text) and len(ent.text) > 2:
                found_skills.append(ent.text)
    
    return list(set(found_skills))

def extract_sections(text):
    """Attempt to extract common resume sections."""
    sections = {
        "experience": "",
        "education": "",
        "skills": "",
        "projects": "",
        "certifications": ""
    }
    
    # Simple regex-based section extraction
    patterns = {
        "experience": r"(?i)(experience|work history|employment history|professional experience)[:\s]*(.*?)(?=(?i)education|skills|projects|certifications|$)",
        "education": r"(?i)(education|academic background|qualifications)[:\s]*(.*?)(?=(?i)experience|skills|projects|certifications|$)",
        "skills": r"(?i)(skills|technical skills|competencies)[:\s]*(.*?)(?=(?i)experience|education|projects|certifications|$)",
        "projects": r"(?i)(projects|personal projects|project experience)[:\s]*(.*?)(?=(?i)experience|education|skills|certifications|$)",
        "certifications": r"(?i)(certifications|certificates|licenses)[:\s]*(.*?)(?=(?i)experience|education|skills|projects|$)"
    }
    
    for section, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sections[section] = match.group(2).strip()
    
    return sections

# --- Analysis Functions ---
def calculate_keyword_similarity(jd_text, resume_text):
    """Calculate keyword-based similarity using TF-IDF and cosine similarity."""
    # Preprocess texts
    jd_processed = preprocess_text(jd_text)
    resume_processed = preprocess_text(resume_text)
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([jd_processed, resume_processed])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return min(1.0, max(0.0, similarity)) * 100  # Convert to percentage
    except:
        return 0.0

def extract_requirements_from_jd(jd_text):
    """Extract key requirements from job description."""
    requirements = {
        "must_have_skills": [],
        "good_to_have_skills": [],
        "education": [],
        "experience": None
    }
    
    # Extract skills (simple approach)
    jd_skills = extract_skills(jd_text)
    requirements["must_have_skills"] = jd_skills
    
    # Try to extract education requirements
    education_patterns = [
        r"(?i)(bachelor|master|phd|bs|ms|ph\.?d)\.?.*?(degree|in|of)?.*?(\w+)",
        r"(?i)(degree|diploma|certification).*?(in|of).*?(\w+)"
    ]
    
    for pattern in education_patterns:
        matches = re.findall(pattern, jd_text)
        for match in matches:
            education = " ".join([m for m in match if m]).strip()
            if education and education not in requirements["education"]:
                requirements["education"].append(education)
    
    # Try to extract experience requirements
    exp_pattern = r"(?i)(\d+)\+?\s*years?.*?experience"
    exp_match = re.search(exp_pattern, jd_text)
    if exp_match:
        requirements["experience"] = exp_match.group(1)
    
    return requirements

def identify_gaps(jd_requirements, resume_skills, resume_sections):
    """Identify gaps between JD requirements and resume content."""
    gaps = {
        "missing_skills": [],
        "missing_education": [],
        "missing_experience": None,
        "weaknesses": []
    }
    
    # Check for missing skills
    for skill in jd_requirements["must_have_skills"]:
        if skill not in resume_skills:
            gaps["missing_skills"].append(skill)
    
    # Check for education gaps
    for education_req in jd_requirements["education"]:
        if not any(edu.lower() in resume_sections["education"].lower() for edu in education_req.split() if len(edu) > 3):
            gaps["missing_education"].append(education_req)
    
    # Check experience
    if jd_requirements["experience"]:
        # Simple check for experience keywords
        if not re.search(r"(\d+)\s*\+?\s*years?", resume_sections["experience"], re.IGNORECASE):
            gaps["missing_experience"] = f"At least {jd_requirements['experience']} years of experience required"
    
    return gaps

# --- AI Analysis Function ---
def get_ai_hiring_assistant_evaluation(jd_text, resume_text, jd_requirements, resume_skills, gaps):
    """
    Uses Google Gemini with an advanced prompt to act as an AI Hiring Assistant.
    """
    try:
        if 'GOOGLE_API_KEY' not in st.secrets:
            st.error("Google API key not found. Please add it to your Streamlit secrets.")
            return "[ERROR]: API key missing"
            
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        prompt = f"""
        **Persona:** You are a highly experienced Senior Technical Recruiter at a top-tier technology company. 
        You have 15 years of experience and a keen eye for talent. Your goal is to provide a comprehensive, 
        actionable analysis for your busy hiring managers.

        **Task:** Analyze the following Job Description (JD) and Resume. Provide a structured, professional evaluation.

        **Job Description:**
        ---
        {jd_text[:3000]}  # Limiting size to avoid token limits
        ---

        **Candidate's Resume:**
        ---
        {resume_text[:3000]}  # Limiting size to avoid token limits
        ---

        **Additional Context:**
        - Required skills from JD: {', '.join(jd_requirements['must_have_skills'])}
        - Candidate's skills: {', '.join(resume_skills)}
        - Identified gaps: {json.dumps(gaps)}

        **Output Instructions:**
        Provide your analysis using the following structure ONLY. Do not add any extra conversational text.

        **[SCORE]:** A single integer relevance score from 0 to 100, based on overall fit.
        **[VERDICT]:** A concise verdict: "High Fit", "Medium Fit", or "Low Fit".
        **[SUMMARY]:** A 3-4 sentence executive summary explaining the reasoning behind your verdict.
        **[MISSING_ELEMENTS]:** A bulleted list of missing skills, qualifications, or experience based on the JD requirements.
        **[STRENGTHS]:** A bulleted list of the candidate's strengths relevant to this position.
        **[RECOMMENDATIONS]:** A bulleted list of specific recommendations for the candidate to improve their fit for this role.
        **[INTERVIEW_QUESTIONS]:** A bulleted list of 3-4 specific, insightful questions to ask the candidate during an interview.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[ERROR]: An error occurred with the Google API: {e}"

def parse_ai_response(response_text):
    """Parses the structured text from the LLM into a dictionary."""
    if response_text.startswith("[ERROR]"):
        return {"error": response_text}

    parsed_data = {}
    sections = {
        "[SCORE]": "score", 
        "[VERDICT]": "verdict", 
        "[SUMMARY]": "summary",
        "[MISSING_ELEMENTS]": "missing_elements", 
        "[STRENGTHS]": "strengths",
        "[RECOMMENDATIONS]": "recommendations", 
        "[INTERVIEW_QUESTIONS]": "interview_questions"
    }
    
    for header, key in sections.items():
        pattern = re.compile(rf"{re.escape(header)}(.*?)(?=\n\[[A-Z_]+\]|$)", re.DOTALL)
        match = pattern.search(response_text)
        if match:
            parsed_data[key] = match.group(1).strip()
        else:
            parsed_data[key] = "N/A"
    
    return parsed_data

# --- UI Components ---
def render_sidebar():
    """Render the sidebar with configuration options."""
    st.sidebar.image("https://innomatics.in/wp-content/uploads/2022/11/Innomatics-Logo-1-e1669884558233.png", width=200)
    st.sidebar.header("Configuration")
    
    # File uploaders
    st.sidebar.subheader("Upload Files")
    uploaded_jd = st.sidebar.file_uploader("Upload Job Description (PDF/DOCX)", type=["pdf", "docx"])
    uploaded_resume = st.sidebar.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
    
    # Save uploaded files
    if uploaded_jd:
        jd_path = os.path.join(JD_DIR, uploaded_jd.name)
        with open(jd_path, "wb") as f:
            f.write(uploaded_jd.getbuffer())
    
    if uploaded_resume:
        resume_path = os.path.join(RESUMES_DIR, uploaded_resume.name)
        with open(resume_path, "wb") as f:
            f.write(uploaded_resume.getbuffer())
    
    # Get list of files
    jd_files = [f for f in os.listdir(JD_DIR) if f.endswith(('.pdf', '.docx'))]
    resume_files = [f for f in os.listdir(RESUMES_DIR) if f.endswith(('.pdf', '.docx'))]
    
    if not jd_files:
        st.sidebar.warning("No job descriptions found. Please upload one.")
    if not resume_files:
        st.sidebar.warning("No resumes found. Please upload one.")
    
    selected_jd_file = st.sidebar.selectbox("Select Job Description", jd_files) if jd_files else None
    analysis_mode = st.sidebar.radio("Choose Mode", ("Single Resume Analysis", "Batch Screening"))
    
    selected_resume_file = None
    if analysis_mode == "Single Resume Analysis" and resume_files:
        selected_resume_file = st.sidebar.selectbox("Select Resume", resume_files)
    
    return selected_jd_file, analysis_mode, selected_resume_file

def render_single_analysis(jd_path, resume_path):
    """Render the single resume analysis view."""
    jd_text = extract_text(jd_path)
    resume_text = extract_text(resume_path)
    
    if not jd_text or not resume_text:
        st.error("Could not extract text from one or both files.")
        return
    
    # Preprocess and analyze
    with st.spinner("Analyzing resume and job description..."):
        # Extract requirements from JD
        jd_requirements = extract_requirements_from_jd(jd_text)
        
        # Extract skills and sections from resume
        resume_skills = extract_skills(resume_text)
        resume_sections = extract_sections(resume_text)
        
        # Calculate keyword similarity
        keyword_score = calculate_keyword_similarity(jd_text, resume_text)
        
        # Identify gaps
        gaps = identify_gaps(jd_requirements, resume_skills, resume_sections)
        
        # Get AI analysis
        ai_response = get_ai_hiring_assistant_evaluation(jd_text, resume_text, jd_requirements, resume_skills, gaps)
        results = parse_ai_response(ai_response)
    
    if "error" in results:
        st.error(results["error"])
        return
    
    try:
        score = int(results.get('score', 0))
    except (ValueError, TypeError):
        score = 0
    
    # Display results
    st.success("Analysis Complete!")
    
    # Score and verdict
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="**Overall Relevance Score**", value=f"{score}/100")
    with col2:
        st.metric(label="**Keyword Match Score**", value=f"{keyword_score:.1f}/100")
    with col3:
        st.metric(label="**AI Verdict**", value=results.get('verdict', 'N/A'))
    
    # Tabs for detailed analysis
    summary_tab, skills_tab, gaps_tab, recommendations_tab = st.tabs([
        "📊 Summary", "🛠️ Skills Analysis", "⚠️ Gap Analysis", "💡 Recommendations"
    ])
    
    with summary_tab:
        st.subheader("Executive Summary")
        st.write(results.get('summary', 'No summary available.'))
        
        st.subheader("Interview Questions")
        st.write(results.get('interview_questions', 'No questions generated.'))
    
    with skills_tab:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("JD Requirements")
            st.write("**Must-have skills:**")
            for skill in jd_requirements["must_have_skills"]:
                status = "✅" if skill in resume_skills else "❌"
                st.write(f"{status} {skill}")
            
            if jd_requirements["education"]:
                st.write("**Education requirements:**")
                for edu in jd_requirements["education"]:
                    st.write(f"- {edu}")
            
            if jd_requirements["experience"]:
                st.write(f"**Experience required:** {jd_requirements['experience']}+ years")
        
        with col2:
            st.subheader("Candidate's Skills")
            st.write("**Technical skills found:**")
            for skill in resume_skills:
                st.write(f"- {skill}")
            
            if resume_sections["education"]:
                st.write("**Education:**")
                # Show first few lines of education section
                edu_preview = resume_sections["education"][:200] + "..." if len(resume_sections["education"]) > 200 else resume_sections["education"]
                st.write(edu_preview)
    
    with gaps_tab:
        st.subheader("Identified Gaps")
        st.write(results.get('missing_elements', 'No gap analysis available.'))
        
        if gaps["missing_skills"]:
            st.write("**Missing skills:**")
            for skill in gaps["missing_skills"]:
                st.write(f"- {skill}")
        
        if gaps["missing_education"]:
            st.write("**Potential education gaps:**")
            for edu in gaps["missing_education"]:
                st.write(f"- {edu}")
        
        if gaps["missing_experience"]:
            st.write(f"**Experience gap:** {gaps['missing_experience']}")
    
    with recommendations_tab:
        st.subheader("Improvement Recommendations")
        st.write(results.get('recommendations', 'No recommendations available.'))
        
        st.subheader("Candidate Strengths")
        st.write(results.get('strengths', 'No strengths identified.'))

def render_batch_analysis(jd_path):
    """Render the batch analysis view."""
    jd_text = extract_text(jd_path)
    if not jd_text:
        st.error("Could not extract text from job description.")
        return
    
    resume_files = [f for f in os.listdir(RESUMES_DIR) if f.endswith(('.pdf', '.docx'))]
    if not resume_files:
        st.error("No resumes found for batch analysis.")
        return
    
    st.warning(f"This will screen all {len(resume_files)} resumes. This might take a while.", icon="⚠️")
    
    if st.button("🚀 Start Batch Screening", type="primary", use_container_width=True):
        batch_results = []
        progress_bar = st.progress(0, text="Starting batch analysis...")
        
        for i, resume_file in enumerate(resume_files):
            progress_text = f"Analyzing {i+1}/{len(resume_files)}: {resume_file}"
            progress_bar.progress((i+1)/len(resume_files), text=progress_text)
            
            resume_path = os.path.join(RESUMES_DIR, resume_file)
            resume_text = extract_text(resume_path)
            
            if resume_text:
                # Basic analysis for batch processing
                jd_requirements = extract_requirements_from_jd(jd_text)
                resume_skills = extract_skills(resume_text)
                keyword_score = calculate_keyword_similarity(jd_text, resume_text)
                
                # Simple scoring based on keyword match and skill overlap
                skill_match = len(set(jd_requirements["must_have_skills"]) & set(resume_skills))
                skill_match_score = (skill_match / len(jd_requirements["must_have_skills"])) * 100 if jd_requirements["must_have_skills"] else 0
                
                # Combined score (weighted average)
                combined_score = (keyword_score * 0.4) + (skill_match_score * 0.6)
                
                # Simple verdict
                if combined_score >= 70:
                    verdict = "High Fit"
                elif combined_score >= 40:
                    verdict = "Medium Fit"
                else:
                    verdict = "Low Fit"
                
                batch_results.append({
                    "Resume": resume_file,
                    "Score": round(combined_score),
                    "Verdict": verdict,
                    "Keyword Match": round(keyword_score),
                    "Skills Match": f"{skill_match}/{len(jd_requirements['must_have_skills'])}",
                    "Missing Skills": ", ".join(set(jd_requirements["must_have_skills"]) - set(resume_skills))
                })
        
        progress_bar.empty()
        st.success("Batch screening complete!")
        
        if batch_results:
            df = pd.DataFrame(batch_results).sort_values(by="Score", ascending=False).reset_index(drop=True)
            
            # Display results
            st.dataframe(
                df, 
                use_container_width=True,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score", 
                        min_value=0, 
                        max_value=100,
                        format="%d"
                    ),
                }
            )
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"batch_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# --- Main App ---
def main():
    st.title("🤖 Innomatics AI Hiring Assistant")
    st.markdown("Automated resume evaluation against job requirements with detailed gap analysis and recommendations.")
    
    # Render sidebar and get configuration
    selected_jd_file, analysis_mode, selected_resume_file = render_sidebar()
    
    if not selected_jd_file:
        st.info("Please upload a job description to get started.")
        return
    
    jd_path = os.path.join(JD_DIR, selected_jd_file)
    
    if analysis_mode == "Single Resume Analysis":
        if not selected_resume_file:
            st.info("Please select a resume to analyze.")
            return
        
        resume_path = os.path.join(RESUMES_DIR, selected_resume_file)
        render_single_analysis(jd_path, resume_path)
        
    else:  # Batch Screening
        render_batch_analysis(jd_path)
    
    # Context expanders
    with st.expander("View Raw Text Content"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Job Description Text")
            jd_text = extract_text(jd_path)
            if jd_text:
                st.text(jd_text[:1000] + "..." if len(jd_text) > 1000 else jd_text)
        
        if analysis_mode == "Single Resume Analysis" and selected_resume_file:
            with col2:
                st.subheader("Resume Text")
                resume_path = os.path.join(RESUMES_DIR, selected_resume_file)
                resume_text = extract_text(resume_path)
                if resume_text:
                    st.text(resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text)


if __name__ == "__main__":
    main()