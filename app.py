import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from PyPDF2 import PdfReader
import io
import re
import random
import torch
from sentence_transformers import SentenceTransformer
from helpers import clean_text, generate_distractors_from_doc
from sklearn.metrics.pairwise import cosine_similarity
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TORCH_LOAD_META_TENSORS"] = "0"

# -------------------- MODEL SETUP --------------------
# Summarization and QA
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)


qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=0 if torch.cuda.is_available() else -1)

# Question generation model
tokenizer_qg = AutoTokenizer.from_pretrained("valhalla/t5-small-qg-hl")
model_qg = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qg-hl")
qg_pipeline = pipeline("text2text-generation", model=model_qg, tokenizer=tokenizer_qg)

# Embedding model for similarity
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Smart Assistant", layout="wide")
st.title("📄 Smart Assistant - Document Reader")

# -------------------- USER INSTRUCTIONS --------------------
st.markdown("""
### ℹ️ How to Use the Smart Assistant:

1. 📂 Upload a **PDF or TXT** document.  
2. 🧠 Read the **auto-generated summary**.  
3. 💬 Switch to **Ask Anything** tab to ask questions.  
4. 🧪 Switch to **Challenge Me** tab to test your knowledge with MCQs.  
""")

# -------------------- FILE UPLOAD --------------------
uploaded_file = st.file_uploader("Upload a PDF or TXT document", type=["pdf", "txt"])
text = ""

if uploaded_file:
    st.success("✅ File uploaded successfully!")

    if uploaded_file.type == "application/pdf":
        pdf = PdfReader(uploaded_file)
        for page in pdf.pages:
            text += page.extract_text()
    else:
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        text = stringio.read()

    # Display raw document
    st.subheader("📄 Extracted Document Content:")
    st.text_area("Document Text", text[:2000], height=300)

    # -------------------- SUMMARY --------------------
    if len(text) > 100:
        st.subheader("🧠 Auto-generated Summary:")
        with st.spinner("Summarizing..."):
            summary = summarizer(text[:1000], max_length=150, min_length=40, do_sample=False)
            st.success("✅ Summary generated:")
            st.write(summary[0]["summary_text"])

    # -------------------- QUESTION ANSWERING --------------------
    st.subheader("💬 Ask Anything About the Document")
    user_question = st.text_input("Enter your question:")
    if user_question:
        answer = qa_model(question=user_question, context=text)
        st.write("**Answer:**", answer["answer"])

    # -------------------- CLEAN TEXT --------------------
    def clean_text(text):
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'https?://\S+', '', text)
        return text.strip()

    cleaned_text = clean_text(text)

    # -------------------- DISTRACTOR GENERATION --------------------
    def generate_distractors_from_doc(correct_answer, context_text, top_n=3):
        """Generate distractor options from the same document using sentence similarity"""
        sentences = list(set([s.strip() for s in context_text.split(".") if 5 < len(s.strip()) < 100]))
        correct_embedding = embedding_model.encode(correct_answer, convert_to_tensor=True)
        sentence_embeddings = embedding_model.encode(sentences, convert_to_tensor=True)
        scores = cosine_similarity([correct_embedding.cpu().numpy()], sentence_embeddings.cpu().numpy())[0]

        distractors = []
        for idx in scores.argsort()[::-1]:
            sent = sentences[idx]
            if correct_answer.lower() not in sent.lower():
                phrase = " ".join(sent.split()[:6]) + "..."
                if phrase not in distractors and phrase.lower() != correct_answer.lower():
                    distractors.append(phrase)
            if len(distractors) >= top_n:
                break
        return distractors

    # -------------------- MCQ CHALLENGE --------------------
    st.subheader("🧠 Challenge Me with MCQs!")

    if "questions_data" not in st.session_state:
        st.session_state.questions_data = []

    if st.button("Generate 3 MCQs"):
        sentences = [s.strip() for s in cleaned_text.split(".") if len(s.strip().split()) >= 5]
        questions, correct_answers, options_list = [], [], []

        for sentence in sentences:
            if len(questions) >= 3:
                break
            try:
                prompt = f"generate question: {sentence} </hl>"
                output = qg_pipeline(prompt)
                question = output[0]["generated_text"].strip().replace("<hl>", "").replace("</hl>", "")

                # Get correct answer from QA model
                answer = qa_model(question=question, context=text)["answer"]

                # Get distractors from document
                distractors = generate_distractors_from_doc(answer, text, top_n=3)
                all_options = [answer] + distractors
                all_options = list(set(all_options))  # remove exact duplicates
                random.shuffle(all_options)

                if len(all_options) >= 2:  # ensure we have at least 2 options
                    questions.append(question)
                    correct_answers.append(answer)
                    options_list.append(all_options)
            except Exception:
                continue

        # Store in session for answer checking
        st.session_state.questions_data = [
            {"question": q, "options": opts, "answer": ans}
            for q, opts, ans in zip(questions, options_list, correct_answers)
        ]

    # -------------------- MCQ DISPLAY & SUBMIT --------------------
    if st.session_state.questions_data:
        st.markdown("### Answer the following MCQs:")
        user_responses = []

        for idx, qdata in enumerate(st.session_state.questions_data):
            st.markdown(f"**Q{idx+1}: {qdata['question']}**")
            selected = st.radio(
                f"Choose your answer for Q{idx+1}:", 
                qdata["options"], 
                key=f"user_q{idx}",
                index=None  # No default selection
            )
            user_responses.append(selected)

        if st.button("Submit Answers"):
            score = 0
            for idx, (qdata, user_ans) in enumerate(zip(st.session_state.questions_data, user_responses)):
                correct = qdata["answer"]
                if user_ans == correct:
                    st.success(f"✅ Q{idx+1} is Correct!")
                    score += 1
                elif user_ans is None:
                    st.warning(f"⚠️ Q{idx+1} was not answered.")
                else:
                    st.error(f"❌ Q{idx+1} is Wrong! Correct Answer: **{correct}**")
            st.info(f"🎯 Your Total Score: {score} / 3")
