import streamlit as st
from huggingface_hub import login
from deep_translator import GoogleTranslator
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Page configuration
st.set_page_config(page_title="Hindi Text Processing", layout="wide")

# Initialize Translator
#translator = Translator()

# Cache heavy models
@st.cache_resource
def load_paraphrase_model():
    model_name = "ramsrigouthamg/t5_paraphraser"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource
def load_grammar_model():
    model_name = "prithivida/grammar_error_correcter_v1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

paraphrase_tokenizer, paraphrase_model = load_paraphrase_model()
gc_tokenizer, gc_model = load_grammar_model()
sentence_model = load_sentence_model()

def safe_translate(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        return "Translation Error"


def translate_to_english(text):
    return safe_translate(text, 'hi', 'en')

def translate_to_hindi(text):
    return safe_translate(text, 'en', 'hi')

def paraphrase(text):
    inputs = paraphrase_tokenizer.encode("paraphrase: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = paraphrase_model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
    return paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True)

def grammar_correction(text):
    inputs = gc_tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = gc_model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
    return gc_tokenizer.decode(outputs[0], skip_special_tokens=True)

def process_hindi_paragraph(text):
    en = translate_to_english(text)
    para = paraphrase(en)
    corrected = grammar_correction(para)
    return translate_to_hindi(corrected)

def validate_output(original, processed):
    original_en = translate_to_english(original)
    processed_en = translate_to_english(processed)
    orig_emb = sentence_model.encode([original_en])
    proc_emb = sentence_model.encode([processed_en])
    sim_score = cosine_similarity(orig_emb, proc_emb)[0][0]
    return original_en, processed_en, sim_score

def ngram_similarity(text1, text2, n):
    def get_ngrams(text, n):
        tokens = text.split()
        return set(zip(*[tokens[i:] for i in range(n)]))
    set1 = get_ngrams(text1, n)
    set2 = get_ngrams(text2, n)
    return len(set1 & set2) / max(1, len(set1 | set2))

# UI
st.title("📝 Hindi Text Processing Application")
st.markdown("""
Enter a Hindi paragraph to translate, paraphrase, correct grammar, and evaluate its similarity to the original.
""")

hindi_text = st.text_area("📌 Enter Hindi Text", height=200, placeholder="Type or paste your Hindi text here...")

if st.button("🚀 Process Text") and hindi_text:
    with st.spinner("🔁 Translating and processing text..."):
        processed_text = process_hindi_paragraph(hindi_text)
        orig_en, proc_en, cosine_score = validate_output(hindi_text, processed_text)

        # Word and character counts
        orig_wc, orig_cc = len(hindi_text.split()), len(hindi_text)
        proc_wc, proc_cc = len(processed_text.split()), len(processed_text)

        # N-gram similarity
        uni_sim = ngram_similarity(orig_en, proc_en, 1)
        bi_sim = ngram_similarity(orig_en, proc_en, 2)

    # Display results
    st.subheader("🔹 Original Hindi Text")
    st.success(hindi_text)

    st.subheader("🔹 Processed Hindi Text")
    st.info(processed_text)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Original Word Count", orig_wc)
        st.metric("Original Char Count", orig_cc)
    with col2:
        st.metric("Processed Word Count", proc_wc)
        st.metric("Processed Char Count", proc_cc)

    st.subheader("📊 Similarity Scores")
    st.write(f"**Cosine Similarity:** {cosine_score:.4f}")
    st.write(f"**Unigram Similarity:** {uni_sim:.4f}")
    st.write(f"**Bigram Similarity:** {bi_sim:.4f}")

    with st.expander("🔍 View English Translations"):
        st.code(orig_en, language='text')
        st.code(proc_en, language='text')
