import os
import streamlit as st
from deep_translator import GoogleTranslator
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import time

# =============================================
# Environment Configuration
# =============================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLOAD"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "1000"

# =============================================
# Dependency Check
# =============================================
def check_environment():
    required = {
        'torch': '2.7.1',
        'transformers': '4.41.1',
        'tiktoken': '0.5.2',
        'sentencepiece': '0.2.0'
    }
    
    missing = []
    wrong_version = []
    
    for pkg, ver in required.items():
        try:
            imported = __import__(pkg)
            if hasattr(imported, '__version__') and imported.__version__ != ver:
                wrong_version.append(f"{pkg} (have {imported.__version__}, need {ver})")
        except ImportError:
            missing.append(pkg)
    
    if missing or wrong_version:
        st.error("Dependency issues found:")
        if missing:
            st.error(f"Missing packages: {', '.join(missing)}")
        if wrong_version:
            st.error(f"Wrong versions: {', '.join(wrong_version)}")
        st.stop()

check_environment()

# =============================================
# Model Loading Functions
# =============================================
@st.cache_resource(ttl=24*3600, show_spinner="Loading paraphrase model...")
def load_paraphrase_model():
    try:
        model_name = "humarin/chatgpt_paraphraser_on_T5_base"  # Smaller alternative
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            legacy=False
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype="auto"
        )
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load paraphrase model: {str(e)}")
        return None, None

@st.cache_resource(ttl=24*3600, show_spinner="Loading grammar model...")
def load_grammar_model():
    try:
        model_name = "vennify/t5-base-grammar-correction"  # Smaller alternative
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load grammar model: {str(e)}")
        return None, None

@st.cache_resource(ttl=24*3600, show_spinner="Loading sentence model...")
def load_sentence_model():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')  # Smaller model
    except Exception as e:
        st.error(f"Failed to load sentence model: {str(e)}")
        return None

# Load all models
paraphrase_tokenizer, paraphrase_model = load_paraphrase_model()
gc_tokenizer, gc_model = load_grammar_model()
sentence_model = load_sentence_model()

# Verify all models loaded
if None in [paraphrase_tokenizer, paraphrase_model, gc_tokenizer, gc_model, sentence_model]:
    st.error("Critical models failed to load. Please check the logs.")
    st.stop()

# =============================================
# Helper Functions
# =============================================
def safe_translate(text, source='auto', target='en'):
    try:
        return GoogleTranslator(source=source, target=target).translate(text)
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text  # Return original text if translation fails

def translate_to_english(text):
    return safe_translate(text, source='hi', target='en')

def translate_to_hindi(text):
    return safe_translate(text, source='en', target='hi')

def paraphrase(text, num_return_sequences=1, num_beams=5):
    try:
        inputs = paraphrase_tokenizer.encode("paraphrase: " + text, 
                                          return_tensors="pt", 
                                          max_length=512, 
                                          truncation=True)
        outputs = paraphrase_model.generate(
            inputs,
            max_length=512,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            early_stopping=True
        )
        return paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Paraphrasing error: {str(e)}")
        return text

def grammar_correction(text):
    try:
        inputs = gc_tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
        outputs = gc_model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
        return gc_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Grammar correction error: {str(e)}")
        return text

def process_hindi_paragraph(text):
    try:
        en_text = translate_to_english(text)
        para_text = paraphrase(en_text)
        corrected_text = grammar_correction(para_text)
        return translate_to_hindi(corrected_text)
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return text

def validate_output(original, processed):
    try:
        original_en = translate_to_english(original)
        processed_en = translate_to_english(processed)
        orig_emb = sentence_model.encode([original_en])
        proc_emb = sentence_model.encode([processed_en])
        sim_score = cosine_similarity(orig_emb, proc_emb)[0][0]
        return original_en, processed_en, sim_score
    except Exception as e:
        st.error(f"Validation error: {str(e)}")
        return original, processed, 0.0

def ngram_similarity(text1, text2, n=2):
    try:
        def get_ngrams(text, n):
            tokens = text.split()
            return set(zip(*[tokens[i:] for i in range(n)]))
        
        set1 = get_ngrams(text1, n)
        set2 = get_ngrams(text2, n)
        return len(set1 & set2) / max(1, len(set1 | set2))
    except:
        return 0.0

# =============================================
# Streamlit UI
# =============================================
st.set_page_config(page_title="Hindi Text Processing", layout="wide")

st.title("📝 Hindi Text Processing Application")
st.markdown("""
Enter a Hindi paragraph to translate, paraphrase, correct grammar, and evaluate its similarity to the original.
""")

hindi_text = st.text_area("📌 Enter Hindi Text", height=200, 
                         placeholder="Type or paste your Hindi text here...",
                         help="Enter at least 2-3 sentences for best results")

if st.button("🚀 Process Text") and hindi_text:
    with st.spinner("🔁 Translating and processing text..."):
        try:
            processed_text = process_hindi_paragraph(hindi_text)
            orig_en, proc_en, cosine_score = validate_output(hindi_text, processed_text)

            # Calculate metrics
            orig_wc, orig_cc = len(hindi_text.split()), len(hindi_text)
            proc_wc, proc_cc = len(processed_text.split()), len(processed_text)
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
                st.metric("Original Character Count", orig_cc)
            with col2:
                st.metric("Processed Word Count", proc_wc)
                st.metric("Processed Character Count", proc_cc)

            st.subheader("📊 Similarity Scores")
            st.write(f"**Cosine Similarity:** {cosine_score:.4f}")
            st.write(f"**Unigram Similarity:** {uni_sim:.4f}")
            st.write(f"**Bigram Similarity:** {bi_sim:.4f}")

            with st.expander("🔍 View English Translations"):
                st.text("Original English Translation:")
                st.code(orig_en, language='text')
                st.text("Processed English Translation:")
                st.code(proc_en, language='text')

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.error("Please try again with different text or check back later.")