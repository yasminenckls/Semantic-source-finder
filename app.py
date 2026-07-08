import re
import streamlit as st
from sentence_transformers import CrossEncoder

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


VECTOR_DB_FOLDER = "vector_db"

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "what",
    "why", "how", "is", "are", "does", "do", "could", "would", "should", "about",
    "le", "la", "les", "un", "une", "des", "et", "ou", "de", "du", "dans", "sur",
    "pour", "avec", "quoi", "quel", "quelle", "quels", "quelles", "comment",
    "pourquoi", "est", "sont", "fcc"
}

DOCUMENT_OPTIONS = {
    "Volume 1 — Feasibility Study": "vol1FCC.pdf",
    "DMO FCC": "CERN-FCC-DMO.pdf",
    "European Strategy": "EuroStrat_PP.pdf",
    "LHC FAQ": "LHC_TheGuide.pdf", 
    "Synthese DMO" : "CERN-FCC-DMO-Synthese.pdf",
    "REX" : "REX_LHC_LEP-Web.pdf", 
    "PB Chap 1-2" : "essai20_PhysicsBrief.pdf",
    "PB Chap 11" : "essai11_PhysicsBrief.pdf", 
    "Socio-economic Study" : "Socio-Economic_Impact_Study_CERN_2026.pdf"
}


def normalize_text(text):
    return " ".join(text.split())


def extract_keywords(question):
    words = re.findall(r"[A-Za-zÀ-ÿ0-9\-]+", question.lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def is_bad_chunk(text):
    text_lower = text.lower()

    bad_patterns = [
        ". . . . .",
        "table of contents",
        "bibliography",
        "references",
        "acknowledgements",
        "source en français",
        "source en anglais",
        "ibidem",
        "consultable ici",
        "https://",
    ]

    if any(pattern in text_lower for pattern in bad_patterns):
        return True

    if text.count(". .") > 5:
        return True

    if len(text.strip()) < 250:
        return True

    return False


@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    vectorstore = Chroma(
        persist_directory=VECTOR_DB_FOLDER,
        embedding_function=embeddings
    )

    return vectorstore

@st.cache_resource
def load_reranker():
    return CrossEncoder("BAAI/bge-reranker-v2-m3")


st.set_page_config(
    page_title="Semantic Source Finder",
    page_icon="🔎",
    layout="wide"
)
st.caption("Embedding model: BAAI/bge-m3")
st.caption("Reranker: BAAI/bge-reranker-v2-m3")

st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 4rem;
        padding-bottom: 4rem;
        max-width: 1100px;
    }

    [data-testid="stSidebar"] {
        background-color: #e8edf5;
        border-right: 1px solid #d3dbe8;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #0033a0;
    }

    h1 {
        color: #0033a0;
        font-weight: 800;
        letter-spacing: -0.03em;
    }

    h2, h3 {
        color: #1f2937;
    }

    .stTextInput input {
        background-color: white;
        border: 1px solid #c7d2e5;
        border-radius: 10px;
        padding: 0.75rem;
    }

    .stTextInput input:focus {
        border-color: #0033a0;
        box-shadow: 0 0 0 2px rgba(0, 51, 160, 0.15);
    }

    .stButton > button {
        background-color: #0033a0;
        color: white;
        border: 1px solid #0033a0;
        border-radius: 10px;
        padding: 0.6rem 1.1rem;
        font-weight: 700;
    }

    .stButton > button:hover {
        background-color: #0053d6;
        color: white;
        border-color: #0053d6;
    }

    .stMultiSelect span[data-baseweb="tag"] {
        background-color: #0033a0;
        color: white;
        border-radius: 8px;
    }

    .source-card {
        border: 1px solid #d7deea;
        border-left: 6px solid #0033a0;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 16px;
        background-color: white;
        box-shadow: 0px 4px 14px rgba(0, 35, 100, 0.06);
    }

    .source-title {
        font-size: 18px;
        font-weight: 800;
        color: #0033a0;
        margin-bottom: 6px;
    }

    .source-meta {
        font-size: 14px;
        color: #4b5563;
    }

    .streamlit-expanderHeader {
        color: #0033a0;
        font-weight: 700;
    }
    
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #0033a0;
        border-color: #0033a0;
    }

    .stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] {
        background-color: #0033a0;
    }
            
    
</style>
""", unsafe_allow_html=True)

st.markdown("""
# Semantic Source Finder
""")

with st.sidebar:
    st.header("Search settings")

    number_of_results = st.slider(
        "Number of source passages:",
        min_value=1,
        max_value=10,
        value=5,
        key="num_results_slider"
    )

    selected_labels = st.multiselect(
        "Documents to search in:",
        options=list(DOCUMENT_OPTIONS.keys()),
        default=list(DOCUMENT_OPTIONS.keys()),
        key="document_selector"
    )


question = st.text_input(
    "Ask a question about the FCC:",
    placeholder="Example: What are the detector concepts at the FCC-ee?",
    key="fcc_question_input"
)

selected_files = [DOCUMENT_OPTIONS[label] for label in selected_labels]

search_button = st.button(
    "Search sources",
    key="search_sources_button"
)
if "results" not in st.session_state:
    st.session_state.results = []

if "last_question" not in st.session_state:
    st.session_state.last_question = ""



if not search_button:
    st.info(
        "Ask your question in English or French."
    )

if search_button:
    if not question.strip():
        st.warning("Please enter a question.")
    elif not selected_files:
        st.warning("Please select at least one document.")
    else:
        with st.spinner("Searching source passages..."):
            vectorstore = load_vectorstore()

            selected_sources = [f"documents/{file_name}" for file_name in selected_files]

            if len(selected_sources) == 1:
                metadata_filter = {"source": selected_sources[0]}
            else:
                metadata_filter = {
                    "$or": [{"source": source} for source in selected_sources]
                }

            raw_results = vectorstore.similarity_search(
                question,
                k=number_of_results * 5,
                filter=metadata_filter
            )
            raw_results = [doc for doc in raw_results if not is_bad_chunk(doc.page_content)]
            if not raw_results:
                st.warning("No relevant source passage found after filtering.")
                st.session_state.results = []
            else:
                reranker = load_reranker()
                pairs = [(question, doc.page_content) for doc in raw_results]
                scores = reranker.predict(pairs)

                scored_results = list(zip(scores, raw_results))
                scored_results.sort(key=lambda x: x[0], reverse=True)

                results = scored_results[:number_of_results]
                st.session_state.results = results
                st.session_state.last_question = question

selected_for_answer = []

if st.session_state.results:
    st.subheader("Source passages found")

    for i, (score, doc) in enumerate(st.session_state.results, start=1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown page")
        text = " ".join(doc.page_content.split())

        st.markdown(f"""
        <div class="source-card">
            <div class="source-title">Result {i}</div>
            <div class="source-meta">
                <b>Document:</b> {source} &nbsp; | &nbsp;
                <b>Page:</b> {page} &nbsp; | &nbsp;
                <b>Reranker score:</b> {score:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Read source passage"):
            st.write(text[:1200])