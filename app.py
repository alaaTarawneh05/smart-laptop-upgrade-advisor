"""
Smart Laptop Upgrade Advisor Portal
====================================
Activity 4 – Integrated Data Mining, Text Mining, and Graph Mining application
Built with Streamlit. Loads pre-trained K-Means + TF-IDF models from Activity 3.

Run with:    streamlit run app.py
"""
from groq import Groq



import os
import re
import string
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------
# 1. PAGE CONFIG (must be first Streamlit call)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Laptop Upgrade Advisor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------
# 2. CUSTOM CSS – editorial/technical aesthetic
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Page background and global font */
    .stApp {
        background: linear-gradient(180deg, #f7f5f0 0%, #ffffff 100%);
    }
    html, body, [class*="css"]  {
        font-family: 'Georgia', 'Times New Roman', serif;
    }

    /* Big editorial title */
    .portal-title {
        font-family: 'Georgia', serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a3a3a;
        letter-spacing: -0.5px;
        margin-bottom: 0;
        line-height: 1.1;
    }
    .portal-subtitle {
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        color: #d4734a;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 0.2rem;
    }
    .portal-divider {
        border: none;
        border-top: 2px solid #1a3a3a;
        margin: 1rem 0 2rem 0;
    }

    /* Section header */
    .section-header {
        font-family: 'Georgia', serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a3a3a;
        border-left: 5px solid #d4734a;
        padding-left: 12px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    /* Persona card */
    .persona-card {
        background: #1a3a3a;
        color: #f7f5f0;
        padding: 1.6rem 1.8rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        box-shadow: 6px 6px 0 #d4734a;
    }
    .persona-card h3 {
        font-family: 'Georgia', serif;
        font-size: 1.6rem;
        margin: 0 0 0.4rem 0;
        color: #f7f5f0;
    }
    .persona-card .meta {
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        color: #d4734a;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .persona-card p { margin: 0.5rem 0 0 0; line-height: 1.5; }

    /* Recommendation pills */
    .reco-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .reco-pill {
        background: #ffffff;
        border: 2px solid #1a3a3a;
        padding: 1.2rem;
        border-radius: 4px;
    }
    .reco-pill .label {
        font-family: 'Courier New', monospace;
        font-size: 0.75rem;
        color: #6b6b6b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .reco-pill .value {
        font-family: 'Georgia', serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #d4734a;
        margin-top: 0.3rem;
    }

    /* Evidence quote */
    .evidence {
        background: #fff8f1;
        border-left: 4px solid #d4734a;
        padding: 0.9rem 1.2rem;
        margin: 0.6rem 0;
        font-style: italic;
        font-size: 0.95rem;
        color: #333;
    }
    .evidence .cite {
        font-style: normal;
        font-family: 'Courier New', monospace;
        font-size: 0.75rem;
        color: #d4734a;
        display: block;
        margin-top: 0.4rem;
        letter-spacing: 1px;
    }

    /* Chatbot bubbles */
    .chat-user {
        background: #1a3a3a;
        color: #f7f5f0;
        padding: 0.7rem 1rem;
        border-radius: 12px 12px 2px 12px;
        margin: 0.4rem 0 0.4rem 20%;
    }
    .chat-bot {
        background: #ffffff;
        border: 1px solid #1a3a3a;
        padding: 0.7rem 1rem;
        border-radius: 12px 12px 12px 2px;
        margin: 0.4rem 20% 0.4rem 0;
    }
    .chat-bot .cite {
        font-family: 'Courier New', monospace;
        font-size: 0.7rem;
        color: #d4734a;
        margin-top: 0.4rem;
        display: block;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #1a3a3a;
    }
    section[data-testid="stSidebar"] * {
        color: #f7f5f0 !important;
    }
    section[data-testid="stSidebar"] .stButton button {
        background: #d4734a;
        color: #ffffff !important;
        border: none;
        font-family: 'Georgia', serif;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.6rem 1.2rem;
        width: 100%;
        border-radius: 2px;
        letter-spacing: 1px;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background: #b85d36;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# 3. NLTK SETUP (downloads once)
# ------------------------------------------------------------------
@st.cache_resource
def setup_nltk():
    for pkg in ["punkt", "punkt_tab", "stopwords"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    return True

setup_nltk()

STEMMER = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))


# ------------------------------------------------------------------
# 4. PREPROCESSOR (identical to the notebook)
# ------------------------------------------------------------------
def preprocessor(text: str) -> str:
    tokens = nltk.word_tokenize(str(text))
    tokens = [w.lower() for w in tokens]
    tokens = [w for w in tokens if w not in STOP_WORDS]
    tokens = [w for w in tokens if w not in string.punctuation]
    tokens = [w for w in tokens if w.isalnum()]
    tokens = [STEMMER.stem(w) for w in tokens]
    return " ".join(tokens)


# ------------------------------------------------------------------
# 5. CACHED LOADERS
# ------------------------------------------------------------------
@st.cache_resource
def load_models():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    return vectorizer, kmeans


@st.cache_data
def load_dataset():
    df = pd.read_csv("final_laptop_clustering_dataset.csv")
    return df


@st.cache_resource
def build_dataset_tfidf(_vectorizer, clean_stories):
    """Transform the whole dataset once and cache the sparse matrix."""
    return _vectorizer.transform(clean_stories)


# ------------------------------------------------------------------
# 6. CLUSTER METADATA (from Activity 3 personas)
# ------------------------------------------------------------------
CLUSTER_INFO = {
    0: {
        "name": "Student Productivity Users",
        "description": (
            "Users mainly running study apps, browsers, and document editors. "
            "Workload is light to moderate; common pain is older hardware feeling sluggish."
        ),
        "color": "#4a7c7e",
    },
    1: {
        "name": "Creative Content Users",
        "description": (
            "Designers and creators running Photoshop, Illustrator, video editors. "
            "Memory-heavy workloads with large asset libraries."
        ),
        "color": "#c25b56",
    },
    2: {
        "name": "Office Productivity Users",
        "description": (
            "Business users on Office apps, email, meetings, and reports. "
            "Reliability and battery life matter more than raw power."
        ),
        "color": "#d4a14a",
    },
    3: {
        "name": "Gaming Performance Users",
        "description": (
            "Gamers needing strong GPU, RAM, and thermal headroom. "
            "FPS, frame drops, and rendering are the recurring complaints."
        ),
        "color": "#7e4ac2",
    },
    4: {
        "name": "Programming / Developer Users",
        "description": (
            "Developers running IDEs, containers, compilers, and multiple browsers. "
            "Memory pressure from heavy multitasking is the typical bottleneck."
        ),
        "color": "#3a8a5a",
    },
}


# ------------------------------------------------------------------
# 7. CORE LOGIC FUNCTIONS
# ------------------------------------------------------------------
def predict_cluster(clean_text: str, vectorizer, kmeans) -> int:
    vec = vectorizer.transform([clean_text])
    return int(kmeans.predict(vec)[0])


def get_similar_cases(clean_text, vectorizer, dataset_tfidf, df, top_n=15):
    """Return DataFrame of top-N similar cases with similarity column."""
    user_vec = vectorizer.transform([clean_text])
    sims = cosine_similarity(user_vec, dataset_tfidf).flatten()
    top_idx = np.argsort(sims)[-top_n:][::-1]
    out = df.iloc[top_idx].copy()
    out["similarity"] = sims[top_idx]
    out["case_id"] = top_idx
    return out.reset_index(drop=True), sims


def recommend_upgrade_and_gain(similar_df: pd.DataFrame, cluster_id: int):
    """Recommend upgrade and gain based on majority vote in retrieved similar cases."""
    top_for_vote = similar_df.head(10)
    upgrade = top_for_vote["upgrade_first"].mode().iloc[0]
    gain = top_for_vote["gain_class"].mode().iloc[0]
    confidence = (top_for_vote["upgrade_first"] == upgrade).mean()
    return upgrade, gain, confidence


def build_ego_graph(similar_df, dataset_tfidf_matrix, top_n=15, threshold=0.5):
    """Threshold-based ego network with input node + top-N similar nodes."""
    top_idx = similar_df["case_id"].head(top_n).tolist()
    G = nx.Graph()
    G.add_node("Input", label="Input Story")


    # input → case edges
    for _, row in similar_df.head(top_n).iterrows():
        score = float(row["similarity"])
        if score >= threshold:
            G.add_node(f"Case {row['case_id']}")
            G.add_edge("Input", f"Case {row['case_id']}", weight=round(score, 2))

    # case ↔ case edges
    for i in top_idx:
        for j in top_idx:
            if i < j:
                score = cosine_similarity(
                    dataset_tfidf_matrix[i], dataset_tfidf_matrix[j]
                ).flatten()[0]
                if score >= threshold:
                    G.add_edge(f"Case {i}", f"Case {j}", weight=round(float(score), 2))
    return G


def draw_graph(G):
    pos = nx.spring_layout(G, seed=42, k=0.9)
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#f7f5f0")
    ax.set_facecolor("#f7f5f0")

    node_colors = ["#d4734a" if n == "Input" else "#1a3a3a" for n in G.nodes()]
    node_sizes = [2800 if n == "Input" else 1800 for n in G.nodes()]

    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes,
        edgecolors="#1a3a3a", linewidths=1.5, ax=ax,
    )
    nx.draw_networkx_edges(G, pos, edge_color="#8a8a8a", width=1.5, alpha=0.7, ax=ax)
    nx.draw_networkx_labels(
        G, pos, font_size=8, font_color="#f7f5f0", font_weight="bold", ax=ax,
    )
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=7, font_color="#d4734a", ax=ax,
    )
    ax.set_title("Threshold-Based Similarity Ego Network", fontsize=13,
                 fontfamily="serif", color="#1a3a3a", pad=15)
    ax.axis("off")
    plt.tight_layout()
    return fig


def compute_graph_metrics(G):
    pagerank = nx.pagerank(G)
    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    rows = []
    for n in G.nodes():
        rows.append([
            str(n),
            round(pagerank[n], 4),
            round(degree[n], 4),
            round(betweenness[n], 4),
            round(closeness[n], 4),
        ])
    return pd.DataFrame(rows, columns=[
        "Node", "PageRank", "Degree Centrality",
        "Betweenness Centrality", "Closeness Centrality",
    ])


def identify_key_cases(metrics_df: pd.DataFrame):
    """Return most representative, most connected, bridge, and unusual case nodes."""
    df_no_input = metrics_df[metrics_df["Node"] != "Input"]
    if df_no_input.empty:
        return {}
    return {
        "most_representative": df_no_input.sort_values("PageRank", ascending=False).iloc[0]["Node"],
        "most_connected": df_no_input.sort_values("Degree Centrality", ascending=False).iloc[0]["Node"],
        "bridge_case": df_no_input.sort_values("Betweenness Centrality", ascending=False).iloc[0]["Node"],
        "unusual_case": df_no_input.sort_values("Closeness Centrality", ascending=True).iloc[0]["Node"],
    }


# ------------------------------------------------------------------
# 8. CHATBOT (evidence-only retrieval)
# ------------------------------------------------------------------
# Function responsible for answering user questions
# using only the retrieved similar laptop cases

def chatbot_answer(question: str, similar_df: pd.DataFrame) -> dict:
    # Check if the user entered a question
    # If empty, return a warning message
    if not question.strip():
        return {
            "answer": "Please type a question.",
            "case_id": None,
            "cluster": None,
            "quote": None
        }
    


    # Create connection to Groq service
    client = Groq(api_key=#$$$$$$$$$$$$$$$$$$$$$$$#)


# Variable that will store evidence
# from the retrieved laptop cases
    evidence_text = ""

# Loop through the top 5 most similar laptop cases
# retrieved using cosine similarity
    for _, row in similar_df.head(5).iterrows():

        # Extract important information from each case
        case_id = int(row["case_id"])
        cluster_id = int(row["kmeans_cluster"])
        story = str(row["story_text"])
        upgrade = str(row["upgrade_first"])
        gain = str(row["gain_class"])

# Build evidence block containing
# case information, story, upgrade recommendation,
# and expected improvement

        evidence_text += f"""
Case ID: {case_id}
Cluster: {cluster_id}
Story Text: {story}
Recommended First Upgrade: {upgrade}
Expected Gain Class: {gain}
"""

    prompt = f"""
You are an evidence-based chatbot for a Smart Laptop Upgrade Advisor Portal.

Your role is to answer the user's question using only the retrieved laptop cases below.

Important rules:
- Use only the retrieved cases.
- Do not use outside knowledge.
- Do not invent laptop problems, upgrades, or gain classes.
- Keep the answer short and clear.
- Every supported answer must include a citation in this format:
  (CaseID: X, Cluster: Y, short quote: "...")
- The short quote must be copied from the Story Text.
- If the retrieved cases do not contain the answer, reply exactly:
  Not found in retrieved cases.

Retrieved cases:
{evidence_text}

User question:
{question}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "Answer only using the provided retrieved laptop cases."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=300
        )

        answer = response.choices[0].message.content.strip()

        return {
            "answer": answer,
            "case_id": None,
            "cluster": None,
            "quote": None
        }

    except Exception as e:
        return {
            "answer": f"Groq chatbot error: {e}",
            "case_id": None,
            "cluster": None,
            "quote": None
        }
    

# ------------------------------------------------------------------
# 9. LOAD EVERYTHING
# ------------------------------------------------------------------
vectorizer, kmeans = load_models()
df_full = load_dataset()
dataset_tfidf = build_dataset_tfidf(vectorizer, df_full["clean_story"].fillna("").tolist())


# ------------------------------------------------------------------
# 10. SIDEBAR – INPUT FORM
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='font-family:Georgia,serif;font-size:1.4rem;font-weight:700;"
        "border-bottom:1px solid #d4734a;padding-bottom:0.6rem;margin-bottom:1rem;'>"
        "User Inputs</div>", unsafe_allow_html=True
    )

    user_profile = st.selectbox(
        "User Profile",
        ["student", "office", "programmer", "designer", "gamer"],
        index=0,
    )
    budget_class = st.selectbox("Budget Class", ["low", "medium", "high"], index=1)
    price_tier = st.selectbox("Price Tier", ["low", "medium", "high"], index=1)

    story = st.text_area(
        "Laptop Story Description",
        value=(
            "My laptop is very slow when I open multiple tabs and run "
            "applications at the same time. It takes a long time to load "
            "programs and sometimes freezes completely. I mainly use it "
            "for studying and browsing the internet."
        ),
        height=180,
    )

    st.markdown("&nbsp;")
    analyze_clicked = st.button("ANALYZE  →")

    st.markdown(
        "<div style='margin-top:1.5rem;font-family:Courier New,monospace;"
        "font-size:0.7rem;opacity:0.7;letter-spacing:1px;'>"
        "SmartDevice Insight · Activity 4 Portal"
        "</div>", unsafe_allow_html=True
    )


# ------------------------------------------------------------------
# 11. MAIN AREA – HEADER
# ------------------------------------------------------------------
st.markdown('<div class="portal-title">Smart Laptop Upgrade Advisor</div>', unsafe_allow_html=True)
st.markdown('<div class="portal-subtitle">Data Mining · Text Mining · Graph Mining</div>', unsafe_allow_html=True)
st.markdown('<hr class="portal-divider"/>', unsafe_allow_html=True)


# ------------------------------------------------------------------
# 12. RESULTS (only after Analyze)
# ------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state["results"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if analyze_clicked:
    with st.spinner("Analyzing your laptop story..."):
        clean = preprocessor(story)
        cluster_id = predict_cluster(clean, vectorizer, kmeans)
        similar_df, sims_full = get_similar_cases(clean, vectorizer, dataset_tfidf, df_full, top_n=15)
        upgrade, gain, conf = recommend_upgrade_and_gain(similar_df, cluster_id)
        G = build_ego_graph(similar_df, dataset_tfidf, top_n=15, threshold=0.15)
        metrics_df = compute_graph_metrics(G)
        key_cases = identify_key_cases(metrics_df)

        st.session_state["results"] = {
            "story": story,
            "clean": clean,
            "cluster_id": cluster_id,
            "similar_df": similar_df,
            "upgrade": upgrade,
            "gain": gain,
            "confidence": conf,
            "graph": G,
            "metrics_df": metrics_df,
            "key_cases": key_cases,
            "user_profile": user_profile,
            "budget_class": budget_class,
            "price_tier": price_tier,
        }
        st.session_state["chat_history"] = []


results = st.session_state["results"]

if results is None:
    st.info("👈  Enter a laptop story in the sidebar and click **ANALYZE** to begin.")
    st.markdown(
        "<div style='margin-top:2rem;padding:1.5rem;background:#fff8f1;"
        "border-left:4px solid #d4734a;'>"
        "<b style='font-family:Georgia,serif;font-size:1.1rem;color:#1a3a3a;'>"
        "How it works</b><br>"
        "<span style='font-size:0.95rem;color:#444;'>"
        "1. Your story is cleaned and vectorized with TF-IDF.<br>"
        "2. K-Means assigns it to one of 5 user-persona clusters.<br>"
        "3. The 15 most similar historical cases are retrieved via cosine similarity.<br>"
        "4. A NetworkX ego graph is built; PageRank and centralities identify key cases.<br>"
        "5. The chatbot answers questions using only the retrieved evidence."
        "</span></div>",
        unsafe_allow_html=True,
    )
else:
    cluster_id = results["cluster_id"]
    info = CLUSTER_INFO[cluster_id]
    similar_df = results["similar_df"]

    # -------- Cluster Persona Card --------
    st.markdown('<div class="section-header">1. Cluster Persona</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="persona-card">
            <div class="meta">Cluster {cluster_id} · Persona Match</div>
            <h3>{info['name']}</h3>
            <p>{info['description']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------- Recommendation & Gain --------
    st.markdown('<div class="section-header">2. Upgrade Recommendation</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="reco-grid">
            <div class="reco-pill">
                <div class="label">Recommended First Upgrade</div>
                <div class="value">{results['upgrade']}</div>
            </div>
            <div class="reco-pill">
                <div class="label">Estimated Gain Class</div>
                <div class="value">{results['gain']}</div>
            </div>
        </div>
        <div style='font-family:Courier New,monospace;font-size:0.8rem;color:#6b6b6b;
                    margin-bottom:1.5rem;'>
            Confidence: {results['confidence']*100:.0f}% (majority vote across top 10 similar cases)
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------- Two-column: Similar Cases + Text Evidence --------
    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.markdown('<div class="section-header">3. Similar Historical Cases</div>',
                    unsafe_allow_html=True)
        display_df = similar_df.head(10)[[
            "case_id", "similarity", "user_profile", "upgrade_first",
            "gain_class", "kmeans_cluster",
        ]].rename(columns={
            "case_id": "Case ID",
            "similarity": "Similarity",
            "user_profile": "Profile",
            "upgrade_first": "Upgrade",
            "gain_class": "Gain",
            "kmeans_cluster": "Cluster",
        })
        display_df["Similarity"] = display_df["Similarity"].round(3)
        st.dataframe(display_df, hide_index=True, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">4. Text Evidence</div>',
                    unsafe_allow_html=True)
        for _, row in similar_df.head(3).iterrows():
            snippet = str(row["story_text"])[:220]
            if len(str(row["story_text"])) > 220:
                snippet += "..."
            st.markdown(
                f"""
                <div class="evidence">
                    "{snippet}"
                    <span class="cite">
                        ─ Case {int(row['case_id'])} · Cluster {int(row['kmeans_cluster'])} · sim={row['similarity']:.3f}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------- Graph Visual --------
    st.markdown('<div class="section-header">5. Similarity Ego Network</div>',
                unsafe_allow_html=True)
    fig = draw_graph(results["graph"])
    st.pyplot(fig, use_container_width=True)

    # -------- Metrics Table --------
    st.markdown('<div class="section-header">6. Graph Metrics</div>',
                unsafe_allow_html=True)
    st.dataframe(results["metrics_df"], hide_index=True, use_container_width=True)

    kc = results["key_cases"]
    if kc:
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            f"<div class='reco-pill'><div class='label'>Most Representative</div>"
            f"<div class='value' style='font-size:1.2rem'>{kc['most_representative']}</div>"
            f"<div style='font-size:0.7rem;color:#6b6b6b'>highest PageRank</div></div>",
            unsafe_allow_html=True)
        c2.markdown(
            f"<div class='reco-pill'><div class='label'>Most Connected</div>"
            f"<div class='value' style='font-size:1.2rem'>{kc['most_connected']}</div>"
            f"<div style='font-size:0.7rem;color:#6b6b6b'>highest Degree</div></div>",
            unsafe_allow_html=True)
        c3.markdown(
            f"<div class='reco-pill'><div class='label'>Bridge Case</div>"
            f"<div class='value' style='font-size:1.2rem'>{kc['bridge_case']}</div>"
            f"<div style='font-size:0.7rem;color:#6b6b6b'>highest Betweenness</div></div>",
            unsafe_allow_html=True)
        c4.markdown(
            f"<div class='reco-pill'><div class='label'>Unusual Case</div>"
            f"<div class='value' style='font-size:1.2rem'>{kc['unusual_case']}</div>"
            f"<div style='font-size:0.7rem;color:#6b6b6b'>lowest Closeness</div></div>",
            unsafe_allow_html=True)

    # -------- Chatbot --------
    st.markdown('<div class="section-header">7. Evidence-Based Chatbot</div>',
                unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.9rem;color:#555;margin-bottom:0.8rem;'>"
        "Ask anything about the retrieved cases. The bot answers <b>only</b> using "
        "evidence from your similar cases, with full citation."
        "</div>", unsafe_allow_html=True
    )

    # Display chat history
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-user'>{msg['text']}</div>", unsafe_allow_html=True)
        else:
            cite = ""
            if msg.get("case_id") is not None:
                cite = (f"<span class='cite'>─ Case {msg['case_id']} · "
                        f"{msg['cluster']} · short quote</span>")
            st.markdown(
                f"<div class='chat-bot'>{msg['text']}{cite}</div>",
                unsafe_allow_html=True,
            )

    # Chat input
    with st.form(key="chat_form", clear_on_submit=True):
        cc1, cc2 = st.columns([5, 1])
        with cc1:
            user_q = st.text_input("Your question", label_visibility="collapsed",
                                   placeholder="e.g. what RAM upgrade is suggested?")
        with cc2:
            send = st.form_submit_button("Send")
        if send and user_q.strip():
            answer = chatbot_answer(user_q, similar_df)
            st.session_state["chat_history"].append({"role": "user", "text": user_q})
            st.session_state["chat_history"].append({
                "role": "bot",
                "text": answer["answer"],
                "case_id": answer["case_id"],
                "cluster": answer["cluster"],
            })
            st.rerun()
