import streamlit as st
import requests
import plotly.graph_objects as go
import time
from datetime import datetime

st.set_page_config(
    page_title="Thumalien — Détecteur de Fake News",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: radial-gradient(circle at 20% 20%, #151826, #0D0E12 60%);
    color: #E8EAF0;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

section[data-testid="stSidebar"] > div:first-child {
    background: rgba(15, 16, 22, 0.92);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.05);
}

.block-container { padding: 2rem 3rem; max-width: 1100px; }

.thu-header { display: flex; align-items: center; gap: 14px; margin-bottom: 2.5rem; padding-bottom: 1.5rem; border-bottom: 1px solid rgba(255,255,255,0.05); }
.thu-logo { font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #F0F1F5; letter-spacing: -0.5px; }
.thu-logo span { color: #4F8EF7; }
.thu-badge { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; font-weight: 600; color: #4F8EF7; background: rgba(79,142,247,0.1); border: 1px solid rgba(79,142,247,0.3); padding: 3px 10px; border-radius: 100px; letter-spacing: 1px; text-transform: uppercase; }

.stat-pill { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #6B7280; background: rgba(20,21,25,0.6); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 10px 16px; display: flex; align-items: center; gap: 8px; transition: all 0.25s ease; }
.stat-pill:hover { transform: translateY(-3px); box-shadow: 0 10px 30px rgba(79,142,247,0.15); border-color: rgba(79,142,247,0.3); }
.stat-pill strong { color: #E8EAF0; font-size: 0.85rem; }

.stTextArea label { display: none; }
.stTextArea textarea { background: rgba(20,21,25,0.6) !important; backdrop-filter: blur(12px) !important; border: 1px solid rgba(255,255,255,0.06) !important; border-radius: 14px !important; color: #E8EAF0 !important; font-family: 'DM Sans', sans-serif !important; font-size: 0.95rem !important; line-height: 1.7 !important; padding: 18px 20px !important; transition: all 0.25s ease !important; resize: none !important; }
.stTextArea textarea:focus { border-color: #4F8EF7 !important; box-shadow: 0 0 0 3px rgba(79,142,247,0.12) !important; }
.stTextArea textarea::placeholder { color: #2D3148 !important; }

.stButton > button { background: #4F8EF7 !important; color: #fff !important; border: none !important; border-radius: 12px !important; font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important; font-size: 0.9rem !important; padding: 14px 28px !important; cursor: pointer !important; transition: all 0.3s ease !important; width: 100% !important; position: relative; overflow: hidden; }
.stButton > button::before { content: ""; position: absolute; inset: 0; background: linear-gradient(120deg, transparent, rgba(255,255,255,0.15), transparent); opacity: 0; transition: 0.4s; }
.stButton > button:hover::before { opacity: 1; }
.stButton > button:hover { background: #3D7DE8 !important; transform: translateY(-2px) !important; box-shadow: 0 10px 40px rgba(79,142,247,0.35) !important; }

.verdict-card { border-radius: 18px; padding: 2rem 2.5rem; margin: 1.5rem 0; position: relative; overflow: hidden; backdrop-filter: blur(14px); transition: transform 0.3s ease, box-shadow 0.3s ease; }
.verdict-card:hover { transform: translateY(-4px); box-shadow: 0 20px 50px rgba(0,0,0,0.5); }
.verdict-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; }
.verdict-fake { background: rgba(255,69,69,0.08); border: 1px solid rgba(255,69,69,0.2); }
.verdict-fake::before { background: linear-gradient(90deg, #FF4545, transparent); }
.verdict-douteux { background: rgba(255,176,32,0.08); border: 1px solid rgba(255,176,32,0.2); }
.verdict-douteux::before { background: linear-gradient(90deg, #FFB020, transparent); }
.verdict-credible { background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.2); }
.verdict-credible::before { background: linear-gradient(90deg, #22C55E, transparent); }
.verdict-icon { font-size: 2.5rem; margin-bottom: 0.5rem; display: block; }
.verdict-label { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.4rem; }
.verdict-fake .verdict-label { color: #FF4545; }
.verdict-douteux .verdict-label { color: #FFB020; }
.verdict-credible .verdict-label { color: #22C55E; }
.verdict-title { font-size: 2.2rem; font-weight: 700; line-height: 1.1; margin-bottom: 0.4rem; color: #F0F1F5; }
.verdict-confidence { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #6B7280; }
.verdict-confidence strong { color: #A0A7B8; }

.score-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1.5rem 0; }
.score-card { background: rgba(20,21,25,0.6); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.05); border-radius: 14px; padding: 1.4rem; text-align: center; transition: transform 0.3s ease, box-shadow 0.3s ease; }
.score-card:hover { transform: translateY(-3px); box-shadow: 0 10px 30px rgba(0,0,0,0.4); }
.score-card-label { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 0.6rem; color: #6B7280; }
.score-card-value { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 700; }
.score-fake { color: #FF4545; }
.score-douteux { color: #FFB020; }
.score-credible { color: #22C55E; }

.progress-wrap { margin: 0.4rem 0; }
.progress-label { display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #6B7280; margin-bottom: 6px; }
.progress-bar-bg { background: rgba(30,33,48,0.8); border-radius: 100px; height: 6px; overflow: hidden; }
.progress-bar-fill { height: 100%; border-radius: 100px; transition: width 1s ease; box-shadow: 0 0 8px currentColor; }

.emotion-card { background: rgba(20,21,25,0.6); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.05); border-radius: 14px; padding: 1.2rem 1.5rem; display: flex; align-items: center; gap: 1.2rem; transition: transform 0.3s ease; }
.emotion-card:hover { transform: translateY(-3px); }
.emotion-icon { font-size: 2.5rem; }
.emotion-label { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; letter-spacing: 2px; text-transform: uppercase; color: #4B5263; margin-bottom: 4px; }
.emotion-value { font-size: 1.4rem; font-weight: 700; color: #F0F1F5; }
.emotion-conf { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #6B7280; margin-top: 2px; }

.section-title { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: #4B5263; margin: 1.8rem 0 1rem; }

.history-item { background: rgba(20,21,25,0.6); backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.05); border-radius: 10px; padding: 12px 14px; margin-bottom: 10px; }
.history-text { font-size: 0.78rem; color: #9CA3AF; margin-bottom: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 220px; }
.history-meta { display: flex; justify-content: space-between; align-items: center; }
.history-badge { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; font-weight: 600; padding: 2px 8px; border-radius: 100px; }
.hb-fake { background: rgba(255,69,69,0.15); color: #FF4545; }
.hb-douteux { background: rgba(255,176,32,0.15); color: #FFB020; }
.hb-credible { background: rgba(34,197,94,0.15); color: #22C55E; }
.history-time { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: #3D4150; }

.placeholder-zone { background: rgba(20,21,25,0.5); backdrop-filter: blur(10px); border: 1px dashed rgba(255,255,255,0.07); border-radius: 20px; padding: 3rem; text-align: center; margin: 1.5rem 0; }
.placeholder-icon { font-size: 2rem; margin-bottom: 1rem; }
.placeholder-text { color: #3D4150; font-size: 0.85rem; line-height: 1.6; }

.thu-divider { border: none; border-top: 1px solid rgba(255,255,255,0.04); margin: 1.5rem 0; }
.sidebar-header { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: #4B5263; margin-bottom: 1rem; padding-bottom: 0.8rem; border-bottom: 1px solid rgba(255,255,255,0.05); }

.thinking-bar { display: flex; align-items: center; gap: 10px; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #4F8EF7; margin: 1rem 0; animation: pulse 1.5s ease-in-out infinite; }
@keyframes pulse { 0%,100% { opacity: 0.4; } 50% { opacity: 1; } }
</style>
""", unsafe_allow_html=True)

# URL API — HuggingFace Spaces
API_URL = "https://thumalien-thumalien-api.hf.space"

if "history" not in st.session_state:
    st.session_state.history = []
if "total_analyses" not in st.session_state:
    st.session_state.total_analyses = 0

EMOTION_COLORS = {
    "anger": "#FF4545", "fear": "#F59E0B", "disgust": "#8B5CF6",
    "joy": "#22C55E", "neutral": "#6B7280", "sadness": "#3B82F6", "surprise": "#EC4899"
}

with st.sidebar:
    st.markdown('<div class="sidebar-header">Historique</div>', unsafe_allow_html=True)
    if not st.session_state.history:
        st.markdown('<p style="color:#3D4150;font-size:0.8rem;">Aucune analyse encore</p>', unsafe_allow_html=True)
    else:
        for item in reversed(st.session_state.history[-10:]):
            label = item["label"]
            badge_class = "hb-fake" if label == "Fake News" else ("hb-douteux" if label == "Douteux" else "hb-credible")
            emoji = item.get("emoji", "")
            st.markdown(f"""
            <div class="history-item">
                <div class="history-text">{item['text'][:55]}...</div>
                <div class="history-meta">
                    <span class="history-badge {badge_class}">{label}</span>
                    <span style="font-size:0.9rem">{emoji}</span>
                    <span class="history-time">{item['time']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="thu-divider">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#4B5263;line-height:2.2;">
        <div>Analyses : <strong style="color:#6B7280">{st.session_state.total_analyses}</strong></div>
        <div>Modèle : <strong style="color:#6B7280">CamemBERT</strong></div>
        <div>Précision : <strong style="color:#6B7280">72%</strong></div>
        <div>Version : <strong style="color:#6B7280">1.0.0</strong></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="thu-header">
    <div class="thu-logo">Thu<span>malien</span></div>
    <div class="thu-badge">v1.0 · Beta</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="stat-pill">📊 <strong>{st.session_state.total_analyses}</strong> analyses</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-pill">🧠 <strong>CamemBERT</strong></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-pill">🎯 <strong>72%</strong> précision</div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-pill">⚡ <strong>Bluesky</strong></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Analyser un post</div>', unsafe_allow_html=True)

text_input = st.text_area(
    "Post", placeholder="Colle ici un post Bluesky, un titre d'article, ou n'importe quel texte à analyser...",
    height=130, label_visibility="collapsed"
)

char_count = len(text_input) if text_input else 0
st.markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#3D4150;text-align:right;margin-top:-0.5rem;margin-bottom:0.8rem;">{char_count}/512</div>', unsafe_allow_html=True)

col_btn1, col_btn2 = st.columns([4, 1])
with col_btn1:
    analyze_btn = st.button("🔍 Analyser", type="primary", use_container_width=True)
with col_btn2:
    st.button("✕", use_container_width=True)

if analyze_btn and text_input.strip():
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown(
        '<div class="thinking-bar">⟳ &nbsp; Analyse intelligente en cours...</div>',
        unsafe_allow_html=True
    )
    time.sleep(1.2)
    thinking_placeholder.empty()

    try:
        response = requests.post(f"{API_URL}/analyze", json={"text": text_input}, timeout=30)
        result    = response.json()
        label     = result["label"]
        confidence= result["confidence"]
        scores    = result["scores"]
        emotion   = result.get("emotion", {})

        st.session_state.history.append({
            "text": text_input, "label": label,
            "confidence": confidence, "time": datetime.now().strftime("%H:%M"),
            "emoji": emotion.get("emoji", "")
        })
        st.session_state.total_analyses += 1

        if label == "Fake News":
            icon, card_class, title = "❌", "verdict-fake", "Fake News Détectée"
        elif label == "Douteux":
            icon, card_class, title = "⚠️", "verdict-douteux", "Contenu Douteux"
        else:
            icon, card_class, title = "✅", "verdict-credible", "Contenu Crédible"

        st.markdown(f"""
        <div class="verdict-card {card_class}">
            <span class="verdict-icon">{icon}</span>
            <div class="verdict-label">{label}</div>
            <div class="verdict-title">{title}</div>
            <div class="verdict-confidence">Confiance : <strong>{confidence*100:.1f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

        col_gauge, col_radar = st.columns(2)

        with col_gauge:
            gauge_color = "#FF4545" if label == "Fake News" else ("#FFB020" if label == "Douteux" else "#22C55E")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                number={"suffix": "%", "font": {"size": 26, "color": "#E8EAF0", "family": "JetBrains Mono"}},
                title={"text": "Confiance", "font": {"color": "#6B7280", "size": 11, "family": "JetBrains Mono"}},
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"color": "#4B5263", "size": 9}},
                    "bar": {"color": gauge_color, "thickness": 0.25},
                    "bgcolor": "rgba(20,21,25,0.6)",
                    "bordercolor": "#1E2130",
                    "steps": [{"range": [0, 100], "color": "rgba(30,33,48,0.5)"}],
                }
            ))
            fig_gauge.update_layout(
                height=220, margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_radar:
            fig_radar = go.Figure(go.Scatterpolar(
                r=[scores['credible']*100, scores['douteux']*100, scores['fake_news']*100],
                theta=['Crédible', 'Douteux', 'Fake News'],
                fill='toself',
                fillcolor='rgba(79,142,247,0.1)',
                line=dict(color='#4F8EF7', width=2),
                marker=dict(color='#4F8EF7', size=6)
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0,100], gridcolor="#1E2130", tickfont=dict(color="#4B5263", size=9)),
                    angularaxis=dict(gridcolor="#1E2130", tickfont=dict(color="#9CA3AF", size=10))
                ),
                height=220, margin=dict(l=40, r=40, t=30, b=30),
                paper_bgcolor="rgba(0,0,0,0)", showlegend=False
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown('<div class="section-title">Scores détaillés</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="score-grid">
            <div class="score-card"><div class="score-card-label">✅ Crédible</div><div class="score-card-value score-credible">{scores['credible']*100:.1f}%</div></div>
            <div class="score-card"><div class="score-card-label">⚠️ Douteux</div><div class="score-card-value score-douteux">{scores['douteux']*100:.1f}%</div></div>
            <div class="score-card"><div class="score-card-label">❌ Fake News</div><div class="score-card-value score-fake">{scores['fake_news']*100:.1f}%</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Répartition visuelle</div>', unsafe_allow_html=True)
        for bar_label, bar_val, bar_color in [
            ("Crédible", scores['credible']*100, "#22C55E"),
            ("Douteux", scores['douteux']*100, "#FFB020"),
            ("Fake News", scores['fake_news']*100, "#FF4545")
        ]:
            st.markdown(f"""
            <div class="progress-wrap">
                <div class="progress-label"><span>{bar_label}</span><span>{bar_val:.1f}%</span></div>
                <div class="progress-bar-bg"><div class="progress-bar-fill" style="width:{bar_val}%;background:{bar_color};"></div></div>
            </div>
            """, unsafe_allow_html=True)

        if emotion:
            em_color = EMOTION_COLORS.get(emotion.get("emotion", "neutral"), "#6B7280")
            st.markdown('<div class="section-title">Analyse émotionnelle</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="emotion-card" style="border-left: 3px solid {em_color};">
                <div class="emotion-icon">{emotion.get('emoji', '😐')}</div>
                <div>
                    <div class="emotion-label">Émotion dominante</div>
                    <div class="emotion-value" style="color:{em_color};">{emotion.get('emotion_fr', 'Neutre')}</div>
                    <div class="emotion-conf">Confiance : {emotion.get('confidence', 0)*100:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.markdown(f"""
        <div style="background:rgba(255,69,69,0.08);border:1px solid rgba(255,69,69,0.2);border-radius:12px;padding:1rem 1.5rem;margin:1rem 0;">
            <div style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#FF4545;margin-bottom:4px;">ERREUR API</div>
            <div style="font-size:0.85rem;color:#9CA3AF;">Erreur : {str(e)}</div>
        </div>
        """, unsafe_allow_html=True)

elif analyze_btn and not text_input.strip():
    st.markdown('<div style="background:rgba(255,176,32,0.06);border:1px solid rgba(255,176,32,0.2);border-radius:12px;padding:1rem 1.5rem;margin:1rem 0;"><div style="font-size:0.85rem;color:#FFB020;">Entre un texte à analyser</div></div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="placeholder-zone">
        <div class="placeholder-icon">🔍</div>
        <div class="placeholder-text">Colle un post Bluesky ou n'importe quel texte<br>et l'IA analysera sa crédibilité en temps réel</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<hr class="thu-divider">
<div style="display:flex;justify-content:space-between;align-items:center;">
    <div style="font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#3D4150;">Thumalien · Mastère BDIA · SUP DE VINCI 2025</div>
    <div style="font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#3D4150;">CamemBERT · F1=0.67 · Bluesky NLP</div>
</div>
""", unsafe_allow_html=True)