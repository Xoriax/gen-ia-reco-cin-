"""
Streamlit application for the movie recommendation system.
Replaces the previous Flask + HTML/JS frontend with a single Python app:
- "Recommandation" tab: hybrid questionnaire (EF1.1) + SBERT-based results + Gemini justification (EF4.3)
- "Évaluation" tab: dashboard for generated-results evaluation (C5.3)
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from src.recommender.movie_recommender import recommend_movies, load_movie_index
from src.services.tmdb_service import fetch_movie_poster
from src.evaluation import metrics

st.set_page_config(
    page_title="CineMatch - Recommandation IA",
    layout="wide",
)

PERIOD_OPTIONS = {
    "Toutes périodes": None,
    "2020 - aujourd'hui": "present-2020",
    "2015 - 2020": "2020-2015",
    "2010 - 2015": "2015-2010",
    "2000 - 2010": "2010-2000",
    "1980 - 2000": "2000-1980",
    "Avant 1980": "<1980",
}

st.markdown(
    """
    <style>
    .movie-card {
        background: #1c1f2e;
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }
    .movie-score {
        font-weight: 700;
        color: #8ab4f8;
    }
    .justification-box {
        background: linear-gradient(135deg, #3a2f6e 0%, #24345c 100%);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        color: #f0f0f5;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Chargement du référentiel de films (SBERT)...")
def get_index():
    return load_movie_index()


def render_recommendation_tab():
    st.title("CineMatch")
    st.caption("Décrivez le film que vous cherchez, on s'occupe du reste - analyse sémantique + IA générative.")

    with st.form("preferences_form"):
        col1, col2 = st.columns(2)
        with col1:
            description = st.text_area(
                "Décrivez le film idéal (ambiance, style, émotions recherchées)",
                placeholder="ex: un thriller psychologique avec un twist final",
                height=120,
            )
            similar_title = st.text_input("Un film que vous avez aimé (optionnel)", placeholder="ex: Inception")
        with col2:
            action_intensity = st.slider("Intensité de l'action", 1, 5, 3)
            narrative_complexity = st.slider("Complexité narrative", 1, 5, 3)
            darkness = st.slider("Noirceur / gravité", 1, 5, 3)
            realism = st.slider("Réalisme vs fantaisie", 1, 5, 3)

        period_label = st.radio("Période souhaitée", list(PERIOD_OPTIONS.keys()), horizontal=True)
        submitted = st.form_submit_button("Obtenir mes recommandations", width="stretch")

    if submitted:
        with st.spinner("Analyse sémantique en cours..."):
            result = recommend_movies(
                description=description,
                similar_title=similar_title,
                action_intensity=action_intensity,
                narrative_complexity=narrative_complexity,
                darkness=darkness,
                realism=realism,
                period=PERIOD_OPTIONS[period_label],
                top_k=5,
                use_weights=True,
                enable_augmentation=True,
            )
        st.session_state["last_result"] = result
        st.session_state["last_params"] = dict(
            description=description, similar_title=similar_title,
            action_intensity=action_intensity, narrative_complexity=narrative_complexity,
            darkness=darkness, realism=realism, period=PERIOD_OPTIONS[period_label],
        )

    result = st.session_state.get("last_result")
    if not result:
        return

    recommendations = result.get("recommendations", [])
    justification = result.get("genai_justification")

    if justification:
        st.markdown(f'<div class="justification-box"><b>Pourquoi ces films ?</b><br>{justification}</div>',
                     unsafe_allow_html=True)

    st.subheader("Vos recommandations")
    cols = st.columns(min(len(recommendations), 5) or 1)
    for col, rec in zip(cols, recommendations):
        with col:
            poster_url = fetch_movie_poster(rec["title"], rec.get("year"))
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
            if poster_url:
                st.image(poster_url, width="stretch")
            st.markdown(f"**{rec['title']}** ({rec.get('year', '?')})")
            st.caption(rec.get("genre", ""))
            st.markdown(f'<span class="movie-score">Score : {rec["score"]:.0%}</span>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


def render_evaluation_tab():
    st.title("Évaluation des résultats générés")
    st.caption("Preuves d'usage de Gemini : distribution des scores, comparaison avant/après, coûts et limites.")

    log_df = metrics.load_genai_log()
    result = st.session_state.get("last_result")

    st.subheader("1. Distribution des scores de la dernière recommandation")
    if result and result.get("recommendations"):
        dist_df = metrics.compute_score_distribution(result["recommendations"])
        st.plotly_chart(px.bar(dist_df, x="title", y="score", labels={"score": "Score final"}),
                         width="stretch")

        diversity = metrics.compute_genre_diversity(result["recommendations"])
        st.metric("Genres uniques dans le top résultats", diversity["unique_genres"])
        st.caption(", ".join(diversity["genres"]))
    else:
        st.info("Lancez d'abord une recommandation dans l'onglet 'Recommandation'.")

    st.subheader("2. Comparaison avant / après enrichissement Gemini (EF4.1)")
    test_query = st.text_input("Requête courte à tester", value="action")
    if st.button("Comparer avec / sans enrichissement Gemini"):
        with st.spinner("Exécution des deux scénarios..."):
            comparison = metrics.compute_before_after_enrichment(recommend_movies, description=test_query, top_k=3)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Sans enrichissement**")
            top = comparison["without_enrichment"]
            st.write(f"{top['title']} - score {top['score']:.1%}" if top else "Aucun résultat")
        with c2:
            st.markdown("**Avec enrichissement Gemini**")
            top = comparison["with_enrichment"]
            st.write(f"{top['title']} - score {top['score']:.1%}" if top else "Aucun résultat")
        st.metric("Delta de score sur le top résultat", f"{comparison['score_delta']:+.1%}")

    st.subheader("3. Comparaison de réglages (ajustement de la température Gemini)")
    tuning_prompt = st.text_area(
        "Prompt à tester avec deux températures différentes",
        value="Explique en 2 phrases pourquoi Inception plaît aux amateurs de thrillers psychologiques.",
    )
    if st.button("Comparer les réglages"):
        with st.spinner("Appel Gemini avec temperature=0.2 puis 0.9..."):
            temp_comp = metrics.compute_temperature_comparison(tuning_prompt)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Temperature = 0.2 (déterministe)**")
            st.write(temp_comp["temperature_0_2"]["text"])
            st.caption(f"{temp_comp['temperature_0_2']['length']} mots")
        with c2:
            st.markdown("**Temperature = 0.9 (créatif)**")
            st.write(temp_comp["temperature_0_9"]["text"])
            st.caption(f"{temp_comp['temperature_0_9']['length']} mots")

    st.subheader("4. Cache, coût et latence des appels Gemini")
    hit_rate = metrics.compute_cache_hit_rate(log_df)
    latency = metrics.compute_latency_stats(log_df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Appels totaux", hit_rate["total"])
    c2.metric("Cache hit", f"{hit_rate['cache_hit_pct']}%")
    c3.metric("Fallback (sans API)", f"{hit_rate['fallback_pct']}%")
    c4.metric("Latence moyenne (appels réels)", f"{latency['mean_ms']} ms")

    if not log_df.empty:
        pie_df = pd.DataFrame({
            "type": ["Cache", "API réelle", "Fallback"],
            "count": [
                int(log_df["cache_hit"].sum()),
                len(log_df) - int(log_df["cache_hit"].sum()) - int(log_df["fallback"].sum()),
                int(log_df["fallback"].sum()),
            ],
        })
        st.plotly_chart(px.pie(pie_df, names="type", values="count"), width="stretch")

    st.subheader("5. Grille d'évaluation manuelle des dernières générations")
    grid = metrics.append_eval_grid_entries(log_df, n=5)
    if not grid.empty:
        edited = st.data_editor(
            grid,
            column_config={"rating": st.column_config.NumberColumn("Note (1-5)", min_value=1, max_value=5, step=1)},
            width="stretch",
            key="eval_grid_editor",
        )
        if st.button("Enregistrer les notes"):
            metrics.save_eval_grid(edited)
            st.success("Grille d'évaluation enregistrée.")
    else:
        st.info("Aucun appel Gemini réel enregistré pour le moment.")

    st.subheader("6. Limites & risques identifiés")
    st.markdown(
        """
        - **Hallucination** : Gemini ne choisit jamais les films, il commente uniquement le top 3 déjà
          sélectionné par le scoring SBERT, ce qui limite le risque d'invention de titres inexistants.
        - **Dépendance externe / coût** : chaque appel consomme du quota API ; un cache local et un
          fallback automatique évitent les appels redondants ou les pannes bloquantes.
        - **Latence** : environ 0.5 à 2 secondes par appel réel (voir stats ci-dessus), imperceptible en cache.
        - **Biais du modèle** : tendance à privilégier des œuvres anglophones/populaires dans ses reformulations.
        - **Absence de garantie factuelle** : le texte généré est une justification narrative, pas une
          vérité vérifiée. Les métadonnées affichées (année, score, genre) proviennent uniquement du
          référentiel local (CSV + SBERT), jamais de Gemini.
        """
    )


def main():
    with st.spinner("Initialisation du moteur de recommandation..."):
        get_index()

    tab1, tab2 = st.tabs(["Recommandation", "Évaluation"])
    with tab1:
        render_recommendation_tab()
    with tab2:
        render_evaluation_tab()


if __name__ == "__main__":
    main()
