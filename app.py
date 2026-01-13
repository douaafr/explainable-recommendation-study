import streamlit as st
import random
import uuid
from datetime import datetime
import os
from pathlib import Path

import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Reco Experiment", page_icon="🎬", layout="centered")


# =========================================================
# PARAMÈTRES EXPÉRIENCE
# =========================================================
N_RECS = 8

REAL_OPTIONS = ["Films similaires", "Popularité", "Nouveauté", "Publicité", "Je ne sais pas"]

# Contrôle des conditions:
# - reco 1 = avec explication
# - au moins 4 avec explication
# - au moins 3 sans explication
# - 1 aléatoire
MIN_WITH_EXPL = 4
MIN_WITHOUT_EXPL = 3

# "Nouveauté" : on force au moins 1 reco récente affichée AVEC explication
NOVELTY_MIN_YEAR = 2023


# =========================================================
# POSTERS pour les recommandations
# =========================================================
POSTERS_DIR = Path("posters")  # dossier local à côté de app.py
POSTER_EXTS = [".jpg", ".jpeg", ".png", ".webp"]


def get_poster_path(rec_id: str):
    """
    Retourne le chemin du poster pour une reco (ex: posters/R02.jpg),
    ou None si introuvable.
    """
    if not rec_id:
        return None
    for ext in POSTER_EXTS:
        p = POSTERS_DIR / f"{rec_id}{ext}"
        if p.exists():
            return p
    return None


# =========================================================
# DONNÉES 
# =========================================================
def normalize_movie(m: dict) -> dict:
    mm = dict(m)
    mm.setdefault("genres", [])
    mm.setdefault("year", None)
    mm.setdefault("pop_score", 3)  # 1..5
    if not isinstance(mm["genres"], list):
        mm["genres"] = [str(mm["genres"])]
    return mm


ITEMS = [
    {"id": "S01", "title": "Dune", "year": 2021, "genres": ["Science Fiction", "Adventure"], "pop_score": 5},
    {"id": "S02", "title": "Oppenheimer", "year": 2023, "genres": ["Biography", "Drama", "Thriller"], "pop_score": 5},
    {"id": "S03", "title": "Interstellar", "year": 2014, "genres": ["Science Fiction", "Drama", "Adventure"], "pop_score": 5},
    {"id": "S04", "title": "Inception", "year": 2010, "genres": ["Science Fiction", "Thriller", "Action"], "pop_score": 5},
    {"id": "S05", "title": "The Matrix", "year": 1999, "genres": ["Science Fiction", "Action"], "pop_score": 5},
    {"id": "S06", "title": "The Dark Knight", "year": 2008, "genres": ["Action", "Crime", "Drama"], "pop_score": 5},
    {"id": "S07", "title": "The Shawshank Redemption", "year": 1994, "genres": ["Drama"], "pop_score": 5},
    {"id": "S08", "title": "Intouchables", "year": 2011, "genres": ["Comedy", "Drama"], "pop_score": 5},
    {"id": "S09", "title": "Spider-Man: Into the Spider-Verse", "year": 2018, "genres": ["Animation", "Action", "Adventure"], "pop_score": 5},
    {"id": "S10", "title": "Avatar: The Way of Water", "year": 2022, "genres": ["Science Fiction", "Adventure"], "pop_score": 5},
]

RECS_POOL = [
    {"id": "R01", "title": "The Truman Show", "year": 1998, "genres": ["Comedy", "Drama", "Science Fiction"], "pop_score": 5},
    {"id": "R02", "title": "Memento", "year": 2000, "genres": ["Mystery", "Thriller"], "pop_score": 5},
    {"id": "R03", "title": "1917", "year": 2019, "genres": ["War", "Drama"], "pop_score": 5},
    {"id": "R04", "title": "Saving Private Ryan", "year": 1998, "genres": ["War", "Drama"], "pop_score": 5},
    {"id": "R05", "title": "Schindler's List", "year": 1993, "genres": ["Biography", "Drama", "History"], "pop_score": 5},
    {"id": "R06", "title": "Se7en", "year": 1995, "genres": ["Crime", "Mystery", "Thriller"], "pop_score": 5},
    {"id": "R07", "title": "The Usual Suspects", "year": 1995, "genres": ["Crime", "Mystery", "Thriller"], "pop_score": 5},
    {"id": "R08", "title": "The Godfather", "year": 1972, "genres": ["Crime", "Drama"], "pop_score": 5},
    {"id": "R09", "title": "Joker", "year": 2019, "genres": ["Crime", "Drama", "Thriller"], "pop_score": 5},
    {"id": "R10", "title": "The Wolf of Wall Street", "year": 2013, "genres": ["Biography", "Comedy", "Crime"], "pop_score": 5},
    {"id": "R11", "title": "Before Sunrise", "year": 1995, "genres": ["Romance", "Drama"], "pop_score": 4},
    {"id": "R12", "title": "Pan's Labyrinth", "year": 2006, "genres": ["Fantasy", "War", "Drama"], "pop_score": 4},
    {"id": "R13", "title": "The Lord of the Rings: The Return of the King", "year": 2003, "genres": ["Fantasy", "Adventure"], "pop_score": 5},
    {"id": "R14", "title": "Star Wars: Episode IV – A New Hope", "year": 1977, "genres": ["Science Fiction", "Adventure"], "pop_score": 5},
    {"id": "R15", "title": "Spider-Man: Across the Spider-Verse", "year": 2023, "genres": ["Animation", "Action", "Adventure"], "pop_score": 5},
    {"id": "R16", "title": "3 Idiots", "year": 2009, "genres": ["Comedy", "Drama"], "pop_score": 5},
    {"id": "R17", "title": "Rush", "year": 2013, "genres": ["Biography", "Drama", "Sport"], "pop_score": 4},
    {"id": "R18", "title": "Avengers: Endgame", "year": 2019, "genres": ["Action", "Adventure", "Science Fiction"], "pop_score": 5},
    {"id": "R19", "title": "Spider-Man: No Way Home", "year": 2021, "genres": ["Action", "Adventure", "Science Fiction"], "pop_score": 5},
    {"id": "R20", "title": "Dune: Part Two", "year": 2024, "genres": ["Science Fiction", "Adventure"], "pop_score": 5},
]

ITEMS = [normalize_movie(x) for x in ITEMS]
RECS_POOL = [normalize_movie(x) for x in RECS_POOL]


# =========================================================
# GOOGLE SHEETS
# =========================================================
@st.cache_resource
def get_worksheet():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    client = gspread.authorize(creds)
    sh = client.open_by_key(st.secrets["sheets"]["spreadsheet_id"])
    ws = sh.worksheet(st.secrets["sheets"]["worksheet_name"])
    return ws


def col_to_a1(col: int) -> str:
    letters = ""
    while col:
        col, rem = divmod(col - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def normalize_header_row(ws):
    """
    Corrige le cas où la ligne 1 a des cellules vides au début (ex: A1 vide, headers en B1...).
    On "décale" les headers non vides vers A1.
    """
    raw = ws.row_values(1)
    if not raw:
        return

    cleaned = [str(h).strip() for h in raw]
    # si tout est vide -> rien à faire
    if all(h == "" for h in cleaned):
        return

    # trouve le premier header non vide
    first_non_empty = None
    for i, h in enumerate(cleaned):
        if h != "":
            first_non_empty = i
            break

    # s'il y a des vides au début, on réécrit à partir de A1
    if first_non_empty and first_non_empty > 0:
        new_headers = [h for h in cleaned if h != ""]
        ws.update("A1", [new_headers])


def get_headers(ws):
    normalize_header_row(ws)
    headers = ws.row_values(1)
    headers_clean = [str(h).strip() for h in headers if str(h).strip() != ""]
    return headers_clean


def ensure_columns_exist(ws, required_cols: list[str]):
    normalize_header_row(ws)

    headers = ws.row_values(1)
    if not headers or all(str(h).strip() == "" for h in headers):
        ws.update("A1", [required_cols])
        return

    headers_clean = [str(h).strip() for h in headers if str(h).strip() != ""]
    missing = [c for c in required_cols if c not in headers_clean]
    if missing:
        start_col = len(headers_clean) + 1
        last_col = start_col + len(missing) - 1
        ws.update(f"{col_to_a1(start_col)}1:{col_to_a1(last_col)}1", [missing])


def find_existing_row(ws, participant_id: str, rec_index: int):
    headers = get_headers(ws)
    if "participant_id" not in headers or "rec_index" not in headers:
        return None

    pid_col = headers.index("participant_id") + 1
    rec_col = headers.index("rec_index") + 1

    pid_vals = ws.col_values(pid_col)[1:]
    rec_vals = ws.col_values(rec_col)[1:]

    for i, (p, r) in enumerate(zip(pid_vals, rec_vals), start=2):
        if str(p) == str(participant_id) and str(r) == str(rec_index):
            return i
    return None


def upsert_row_by_header(ws, row_dict: dict):
    headers = get_headers(ws)

    values = [""] * len(headers)
    for k, v in row_dict.items():
        if k in headers:
            values[headers.index(k)] = v

    row_number = find_existing_row(ws, row_dict.get("participant_id", ""), row_dict.get("rec_index", ""))

    last_col_letter = col_to_a1(len(headers))
    if row_number is None:
        ws.append_row(values, value_input_option="USER_ENTERED")
    else:
        ws.update(f"A{row_number}:{last_col_letter}{row_number}", [values])


# =========================================================
# RECO + RAISONS 
# =========================================================
def shared_genres(a: dict, b: dict):
    return list(set(a.get("genres", [])) & set(b.get("genres", [])))


def era_bucket(year: int) -> str:
    if not year:
        return "Unknown"
    decade = (year // 10) * 10
    return f"{decade}s"


def score_and_reasons(selected_movies: list[dict], rec: dict):
    """
    reasons_struct: max 3 raisons candidates, mais on n'en AFFICHERA qu'UNE seule.
    """
    reasons = []
    score = 0

    # same_genre: lié strictement à un film choisi
    genre_matches = []
    for sm in selected_movies:
        g = shared_genres(sm, rec)
        if g:
            genre_matches.append((sm, g))
    if genre_matches:
        sm, g = random.choice(genre_matches)
        reasons.append({
            "type": "same_genre",
            "selected_title": sm["title"],
            "selected_id": sm["id"],
            "match_value": random.choice(g),
        })
        score += 4

    # same_era: lié à un film choisi (décennie)
    if rec.get("year") and any(sm.get("year") for sm in selected_movies):
        candidates = [sm for sm in selected_movies if sm.get("year") and era_bucket(sm["year"]) == era_bucket(rec["year"])]
        if candidates:
            sm = random.choice(candidates)
            reasons.append({
                "type": "same_era",
                "selected_title": sm["title"],
                "selected_id": sm["id"],
                "match_value": era_bucket(rec["year"]),
            })
            score += 2

    # popularity
    if rec.get("pop_score", 3) >= 5:
        reasons.append({"type": "popularity"})
        score += 1

    # novelty
    if rec.get("year") and rec["year"] >= NOVELTY_MIN_YEAR:
        reasons.append({"type": "novelty", "match_value": rec["year"]})
        score += 2

    # on garde au plus 3 candidates
    random.shuffle(reasons)  # ✅ varie l’ordre => pas toujours la même
    reasons_struct = reasons[:3]
    return score, reasons_struct


def reason_to_expected_label(reason_type: str) -> str:
    if reason_type in ["same_genre", "same_era"]:
        return "Films similaires"
    if reason_type == "popularity":
        return "Popularité"
    if reason_type == "novelty":
        return "Nouveauté"
    return ""


def generate_explanation_text_personal(rec: dict, reasons_struct: list[dict]):
    
    
    if not reasons_struct:
        return ("Je vous propose ce film car il présente des éléments proches de vos choix.", [], "")

    r = random.choice(reasons_struct)
    starters = [
        "Je vous le recommande car",
        "Je vous propose ce film parce que",
        "Ce choix peut vous plaire car",
        "Je vous suggère ce film car",
    ]
    start = random.choice(starters)

    if r["type"] == "same_genre":
        text = (
            f"{start} vous avez sélectionné « {r['selected_title']} », "
            f"et ce film partage le genre « {r['match_value']} »."
        )
        shown = ["same_genre"]

    elif r["type"] == "same_era":
        text = (
            f"{start} il se situe dans la même période ({r['match_value']}) "
            f"que « {r['selected_title']} »."
        )
        shown = ["same_era"]

    elif r["type"] == "novelty":
        text = f"{start} c’est une sortie récente ({r['match_value']})."
        shown = ["novelty"]

    elif r["type"] == "popularity":
        text = f"{start} c’est un film très populaire, souvent apprécié par un large public."
        shown = ["popularity"]

    else:
        text = "Je vous propose ce film car il présente des éléments proches de vos choix."
        shown = []

    expected = reason_to_expected_label(shown[0]) if shown else ""
    return text, shown, expected


def pick_recommendations(selected_movies: list[dict], pool: list[dict], n_recs: int):
    scored = []
    for r in pool:
        s, reasons_struct = score_and_reasons(selected_movies, r)
        rr = dict(r)
        rr["reasons_struct"] = reasons_struct
        rr["reasons"] = [x["type"] for x in reasons_struct]
        expl_text, shown, expected = generate_explanation_text_personal(rr, reasons_struct)
        rr["explanation_text"] = expl_text
        rr["shown_reasons"] = shown
        rr["expected_reason"] = expected
        scored.append((s, rr))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [rr for _, rr in scored[:max(n_recs * 4, n_recs)]]
    return random.sample(top, k=min(n_recs, len(top)))


def force_one_novelty(recs: list[dict], pool: list[dict]) -> list[dict]:
    """Force au moins 1 film récent dans les recs."""
    if any(r.get("year") and r["year"] >= NOVELTY_MIN_YEAR for r in recs):
        return recs

    candidates = [r for r in pool if r.get("year") and r["year"] >= NOVELTY_MIN_YEAR]
    if not candidates:
        return recs

    novelty = max(candidates, key=lambda x: x.get("year", 0))
    # ⚠️ on remplace la dernière reco par un dict complet cohérent
    recs[-1] = dict(novelty)
    return recs


def build_flags(n_recs: int):
    """
    8 recos:
    - 4 True
    - 3 False
    - 1 random
    + flags[0] = True
    """
    flags = [True] * MIN_WITH_EXPL + [False] * MIN_WITHOUT_EXPL
    flags.append(random.choice([True, False]))
    random.shuffle(flags)
    flags[0] = True
    return flags[:n_recs]


# =========================================================
# SESSION
# =========================================================
def reset_experiment():
    st.session_state.clear()
    st.rerun()


if "participant_id" not in st.session_state:
    st.session_state.participant_id = str(uuid.uuid4())[:8]
if "step" not in st.session_state:
    st.session_state.step = 1
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "answers" not in st.session_state:
    st.session_state.answers = [None] * N_RECS
if "last_idx" not in st.session_state:
    st.session_state.last_idx = None


# =========================================================
# UI
# =========================================================
st.title("🎬 Expérience – Recommandation de films")
st.caption("Choisissez 3 films. Ensuite, vous verrez plusieurs recommandations et répondrez à quelques questions.")


# ----------------------------
# STEP 1 — Choix films
# ----------------------------
if st.session_state.step == 1:
    st.subheader("1) Choisissez 3 films parmi 10")

    options = [f"{m['id']} — {m['title']}" for m in ITEMS]
    selected = st.multiselect("Sélection (3 films)", options)

    if len(selected) < 3:
        st.info("Sélectionnez 3 films pour continuer.")
    elif len(selected) > 3:
        st.error("Vous avez sélectionné plus de 3 films. Merci d’en retirer un.")

    can_continue = (len(selected) == 3)

    if st.button("Continuer ➜", disabled=not can_continue):
        choice_ids = [s.split(" — ")[0] for s in selected]
        selected_movies = [m for m in ITEMS if m["id"] in choice_ids]

        st.session_state.choice_ids = [m["id"] for m in selected_movies]
        st.session_state.choice_titles = [m["title"] for m in selected_movies]
        st.session_state.selected_movies = selected_movies

        recs = pick_recommendations(selected_movies, RECS_POOL, N_RECS)
        recs = force_one_novelty(recs, RECS_POOL)

        # recompute explications / raisons (au cas où nouveauté a été forcée)
        recs_fixed = []
        for r in recs:
            s, reasons_struct = score_and_reasons(selected_movies, r)
            rr = dict(r)
            rr["reasons_struct"] = reasons_struct
            rr["reasons"] = [x["type"] for x in reasons_struct]
            expl_text, shown, expected = generate_explanation_text_personal(rr, reasons_struct)
            rr["explanation_text"] = expl_text
            rr["shown_reasons"] = shown
            rr["expected_reason"] = expected
            recs_fixed.append(rr)

        st.session_state.recs = recs_fixed

        flags = build_flags(N_RECS)

        # force la nouveauté à être avec explication
        for i, r in enumerate(st.session_state.recs):
            if r.get("year") and r["year"] >= NOVELTY_MIN_YEAR:
                flags[i] = True
                # et on force l'explication à "novelty" si présent
                r["shown_reasons"] = ["novelty"]
                r["expected_reason"] = "Nouveauté"
                r["explanation_text"] = f"Je vous propose ce film parce que c’est une sortie récente ({r['year']})."
                break

        # reco 1 toujours avec explication
        flags[0] = True
        st.session_state.rec_flags = flags

        st.session_state.answers = [None] * N_RECS
        st.session_state.current_idx = 0
        st.session_state.last_idx = None
        st.session_state.step = 2
        st.rerun()


# ----------------------------
# STEP 2 — Recos + questions
# ----------------------------
elif st.session_state.step == 2:
    idx = st.session_state.current_idx
    rec = st.session_state.recs[idx]
    show_expl = st.session_state.rec_flags[idx]

    st.progress((idx + 1) / N_RECS)
    st.subheader(f"Recommandation {idx + 1} / {N_RECS}")
    st.write("**Vos films choisis :** " + ", ".join([f"**{t}**" for t in st.session_state.choice_titles]))

    # ✅ POSTER + TITRE (avant explication et avant questions)
    poster_path = get_poster_path(rec.get("id", ""))

    if poster_path:
        col_img, col_title = st.columns([1, 3], vertical_alignment="center")
        with col_img:
            st.image(str(poster_path), use_container_width=True)
        with col_title:
            st.markdown(f"## ⭐ {rec.get('title','')}")
    else:
        st.markdown(f"## ⭐ {rec.get('title','')}")

    if show_expl:
        st.markdown("**Explication (IA) :**")
        st.markdown(rec.get("explanation_text", ""))
        condition_label = "avec_explication"
    else:
        st.caption("Aucune explication n’est affichée pour cette recommandation.")
        condition_label = "sans_explication"

    st.divider()
    st.subheader("Questions (pour cette recommandation)")

    if show_expl:
        trust_question = "En vous basant sur l’explication fournie par l’IA, dans quelle mesure faites-vous confiance à cette recommandation ?"
    else:
        trust_question = "En vous basant uniquement sur la recommandation (sans explication), dans quelle mesure faites-vous confiance à cette recommandation ?"

    perceived_key = f"perceived_{idx}"
    real_key = f"real_{idx}"
    trust_key = f"trust_{idx}"

    if st.session_state.last_idx != idx:
        existing = st.session_state.answers[idx]
        st.session_state[perceived_key] = int(existing["q_understand_perceived"]) if existing else 3
        st.session_state[real_key] = existing["q_understand_real"] if existing else REAL_OPTIONS[0]
        st.session_state[trust_key] = int(existing["q_trust"]) if existing else 3
        st.session_state.last_idx = idx

    # ordre: perçue -> réelle -> confiance
    q_understand_perceived = st.slider("Je comprends pourquoi ce film m’est recommandé.", 1, 5, key=perceived_key)
    q_understand_real = st.radio("Pourquoi ce film est-il recommandé ?", REAL_OPTIONS, key=real_key)
    q_trust = st.slider(trust_question, 1, 5, key=trust_key)

    def save_local():
        st.session_state.answers[idx] = {
            "q_understand_perceived": q_understand_perceived,
            "q_understand_real": q_understand_real,
            "q_trust": q_trust,
        }

    def build_row_for_sheet():
        c1_id, c2_id, c3_id = st.session_state.choice_ids
        c1_title, c2_title, c3_title = st.session_state.choice_titles

        shown = rec.get("shown_reasons", [])
        shown_compact = ",".join(shown)
        expected = rec.get("expected_reason", "")

        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "participant_id": st.session_state.participant_id,
            "rec_index": idx + 1,
            "condition": condition_label,

            "choice_1": c1_id,
            "choice_2": c2_id,
            "choice_3": c3_id,

            "rec_id": rec.get("id", ""),
            "rec_title": rec.get("title", ""),

            "q_understand_perceived": q_understand_perceived,
            "q_understand_real": q_understand_real,
            "q_trust": q_trust,

            "shown_reasons": shown_compact,
            "shown_explanation_text": rec.get("explanation_text", "") if show_expl else "",
            "expected_reason": expected,

            "all_reasons": ",".join(rec.get("reasons", [])),

            "choice_1_title": c1_title,
            "choice_2_title": c2_title,
            "choice_3_title": c3_title,
        }

        # alias si tu as une colonne "shown_explanation"
        row["shown_explanation"] = row["shown_explanation_text"]
        return row

    col1, col2 = st.columns(2)

    with col1:
        if idx > 0 and st.button("⬅️ Précédent"):
            save_local()
            st.session_state.current_idx -= 1
            st.rerun()

    with col2:
        label = "Suivant ➜" if idx < N_RECS - 1 else "Terminer ✅"
        if st.button(label):
            save_local()

            ws = get_worksheet()
            ensure_columns_exist(ws, [
                "timestamp", "participant_id", "rec_index", "condition",
                "choice_1", "choice_2", "choice_3",
                "rec_id", "rec_title",
                "q_trust", "q_understand_perceived", "q_understand_real",
                "shown_reasons", "shown_explanation_text",
                "shown_explanation",
                "expected_reason",
                "all_reasons",
                "choice_1_title", "choice_2_title", "choice_3_title"
            ])

            row_dict = build_row_for_sheet()

            with st.spinner("Enregistrement en cours..."):
                try:
                    upsert_row_by_header(ws, row_dict)
                except Exception as e:
                    st.error("Erreur lors de l’enregistrement dans Google Sheets.")
                    st.exception(e)
                    st.stop()

            if idx < N_RECS - 1:
                st.session_state.current_idx += 1
                st.rerun()
            else:
                st.session_state.step = 3
                st.rerun()


# ----------------------------
# STEP 3 — Fin
# ----------------------------
else:
    st.success("Merci ! L’expérience est terminée ✅")
    st.write("Vos réponses ont bien été enregistrées.")
    if st.button("Recommencer"):
        reset_experiment()
