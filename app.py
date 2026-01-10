import streamlit as st
import random
import uuid
from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Reco Experiment", page_icon="🎬", layout="centered")

# ----------------------------
# PARAMÈTRES EXPÉRIENCE
# ----------------------------
N_RECS = 7  # 7 recommandations par utilisateur

ITEMS = [{"id": f"F{i:02d}", "title": f"Film {i}"} for i in range(1, 11)]

# Recommandations + explications IA (tu remplaceras plus tard par tes vrais contenus)
RECS_POOL = [
    {"id": "R01", "title": "Reco 1", "explanation": "Cette recommandation est proposée car elle partage des thèmes et des genres proches de vos choix initiaux."},
    {"id": "R02", "title": "Reco 2", "explanation": "Ce film est recommandé car il présente une ambiance et une structure narrative similaires à celles de vos films sélectionnés."},
    {"id": "R03", "title": "Reco 3", "explanation": "Cette recommandation repose sur des similarités de genre (ex. thriller, science-fiction) et de thématiques avec vos choix."},
    {"id": "R04", "title": "Reco 4", "explanation": "Ce film est recommandé car il correspond aux genres dominants et aux thèmes principaux des films que vous avez choisis."},
    {"id": "R05", "title": "Reco 5", "explanation": "Cette recommandation s’appuie sur des correspondances entre les éléments clés (genre, intrigue, ton) et vos films sélectionnés."},
    {"id": "R06", "title": "Reco 6", "explanation": "Ce film est recommandé car il partage des caractéristiques communes avec vos choix (thèmes, rythme, type d’intrigue)."},
    {"id": "R07", "title": "Reco 7", "explanation": "Cette recommandation est proposée car elle présente des similarités globales avec vos choix initiaux (genre et thèmes)."},
    {"id": "R08", "title": "Reco 8", "explanation": "Ce film est recommandé car il reprend des éléments de style et de thèmes proches de ceux des films que vous avez sélectionnés."},
    {"id": "R09", "title": "Reco 9", "explanation": "Cette recommandation est basée sur des proximités de contenu (genres, thématiques) avec les films que vous avez choisis."},
    {"id": "R10", "title": "Reco 10", "explanation": "Ce film est recommandé car il correspond à des critères similaires à vos choix (genre, thèmes, ambiance)."},
]

REAL_OPTIONS = ["Films similaires", "Popularité", "Publicité", "Je ne sais pas"]

SHEET_HEADERS = [
    "timestamp",
    "participant_id",
    "rec_index",
    "explanation",
    "choice_1",
    "choice_2",
    "choice_3",
    "rec_id",
    "rec_title",
    "q_trust",
    "q_understand_perceived",
    "q_understand_real",
]


# ----------------------------
# GOOGLE SHEETS HELPERS
# ----------------------------
# ----------------------------
# GOOGLE SHEETS HELPERS (FIX: pas de cache avec ws en param)
# ----------------------------
@st.cache_resource
def get_worksheet():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=scopes
    )
    client = gspread.authorize(creds)
    sh = client.open_by_key(st.secrets["sheets"]["spreadsheet_id"])
    ws = sh.worksheet(st.secrets["sheets"]["worksheet_name"])
    return ws


def ensure_headers(ws):
    headers = ws.row_values(1)
    if not headers or all(h.strip() == "" for h in headers):
        ws.update("A1", [SHEET_HEADERS])


def find_existing_row(ws, participant_id: str, rec_index: int):
    headers = ws.row_values(1)

    # sécurité : si headers incomplets, on évite crash
    if "participant_id" not in headers or "rec_index" not in headers:
        return None

    pid_col = headers.index("participant_id") + 1
    rec_col = headers.index("rec_index") + 1

    pid_vals = ws.col_values(pid_col)[1:]  # sans header
    rec_vals = ws.col_values(rec_col)[1:]

    target_pid = str(participant_id)
    target_rec = str(rec_index)

    for i, (p, r) in enumerate(zip(pid_vals, rec_vals), start=2):
        if p == target_pid and r == target_rec:
            return i
    return None


def upsert_answer_to_sheet(answer_row: dict):
    ws = get_worksheet()
    ensure_headers(ws)

    row_number = find_existing_row(ws, answer_row["participant_id"], answer_row["rec_index"])

    values_in_order = [
        answer_row["timestamp"],
        answer_row["participant_id"],
        answer_row["rec_index"],
        answer_row["explanation"],
        answer_row["choice_1"],
        answer_row["choice_2"],
        answer_row["choice_3"],
        answer_row["rec_id"],
        answer_row["rec_title"],
        answer_row["q_trust"],
        answer_row["q_understand_perceived"],
        answer_row["q_understand_real"],
    ]

    if row_number is None:
        ws.append_row(values_in_order, value_input_option="USER_ENTERED")
    else:
        ws.update(f"A{row_number}:L{row_number}", [values_in_order])


# ----------------------------
# RESET / SESSION INIT
# ----------------------------
def reset_experiment():
    keys_to_clear = list(st.session_state.keys())
    for k in keys_to_clear:
        del st.session_state[k]



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


# ----------------------------
# UI
# ----------------------------
st.title("🎬 Expérience – Recommandation de films")
st.caption("Choisissez 3 films. Ensuite, vous verrez plusieurs recommandations et répondrez à quelques questions.")


# ----------------------------
# STEP 1 — Choix films (exactement 3)
# ----------------------------
if st.session_state.step == 1:
    st.subheader("1) Choisissez 3 films parmi 10")

    options = [f"{f['id']} — {f['title']}" for f in ITEMS]
    selected = st.multiselect("Sélection (3 films)", options)

    if len(selected) < 3:
        st.info("Sélectionnez 3 films pour continuer.")
    elif len(selected) > 3:
        st.error("Vous avez sélectionné plus de 3 films. Merci d’en retirer un.")

    can_continue = (len(selected) == 3)

    if st.button("Continuer ➜", disabled=not can_continue):
        st.session_state.choice_ids = [s.split(" — ")[0] for s in selected]
        st.session_state.choice_titles = [s.split(" — ")[1] for s in selected]

        # Recos (contrôlées) : on en prend N_RECS parmi RECS_POOL
        st.session_state.recs = random.sample(RECS_POOL, k=N_RECS)

        # Mix avec/sans explication (équilibré)
        n_with = N_RECS // 2
        n_without = N_RECS - n_with
        flags = [True] * n_with + [False] * n_without
        random.shuffle(flags)
        st.session_state.rec_flags = flags

        st.session_state.answers = [None] * N_RECS
        st.session_state.current_idx = 0
        st.session_state.last_idx = None
        st.session_state.step = 2
        st.rerun()


# ----------------------------
# STEP 2 — Recommandations + questions
# ----------------------------
elif st.session_state.step == 2:
    idx = st.session_state.current_idx
    rec = st.session_state.recs[idx]
    show_expl = st.session_state.rec_flags[idx]

    st.progress((idx + 1) / N_RECS)
    st.subheader(f"Recommandation {idx + 1} / {N_RECS}")

    # (optionnel) rappeler les choix
    st.write("**Vos films choisis :** " + ", ".join([f"**{t}**" for t in st.session_state.choice_titles]))

    st.markdown(f"## ⭐ {rec['title']}")

    if show_expl:
        st.markdown("**Explication (IA) :**")
        st.markdown(rec["explanation"])
        expl_label = "avec_explication"
    else:
        st.caption("Aucune explication n’est affichée pour cette recommandation.")
        expl_label = "sans_explication"

    st.divider()
    st.subheader("Questions (pour cette recommandation)")

    # --- Texte de confiance selon condition ---
    if show_expl:
        trust_question = "En vous basant sur l’explication fournie par l’IA, dans quelle mesure faites-vous confiance à cette recommandation ?"
    else:
        trust_question = "En vous basant uniquement sur la recommandation (sans explication), dans quelle mesure faites-vous confiance à cette recommandation ?"

    # Keys uniques par reco
    perceived_key = f"perceived_{idx}"
    real_key = f"real_{idx}"
    trust_key = f"trust_{idx}"

    # Initialisation : nouvelle reco => defaults (3/3 + option 0),
    # retour arrière => restaurer valeurs
    if st.session_state.last_idx != idx:
        existing = st.session_state.answers[idx]

        st.session_state[perceived_key] = int(existing["q_understand_perceived"]) if existing else 3
        st.session_state[real_key] = existing["q_understand_real"] if existing else REAL_OPTIONS[0]
        st.session_state[trust_key] = int(existing["q_trust"]) if existing else 3

        st.session_state.last_idx = idx

    # 🔥 Ordre demandé :
    q_understand_perceived = st.slider(
        "Je comprends pourquoi ce film m’est recommandé.",
        1, 5, key=perceived_key
    )

    q_understand_real = st.radio(
        "Pourquoi ce film est-il recommandé ?",
        REAL_OPTIONS, key=real_key
    )

    q_trust = st.slider(
        trust_question,
        1, 5, key=trust_key
    )

    def save_current():
        st.session_state.answers[idx] = {
            "timestamp": datetime.utcnow().isoformat(),
            "participant_id": st.session_state.participant_id,
            "rec_index": idx + 1,
            "explanation": expl_label,
            "choice_1": st.session_state.choice_ids[0],
            "choice_2": st.session_state.choice_ids[1],
            "choice_3": st.session_state.choice_ids[2],
            "rec_id": rec["id"],
            "rec_title": rec["title"],
            "q_trust": q_trust,
            "q_understand_perceived": q_understand_perceived,
            "q_understand_real": q_understand_real
        }

    col1, col2 = st.columns(2)

    with col1:
        if idx > 0 and st.button("⬅️ Précédent"):
            save_current()
            # pas d’écriture sheet ici (moins de latence)
            st.session_state.current_idx -= 1
            st.rerun()

    with col2:
        label = "Suivant ➜" if idx < N_RECS - 1 else "Terminer ✅"
        if st.button(label):
            save_current()

            with st.spinner("Enregistrement en cours..."):
                try:
                    upsert_answer_to_sheet(st.session_state.answers[idx])
                except Exception:
                    st.error("Erreur lors de l’enregistrement dans Google Sheets. Vérifiez la configuration et réessayez.")
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
    st.write("Vos réponses ont bien été enregistrées dans Google Sheets.")
    st.button("Recommencer", on_click=reset_experiment)
