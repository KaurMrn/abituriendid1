from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def _ensure_dependencies() -> None:
    """Ensure required packages are installed."""
    required_packages = {"plotly": "plotly", "pandas": "pandas"}
    missing = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        st.info(f"Installing missing dependencies: {', '.join(missing)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install"] + missing
        )
        st.rerun()


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "outputs" / "abituriendid_summary"
ANSWERS_PATH = DATA_DIR / "dashboard_answers.csv"
METRICS_PATH = DATA_DIR / "dashboard_metrics.csv"

OVERALL_GROUP_TYPE = "Overall"
OVERALL_GROUP_VALUE = "All"
OVERALL_LABEL = "Kõik vastajad"
SCHOOL_GROUP_TYPE = "School location"
GENDER_GROUP_TYPE = "Gender"
SMALL_N_THRESHOLD = 10

LOCATION_COMPARISON_SERIES = [
    (OVERALL_LABEL, OVERALL_GROUP_TYPE, OVERALL_GROUP_VALUE),
    ("Suurlinnad", SCHOOL_GROUP_TYPE, "Suurlinnad"),
    ("Väikelinnad/maakohad", SCHOOL_GROUP_TYPE, "Väikelinnad/maakohad"),
]

GENDER_ORDER = ["Naine", "Mees", "Mittebinaarne", "Ei soovi avaldada"]

SERIES_COLORS = {
    OVERALL_LABEL: "#4866E8",
    "Suurlinnad": "#1B998B",
    "Väikelinnad/maakohad": "#C77800",
    "Naine": "#D8578A",
    "Mees": "#0E7C86",
    "Mittebinaarne": "#8E63CE",
    "Ei soovi avaldada": "#6B7280",
}

LIKERT_ORDER = [
    "pole üldse nõus",
    "pigem ei ole nõus",
    "ei oska öelda",
    "pigem olen nõus",
    "olen täiesti nõus",
]
YES_NO_ORDER = ["Jah", "Ei", "Ei oska öelda"]
SALARY_ORDER = [
    "Alates 1000",
    "Alates 1500",
    "Alates 2000",
    "Alates 2500",
    "Alates 3000",
    "Alates 3500",
]
APP_ORDER = [
    "Snapchati",
    "Instagrami",
    "Tiktoki",
    "YouTube’i",
    "Mõne Eesti uudisteportaali",
    "Facebooki",
    "Muu",
]


def configure_page() -> None:
    st.set_page_config(
        page_title="Abiturientide küsitluse dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
        }
        h1 {
            font-size: 2rem;
            letter-spacing: 0;
        }
        h2, h3 {
            letter-spacing: 0;
        }
        [data-testid="stMetric"] {
            border: 1px solid #e6e8ec;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            background: #ffffff;
        }
        [data-testid="stMetricLabel"] {
            color: #3f4756;
        }
        [data-testid="stMetricValue"] {
            color: #111827;
        }
        [data-testid="stMetricDelta"] {
            color: #3f4756;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not ANSWERS_PATH.exists() or not METRICS_PATH.exists():
        missing = [
            str(path)
            for path in [ANSWERS_PATH, METRICS_PATH]
            if not path.exists()
        ]
        raise FileNotFoundError("Puuduvad andmefailid: " + ", ".join(missing))

    answers = pd.read_csv(ANSWERS_PATH, encoding="utf-8-sig")
    metrics = pd.read_csv(METRICS_PATH, encoding="utf-8-sig")

    answers["question_index"] = pd.to_numeric(answers["question_index"], errors="coerce")
    answers["count"] = pd.to_numeric(answers["count"], errors="coerce").fillna(0).astype(int)
    answers["percent"] = pd.to_numeric(answers["percent"], errors="coerce")
    answers["n_answered"] = pd.to_numeric(answers["n_answered"], errors="coerce").fillna(0).astype(int)

    for column in ["n_answered", "n_scored"]:
        if column in metrics.columns:
            metrics[column] = pd.to_numeric(metrics[column], errors="coerce").fillna(0).astype(int)
    for column in ["agree_pct", "disagree_pct", "dont_know_pct", "mean_score"]:
        if column in metrics.columns:
            metrics[column] = pd.to_numeric(metrics[column], errors="coerce")

    return answers, metrics


def question_options(answers: pd.DataFrame) -> list[str]:
    ordered = (
        answers[["question_index", "question"]]
        .drop_duplicates()
        .sort_values(["question_index", "question"])
    )
    return ordered["question"].tolist()


def question_label(question: str, answers: pd.DataFrame) -> str:
    row = answers.loc[answers["question"].eq(question), ["question_index"]].head(1)
    if row.empty or pd.isna(row.iloc[0]["question_index"]):
        return question
    return f"{int(row.iloc[0]['question_index'])}. {question}"


def question_type_for(question: str, answers: pd.DataFrame) -> str | None:
    row = answers.loc[answers["question"].eq(question), ["question_type"]].head(1)
    if row.empty or pd.isna(row.iloc[0]["question_type"]):
        return None
    return row.iloc[0]["question_type"]


def render_question_card(
    question: str,
    answers: pd.DataFrame,
    metrics: pd.DataFrame,
    active_series: list[tuple[str, str, str]],
) -> None:
    st.subheader(question_label(question, answers))
    rows_by_label = {
        label: rows_for_question(answers, question, group_type, group_value)
        for label, group_type, group_value in active_series
    }
    distribution = build_distribution(rows_by_label)
    render_distribution_chart(distribution, active_series)


def comparison_series(answers: pd.DataFrame, comparison_mode: str) -> list[tuple[str, str, str]]:
    if comparison_mode == "Koolikoht":
        available_values = set(
            answers.loc[answers["group_type"].eq(SCHOOL_GROUP_TYPE), "group_value"]
            .dropna()
            .drop_duplicates()
            .tolist()
        )
        return [
            series
            for series in LOCATION_COMPARISON_SERIES
            if series[1] != SCHOOL_GROUP_TYPE or series[2] in available_values
        ]

    group_values = (
        answers.loc[answers["group_type"].eq(GENDER_GROUP_TYPE), "group_value"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    order_lookup = {value: index for index, value in enumerate(GENDER_ORDER)}
    ordered_values = sorted(
        group_values,
        key=lambda value: (order_lookup.get(value, len(GENDER_ORDER)), str(value).casefold()),
    )
    return [(OVERALL_LABEL, OVERALL_GROUP_TYPE, OVERALL_GROUP_VALUE)] + [
        (value, GENDER_GROUP_TYPE, value) for value in ordered_values
    ]


def answer_order(question_answers: pd.DataFrame) -> list[str]:
    answers = question_answers["answer"].dropna().drop_duplicates().tolist()
    if is_likert_distribution(question_answers):
        return [answer for answer in LIKERT_ORDER if answer in answers]

    known_orders = [LIKERT_ORDER, YES_NO_ORDER, SALARY_ORDER, APP_ORDER]
    for known_order in known_orders:
        if set(answers).issubset(set(known_order)):
            return [answer for answer in known_order if answer in answers]
    return answers


def rows_for_question(
    answers: pd.DataFrame,
    question: str,
    group_type: str,
    group_value: str,
) -> pd.DataFrame:
    return answers.loc[
        answers["question"].eq(question)
        & answers["group_type"].eq(group_type)
        & answers["group_value"].eq(group_value)
    ].copy()


def metric_for_question(
    metrics: pd.DataFrame,
    question: str,
    group_type: str,
    group_value: str,
) -> pd.Series | None:
    rows = metrics.loc[
        metrics["question"].eq(question)
        & metrics["group_type"].eq(group_type)
        & metrics["group_value"].eq(group_value)
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "–"
    return f"{value * 100:.1f}%"


def pp_delta(subgroup_value: float | int | None, overall_value: float | int | None) -> str | None:
    if subgroup_value is None or overall_value is None:
        return None
    if pd.isna(subgroup_value) or pd.isna(overall_value):
        return None
    delta = (subgroup_value - overall_value) * 100
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.1f} pp vs kõik"


def score_delta(subgroup_value: float | int | None, overall_value: float | int | None) -> str | None:
    if subgroup_value is None or overall_value is None:
        return None
    if pd.isna(subgroup_value) or pd.isna(overall_value):
        return None
    delta = subgroup_value - overall_value
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.2f} vs kõik"


def selected_n(rows: pd.DataFrame) -> int:
    if rows.empty:
        return 0
    return int(rows["n_answered"].iloc[0])


def is_likert_distribution(question_answers: pd.DataFrame) -> bool:
    answers = set(question_answers["answer"].dropna().drop_duplicates())
    return bool(answers) and answers.issubset(set(LIKERT_ORDER))


def distribution_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return rows


def build_distribution(
    rows_by_label: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    overall_rows = rows_by_label[OVERALL_LABEL]
    order = answer_order(overall_rows)
    maps_by_label = {
        label: distribution_rows(rows)
        .set_index("answer")[['count', 'percent']]
        .to_dict("index")
        for label, rows in rows_by_label.items()
    }

    records: list[dict[str, object]] = []
    for answer in order:
        record: dict[str, object] = {"answer": answer}
        for label, value_map in maps_by_label.items():
            values = value_map.get(answer, {"count": 0, "percent": 0})
            percent = values.get("percent", 0)
            percent = 0 if pd.isna(percent) else percent
            record[label] = percent
            record[f"{label} count"] = int(values.get("count", 0))
            if label != OVERALL_LABEL:
                record[f"{label} vahe"] = (percent - record[OVERALL_LABEL]) * 100
        records.append(record)

    return pd.DataFrame(records)


def render_distribution_chart(
    distribution: pd.DataFrame,
    active_series: list[tuple[str, str, str]],
) -> None:
    labels = [label for label, _, _ in active_series]
    chart_df = distribution.melt(
        id_vars=["answer"],
        value_vars=labels,
        var_name="Rühm",
        value_name="Osakaal",
    )
    chart_df["Osakaal, %"] = chart_df["Osakaal"] * 100
    order = distribution["answer"].tolist()

    fig = px.bar(
        chart_df,
        x="Osakaal, %",
        y="answer",
        color="Rühm",
        orientation="h",
        barmode="group",
        text="Osakaal, %",
        category_orders={"answer": list(reversed(order))},
        color_discrete_map=SERIES_COLORS,
        labels={"answer": "Vastus", "Osakaal, %": "Osakaal vastanutest (%)"},
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        textfont=dict(color="black"),
        cliponaxis=False,
    )
    fig.update_layout(
        height=max(380, 72 * len(order)),
        margin=dict(l=12, r=42, t=10, b=10),
        legend_title_text="",
        xaxis=dict(range=[0, max(10, chart_df["Osakaal, %"].max() * 1.18)]),
        yaxis_title="",
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_likert_metrics(
    metrics: pd.DataFrame,
    question: str,
    active_series: list[tuple[str, str, str]],
) -> None:
    overall = metric_for_question(metrics, question, OVERALL_GROUP_TYPE, OVERALL_GROUP_VALUE)
    if overall is None:
        return

    st.subheader("Likerti koondnäitajad")
    rows: list[dict[str, object]] = []
    for label, group_type, group_value in active_series:
        row = metric_for_question(metrics, question, group_type, group_value)
        if row is None:
            continue
        rows.append(
            {
                "Rühm": label,
                "N vastas": int(row["n_answered"]),
                "Nõus": pct(row["agree_pct"]),
                "Nõus: vahe": "–" if label == OVERALL_LABEL else pp_delta(row["agree_pct"], overall["agree_pct"]),
                "Ei nõustu": pct(row["disagree_pct"]),
                "Ei oska öelda": pct(row["dont_know_pct"]),
                "Keskmine skoor": f"{row['mean_score']:.2f}" if pd.notna(row["mean_score"]) else "–",
                "Skoori vahe": "–" if label == OVERALL_LABEL else score_delta(row["mean_score"], overall["mean_score"]),
            }
        )
    compare_table = pd.DataFrame(rows)
    st.dataframe(compare_table, hide_index=True, use_container_width=True)


def main() -> None:
    _ensure_dependencies()
    configure_page()

    try:
        answers, metrics = load_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    questions = question_options(answers)

    st.title("Abiturientide küsitluse dashboard")
    comparison_mode = st.radio(
        "Võrdluse alus",
        ["Koolikoht", "Sugu"],
        index=0,
        horizontal=True,
        label_visibility="visible",
    )
    active_series = comparison_series(answers, comparison_mode)

    st.caption(
        "Võrdle kõikide vastajate tulemusi valitud rühmadega. "
        "Üksikute küsimuste kuvamiseks pole vaja valida eraldi küsimust."
    )

    for row_start in range(0, len(questions), 3):
        cols = st.columns(3)
        for col, question in zip(cols, questions[row_start : row_start + 3]):
            with col:
                render_question_card(question, answers, metrics, active_series)

    with st.expander("Andmete päritolu"):
        st.write(f"Vastuste fail: `{ANSWERS_PATH}`")
        st.write(f"Mõõdikute fail: `{METRICS_PATH}`")
        st.write(
            "Kui lähteandmed muutuvad, käivita enne dashboard'i uuesti "
            "`create_abituriendid_summary.py`."
        )


if __name__ == "__main__":
    main()
