from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "outputs" / "abituriendid_summary"
ANSWERS_PATH = DATA_DIR / "dashboard_answers.csv"
SCHOOL_GENDER_PATH = DATA_DIR / "kool+sugu.csv"

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
    OVERALL_LABEL: "#F7C948",
}
PRIMARY_CHART_COLOR = "#F7C948"
PIE_COLOR_SEQUENCE = ["#F7C948", "#F4D35E", "#F9E2A2", "#FFD66B", "#F7C948"]

LIKERT_ORDER = [
    "pole üldse nõus",
    "pigem ei ole nõus",
    "ei oska öelda",
    "pigem olen nõus",
    "olen täiesti nõus",
]
LIKERT_AGGREGATE_ORDER = ["ei ole nõus", "ei oska öelda", "nõus"]
LIKERT_AGGREGATE_MAP = {
    "pole üldse nõus": "ei ole nõus",
    "pigem ei ole nõus": "ei ole nõus",
    "ei oska öelda": "ei oska öelda",
    "pigem olen nõus": "nõus",
    "olen täiesti nõus": "nõus",
}
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
def load_data() -> pd.DataFrame:
    if not ANSWERS_PATH.exists():
        raise FileNotFoundError("Puudub andmefail: " + str(ANSWERS_PATH))

    answers = pd.read_csv(ANSWERS_PATH, encoding="utf-8-sig")
    answers["question_index"] = pd.to_numeric(answers["question_index"], errors="coerce")
    answers["count"] = pd.to_numeric(answers["count"], errors="coerce").fillna(0).astype(int)
    answers["percent"] = pd.to_numeric(answers["percent"], errors="coerce")
    answers["n_answered"] = pd.to_numeric(answers["n_answered"], errors="coerce").fillna(0).astype(int)
    return answers


@st.cache_data(show_spinner=False)
def load_school_gender_data() -> pd.DataFrame:
    if not SCHOOL_GENDER_PATH.exists():
        raise FileNotFoundError("Puudub andmefail: " + str(SCHOOL_GENDER_PATH))

    data = pd.read_csv(SCHOOL_GENDER_PATH, encoding="utf-8-sig", sep=";")
    data.columns = data.columns.str.strip()
    return data


def build_raw_distribution(
    values: pd.Series,
    order: list[str] | None = None,
) -> pd.DataFrame:
    counts = values.dropna().astype(str).value_counts()
    if order is not None:
        ordered_values = [value for value in order if value in counts.index]
        other_values = [value for value in counts.index if value not in ordered_values]
        counts = counts.reindex(ordered_values + other_values).fillna(0).astype(int)

    total = int(counts.sum()) if not counts.empty else 0
    distribution = pd.DataFrame(
        {
            "answer": counts.index.tolist(),
            OVERALL_LABEL: (counts / total).tolist() if total else [0] * len(counts),
            f"{OVERALL_LABEL} count": counts.tolist(),
        }
    )
    return distribution


def collapse_small_categories(
    distribution: pd.DataFrame,
    percent_col: str = OVERALL_LABEL,
    threshold_pct: float = 4.8,
    other_label: str = "Muu",
) -> pd.DataFrame:
    if distribution.empty:
        return distribution
    # percent_col contains fractions (0-1)
    pct_series = distribution[percent_col] * 100
    small_mask = pct_series < threshold_pct
    if not small_mask.any():
        return distribution

    small_counts = distribution.loc[small_mask, f"{percent_col} count"].sum()
    small_pct = distribution.loc[small_mask, percent_col].sum()

    large = distribution.loc[~small_mask].copy()
    # append the aggregated "Muu" row
    other_row = pd.DataFrame(
        {
            "answer": [other_label],
            percent_col: [small_pct],
            f"{percent_col} count": [int(small_counts)],
        }
    )
    result = pd.concat([large, other_row], ignore_index=True)
    return result


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


def comparison_series(answers: pd.DataFrame, comparison_mode: str) -> list[tuple[str, str, str]]:
    if comparison_mode == "Koolikoht":
        return LOCATION_COMPARISON_SERIES

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


def distribution_rows(rows: pd.DataFrame, use_likert_aggregate: bool) -> pd.DataFrame:
    if not use_likert_aggregate:
        return rows

    aggregated = rows.copy()
    aggregated["answer"] = aggregated["answer"].map(LIKERT_AGGREGATE_MAP).fillna(aggregated["answer"])
    return (
        aggregated.groupby("answer", as_index=False, sort=False)[["count", "percent"]]
        .sum()
    )


def build_distribution(
    rows_by_label: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    overall_rows = rows_by_label[OVERALL_LABEL]
    order = answer_order(overall_rows)
    maps_by_label = {
        label: distribution_rows(rows, False)
        .set_index("answer")[["count", "percent"]]
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
    chart_type: str,
) -> None:
    order = distribution["answer"].tolist()
    distribution["percent"] = distribution[OVERALL_LABEL] * 100

    if chart_type == "Baarid":
        fig = px.bar(
            distribution,
            x="percent",
            y="answer",
            orientation="h",
            category_orders={"answer": order},
            labels={"answer": "Vastus", "percent": "Osakaal vastanutest (%)"},
        )
        fig.update_traces(marker_color=PRIMARY_CHART_COLOR, texttemplate="%{x:.1f}%")
        fig.update_layout(
            xaxis=dict(range=[0, max(10, distribution["percent"].max() * 1.18)]),
            yaxis_title="",
        )
        fig.update_yaxes(categoryorder='array', categoryarray=order)
    else:
        fig = px.pie(
            distribution,
            values="percent",
            names="answer",
            color_discrete_sequence=PIE_COLOR_SEQUENCE,
            labels={"percent": "Osakaal vastanutest (%)", "answer": "Vastus"},
        )
        fig.update_traces(textinfo="percent+label", marker=dict(colors=PIE_COLOR_SEQUENCE))

    fig.update_layout(
        height=max(380, 72 * len(order)),
        margin=dict(l=12, r=42, t=10, b=10),
        legend_title_text="",
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_distribution_table(
    distribution: pd.DataFrame,
    active_series: list[tuple[str, str, str]],
) -> None:
    table = pd.DataFrame({"Vastus": distribution["answer"]})
    for label, _, _ in active_series:
        table[f"{label}: arv"] = distribution[f"{label} count"]
        table[f"{label}: %"] = distribution[label].map(pct)
        if label != OVERALL_LABEL:
            table[f"{label}: vahe"] = distribution[f"{label} vahe"].map(lambda value: f"{value:+.1f} pp")
    st.dataframe(table, hide_index=True, use_container_width=True)


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
    configure_page()

    try:
        answers = load_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    questions = question_options(answers)

    st.title("Abiturientide küsitluse dashboard")
    chart_type = st.radio(
        "Diagrammi tüüp",
        ["Baarid", "Sektordiagramm"],
        horizontal=True,
        label_visibility="visible",
    )

    st.caption(
        "Protsendid arvutatakse nende vastajate põhjal, kes vastasid valitud küsimusele."
    )

    school_gender = load_school_gender_data()
    school_distribution = build_raw_distribution(
        school_gender["Kus sa koolis käid?"],
    )
    school_distribution = collapse_small_categories(school_distribution, threshold_pct=4.8, other_label="Muu")
    gender_distribution = build_raw_distribution(
        school_gender["Olen"],
    )

    first_row, second_row = st.columns(2)
    with first_row:
        st.subheader("Kus sa koolis käid?")
        render_distribution_chart(school_distribution, chart_type)

    with second_row:
        st.subheader("Sugu")
        render_distribution_chart(gender_distribution, chart_type)

    for idx in range(0, len(questions), 2):
        cols = st.columns(2)
        for col_index, question in enumerate(questions[idx : idx + 2]):
            with cols[col_index]:
                st.subheader(question_label(question, answers))
                overall_rows = rows_for_question(
                    answers,
                    question,
                    OVERALL_GROUP_TYPE,
                    OVERALL_GROUP_VALUE,
                )
                distribution = build_distribution({OVERALL_LABEL: overall_rows})
                render_distribution_chart(distribution, chart_type)

    with st.expander("Andmete päritolu"):
        st.write(f"Vastuste fail: `{ANSWERS_PATH}`")
        st.write(
            "Kui lähteandmed muutuvad, käivita enne dashboard'i uuesti "
            "`create_abituriendid_summary.py`."
        )


if __name__ == "__main__":
    main()
