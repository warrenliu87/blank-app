import requests
import pandas as pd
import numpy as np
import streamlit as st

import os
import json
import cohere

HICP_BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/prc_hicp_midx"

COUNTRY_NAMES = {
    "AT": "Austria", "BE": "Belgium", "CZ": "Czechia", "DE": "Germany",
    "DK": "Denmark", "EL": "Greece", "ES": "Spain", "FI": "Finland",
    "FR": "France", "IE": "Ireland", "IT": "Italy", "NL": "Netherlands",
    "PL": "Poland", "PT": "Portugal", "SE": "Sweden", "HU": "Hungary",
    "EU27_2020": "EU27"
}

def get_cohere_client():
    api_key = st.secrets.get("COHERE_API_KEY") or os.getenv("COHERE_API_KEY")
    if not api_key:
        return None
    return cohere.ClientV2(api_key=api_key)

BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/une_rt_m"

st.set_page_config(page_title="EU Jobs Radar (Eurostat)", layout="wide")

st.title("EU Jobs Radar")
st.caption("Eurostat unemployment (monthly): compare countries, age groups, trends, and rankings.")

# -------- JSON-stat -> DataFrame (Eurostat-safe converter) --------
def jsonstat_to_df(js: dict) -> pd.DataFrame:
    """
    Convert Eurostat JSON-stat response to a tidy DataFrame.
    Handles Eurostat's sparse 'value' dict where keys are linear indices ("0","1",...).
    """
    dims = js.get("id", [])
    sizes = js.get("size", [])
    if not dims or not sizes:
        return pd.DataFrame()

    dim_meta = js.get("dimension", {})
    if not dim_meta:
        return pd.DataFrame()

    # Build ordered category ids for each dimension
    dim_ids = []
    for d in dims:
        cat = dim_meta[d]["category"]
        idx = cat["index"]  # mapping category_id -> position
        pos_to_id = [None] * len(idx)
        for k, pos in idx.items():
            pos_to_id[pos] = k
        dim_ids.append(pos_to_id)

    # Cartesian product index
    mi = pd.MultiIndex.from_product(dim_ids, names=dims)
    n = len(mi)

    # Dense value array from sparse dict
    values = np.full(n, np.nan, dtype=float)
    v = js.get("value", {})

    if isinstance(v, dict):
        for k, val in v.items():
            try:
                values[int(k)] = val
            except Exception:
                pass
    elif isinstance(v, list):
        # Sometimes it's already dense
        arr = np.array(v, dtype=float)
        if len(arr) == n:
            values = arr
        else:
            # Fallback: resize (rare)
            values[: min(n, len(arr))] = arr[: min(n, len(arr))]
    else:
        return pd.DataFrame()

    df = pd.DataFrame(mi.to_list(), columns=dims)
    df["value"] = values
    return df

@st.cache_data(ttl=3600)
def fetch_hicp_index(geo_list):
    params = []
    for g in geo_list:
        params.append(("geo", g))
    params.extend([
        ("coicop", "CP00"),   # all-items consumer basket
        ("unit", "I15"),      # index, 2015=100
    ])

    r = requests.get(HICP_BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()

    if "value" not in js:
        return pd.DataFrame()

    df = jsonstat_to_df(js)

    if "time" in df.columns and not df.empty:
        df["time"] = pd.to_datetime(
            df["time"].astype(str).str.replace("M", "-", regex=False),
            errors="coerce"
        )

    return df

def compute_latest_inflation_yoy(hicp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per country with:
    - latest_hicp_index
    - inflation_yoy (computed from HICP index)
    """
    if hicp_df.empty:
        return pd.DataFrame(columns=["geo", "latest_hicp_index", "inflation_yoy"])

    hicp_df = hicp_df.dropna(subset=["time"]).sort_values(["geo", "time"])
    wide_hicp = hicp_df.pivot_table(index="time", columns="geo", values="value").sort_index()

    rows = []
    for geo in wide_hicp.columns:
        s = wide_hicp[geo].dropna()
        if len(s) == 0:
            rows.append({"geo": geo, "latest_hicp_index": np.nan, "inflation_yoy": np.nan})
            continue

        latest_idx = float(s.iloc[-1])
        if len(s) >= 13 and s.iloc[-13] != 0:
            inflation_yoy = (float(s.iloc[-1]) / float(s.iloc[-13]) - 1.0) * 100.0
        else:
            inflation_yoy = np.nan

        rows.append({
            "geo": geo,
            "latest_hicp_index": round(latest_idx, 2),
            "inflation_yoy": round(inflation_yoy, 2) if pd.notna(inflation_yoy) else np.nan
        })

    return pd.DataFrame(rows)

def minmax_inverse_score(series: pd.Series) -> pd.Series:
    """
    Lower is better, so inverse min-max scaling.
    """
    s = series.astype(float)
    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=s.index)
    mn, mx = valid.min(), valid.max()
    if mx == mn:
        return pd.Series(1.0, index=s.index)
    return 1 - ((s - mn) / (mx - mn))

def build_country_features(latest_df: pd.DataFrame, inflation_df: pd.DataFrame) -> pd.DataFrame:
    merged = latest_df.merge(inflation_df, on="geo", how="left")

    merged["employment_score"] = minmax_inverse_score(merged["latest"])
    merged["price_score"] = minmax_inverse_score(merged["inflation_yoy"])

    # weighted score for a job seeker: labor market slightly more important than prices
    merged["job_search_score"] = (
        0.65 * merged["employment_score"].fillna(0.5) +
        0.35 * merged["price_score"].fillna(0.5)
    ).round(3)

    merged["country_name"] = merged["geo"].map(COUNTRY_NAMES).fillna(merged["geo"])

    return merged.sort_values("job_search_score", ascending=False).reset_index(drop=True)

def build_retrieved_context(features_df: pd.DataFrame, age: str, sex: str, s_adj: str, latest_month: str) -> str:
    """
    Builds a compact retrieval context for the LLM.
    """
    header = (
        f"Dataset context:\n"
        f"- Unemployment dataset: Eurostat monthly unemployment\n"
        f"- Inflation proxy: Eurostat HICP all-items index, converted to YoY inflation in Python\n"
        f"- Filters: age={age}, sex={sex}, seasonal_adjustment={s_adj}, latest_month={latest_month}\n\n"
        f"Country evidence:\n"
    )

    lines = []
    for _, row in features_df.iterrows():
        lines.append(
            f"- {row['country_name']} ({row['geo']}): "
            f"latest_unemployment={row['latest']}%, "
            f"unemployment_change_12m={row['change_12m']} pp, "
            f"inflation_yoy={row['inflation_yoy']}%, "
            f"job_search_score={row['job_search_score']}"
        )

    return header + "\n".join(lines)

def generate_ai_analysis(
    co_client,
    user_question: str,
    perspective: str,
    retrieved_context: str,
    features_df: pd.DataFrame,
    chat_history: list
):
    top_country = features_df.iloc[0]["country_name"] if not features_df.empty else None
    top_geo = features_df.iloc[0]["geo"] if not features_df.empty else None

    if perspective == "Economist":
        system_prompt = (
            "You are an economic analyst. Use ONLY the retrieved evidence provided. "
            "Do not invent facts. Focus on unemployment trends, inflation pressure, relative country positioning, "
            "and whether labor-market conditions appear to be improving, stable, or worsening."
        )
    else:
        system_prompt = (
            "You are a practical job-search advisor. Use ONLY the retrieved evidence provided. "
            "Do not invent facts. Focus on where a person might prefer to search for work among the selected countries, "
            "balancing unemployment conditions and consumer-price pressure."
        )

    schema_instruction = """
Return a JSON object with exactly these keys:
{
  "answer_markdown": "string",
  "recommended_country_code": "string or null",
  "recommended_country_name": "string or null",
  "country_classification": [
    {
      "geo": "string",
      "label": "improving|stable|worsening|mixed",
      "reason": "string"
    }
  ],
  "key_risks": ["string"],
  "follow_up_question": "string"
}
"""

    prompt = f"""
Retrieved evidence:
{retrieved_context}

User perspective: {perspective}
Heuristic top recommendation from Python preprocessing:
- recommended_country_code={top_geo}
- recommended_country_name={top_country}

User question:
{user_question}

Instructions:
1. Base the answer only on the retrieved evidence above.
2. Explain trade-offs, not just a winner.
3. Use the heuristic recommendation if it is consistent with the evidence.
4. Keep the answer concise but useful.
5. {schema_instruction}
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": prompt})

    res = co_client.chat(
        model="command-a-03-2025",
        messages=messages,
        response_format={"type": "json_object"}
    )

    text = res.message.content[0].text
    return json.loads(text)

@st.cache_data(ttl=3600)
def fetch_unemployment(geo_list, age, sex, s_adj, unit="PC_ACT"):
    # Use repeated query parameters for `geo` (preferred by Eurostat API)
    # Build params as a list of tuples so requests sends multiple `geo` entries.
    params = []
    for g in geo_list:
        params.append(("geo", g))
    params.extend([
        ("age", age),
        ("sex", sex),
        ("s_adj", s_adj),
        ("unit", unit),
    ])

    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()

    if "value" not in js:
        return pd.DataFrame()

    df = jsonstat_to_df(js)

    # Parse Eurostat monthly time like "2024M12" / "2024M1"
    if "time" in df.columns and not df.empty:
        df["time"] = pd.to_datetime(
            df["time"].astype(str).str.replace("M", "-", regex=False),
            errors="coerce"
        )

    return df

# -------- Sidebar controls --------
with st.sidebar:
    st.header("Filters")

    debug = st.checkbox("Show debug info", value=False)

    default_countries = ["ES", "FR", "DE", "IT", "EU27_2020"]
    countries = st.multiselect(
        "Countries (geo)",
        options=sorted(list(set(default_countries + ["PT","NL","BE","SE","PL","IE","EL","AT","FI","DK","CZ","HU"]))),
        default=default_countries,
    )

    age = st.selectbox("Age group (age)", ["TOTAL", "Y15-24", "Y25-74"], index=0)
    sex = st.radio("Sex (sex)", ["T", "M", "F"], index=0, horizontal=True)
    s_adj = st.radio("Seasonal adjustment (s_adj)", ["SA", "NSA"], index=0, horizontal=True)

    months_back = st.slider("History window (months)", min_value=12, max_value=240, value=60, step=12)

# -------- Data load --------
if not countries:
    st.warning("Select at least one country.")
    st.stop()

# (Optional) show the actual API URL for transparency
if debug:
    test_params = {
        "geo": ",".join(countries),
        "age": age,
        "sex": sex,
        "s_adj": s_adj,
        "unit": "PC_ACT",
    }
    st.write("API URL:", requests.Request("GET", BASE_URL, params=test_params).prepare().url)

df = fetch_unemployment(countries, age=age, sex=sex, s_adj=s_adj, unit="PC_ACT")

if debug:
    st.write("Raw df shape:", df.shape)
    if not df.empty:
        st.write(df.head())

# Hard stop if nothing came back
if df.empty:
    st.error("No data returned from Eurostat for the selected filters.")
    st.info("Try: Seasonal adjustment = NSA, Age = TOTAL, or fewer countries.")
    st.stop()

# Ensure time parsing succeeded
if "time" not in df.columns or df["time"].isna().all():
    st.error("Time parsing failed (all dates are missing).")
    st.info("Try: Seasonal adjustment = NSA or Age = TOTAL.")
    st.stop()

# Keep last N months
df = df.dropna(subset=["time"]).sort_values("time")
max_time = df["time"].max()
min_time = max_time - pd.DateOffset(months=months_back)
df = df[df["time"] >= min_time]

if df.empty:
    st.error("No data left after applying the history window filter.")
    st.info("Reduce the history window or change filters (e.g., NSA).")
    st.stop()

# Pivot to wide for charts/tables
wide = df.pivot_table(index="time", columns="geo", values="value").sort_index()

# -------- KPI cards --------
if wide.empty:
    st.error("No data available to display (pivot result is empty).")
    st.info("Try: Seasonal adjustment = NSA, Age = TOTAL, or fewer countries.")
    st.stop()

latest_date = wide.index.max()
latest = (
    wide.tail(1)
    .T
    .rename(columns={latest_date: "latest"})
    .reset_index()
    .rename(columns={"index": "geo"})
)
latest["latest"] = latest["latest"].round(2)

def delta_12m(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) < 13:
        return float("nan")
    return float(series.iloc[-1] - series.iloc[-13])

deltas = {c: delta_12m(wide[c]) for c in wide.columns}
latest["change_12m"] = latest["geo"].map(deltas).round(2)

colA, colB, colC, colD = st.columns(4)
colA.metric("Countries", len(countries))
colB.metric("Latest month", latest_date.strftime("%Y-%m"))
colC.metric("Age", age)
colD.metric("Seasonal adj.", s_adj)

ranked_latest = latest.dropna(subset=["latest"]).sort_values("latest")
best = worst = None
if not ranked_latest.empty:
    best = ranked_latest.iloc[0]
    worst = ranked_latest.iloc[-1]
    c1, c2 = st.columns(2)
    c1.metric("Lowest unemployment (latest)", f"{best['geo']}: {best['latest']}%")
    c2.metric("Highest unemployment (latest)", f"{worst['geo']}: {worst['latest']}%")

st.divider()

# -------- Chart + Table --------
st.subheader("Unemployment rate over time (%)")
st.line_chart(wide)

st.subheader("Latest ranking + 12-month change")
ranked = latest.sort_values(["latest", "geo"]).reset_index(drop=True)
st.dataframe(ranked, use_container_width=True)

csv = ranked.to_csv(index=False).encode("utf-8")
st.download_button("Download ranking (CSV)", data=csv, file_name="eu_jobs_radar_ranking.csv", mime="text/csv")

# -------- Simple narrative (no LLM required) --------
st.subheader("Quick read")
msg = []
if best is not None and worst is not None:
    if pd.notna(best.get("change_12m")) and pd.notna(worst.get("change_12m")):
        msg.append(
            f"Over the last 12 months, {best['geo']} changed by {best['change_12m']} pp "
            f"and {worst['geo']} changed by {worst['change_12m']} pp."
        )
msg.append("Use the filters to compare youth vs total unemployment, or to switch seasonal adjustment.")
st.write(" ".join(msg))
# -------- AI / RAG-style analysis --------
st.divider()
st.subheader("AI Labor Market Analyst")

co_client = get_cohere_client()
if co_client is None:
    st.info("Set COHERE_API_KEY in Streamlit secrets or environment variables to enable AI analysis.")
else:
    perspective = st.radio(
        "Choose analysis perspective",
        ["Economist", "Prospective job searcher"],
        horizontal=True
    )

    hicp_df = fetch_hicp_index(countries)
    inflation_df = compute_latest_inflation_yoy(hicp_df)
    features_df = build_country_features(latest, inflation_df)

    retrieved_context = build_retrieved_context(
        features_df=features_df,
        age=age,
        sex=sex,
        s_adj=s_adj,
        latest_month=latest_date.strftime("%Y-%m")
    )

    # optional: show retrieved evidence for transparency
    with st.expander("Show retrieved evidence used by the AI"):
        st.text(retrieved_context)
        st.dataframe(features_df, use_container_width=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # render old turns
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Ask for an analysis of the selected countries...")

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.spinner("Analyzing selected countries..."):
            result = generate_ai_analysis(
                co_client=co_client,
                user_question=user_question,
                perspective=perspective,
                retrieved_context=retrieved_context,
                features_df=features_df,
                chat_history=st.session_state.chat_history
            )

        answer_md = result.get("answer_markdown", "No answer returned.")
        recommended_country_name = result.get("recommended_country_name")
        recommended_country_code = result.get("recommended_country_code")
        key_risks = result.get("key_risks", [])
        classifications = result.get("country_classification", [])
        follow_up_question = result.get("follow_up_question")

        with st.chat_message("assistant"):
            st.markdown(answer_md)

            if recommended_country_name:
                st.success(
                    f"Recommended country for this perspective: "
                    f"{recommended_country_name} ({recommended_country_code})"
                )

            if classifications:
                st.markdown("**Country classifications**")
                cls_df = pd.DataFrame(classifications)
                st.dataframe(cls_df, use_container_width=True)

            if key_risks:
                st.markdown("**Key risks / caveats**")
                for risk in key_risks:
                    st.write(f"- {risk}")

            if follow_up_question:
                st.markdown(f"**Suggested follow-up:** {follow_up_question}")

        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer_md})