import requests
import pandas as pd
import numpy as np
import streamlit as st

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