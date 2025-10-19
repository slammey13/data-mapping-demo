# mapping_lookup_final_visual_cleanui.py
import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
from openai import OpenAI
import os
from pathlib import Path
import json
import re

st.set_page_config(page_title="Data Mapping Lookup", layout="wide")

# --- Header Styling ---
st.markdown("""
<style>
.header {
    background: linear-gradient(90deg, #0f9d58 0%, #34a853 50%, #66bb6a 100%);
    padding: 10px;
    border-radius: 6px;
    color: white;
    font-size: 20px;
    font-weight: 600;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">🔍 Data Mapping Lookup</div>', unsafe_allow_html=True)
st.write("")

# --- Load mapping file ---
@st.cache_data
def load_mapping(path_candidates):
    for path in path_candidates:
        if os.path.exists(path):
            try:
                return pd.read_excel(path), path
            except Exception:
                try:
                    return pd.read_csv(path), path
                except Exception:
                    pass
    return None, None

candidates = [
    "data_mapping_with_definitions.xlsx",
    "data_mapping.xlsx",
    os.path.join(Path.home(), "Downloads", "data_mapping_with_definitions.xlsx"),
    os.path.join(Path.home(), "Downloads", "data_mapping.xlsx")
]

df, used_path = load_mapping(candidates)
if df is None:
    st.error("No mapping file found. Please ensure your Excel file exists in this folder or Downloads.")
    st.stop()

dw_col = next((c for c in df.columns if "warehouse" in c.lower()), None)
def_col = next((c for c in df.columns if "definition" in c.lower()), None)

if not dw_col:
    st.error("No 'Data Warehouse' column found in mapping file.")
    st.stop()

source_systems = [c for c in df.columns if c not in [dw_col, def_col]]

# --- Create Tabs ---
tab1, tab2 = st.tabs(["🔎 Search by Attribute", "🧠 Search by Definition (AI)"])

# ====================================================================================
#  TAB 1: Fuzzy Attribute Search
# ====================================================================================
with tab1:
    st.markdown("### Search by Attribute")
    user_input = st.text_area("Enter one or more attributes", height=140, placeholder="e.g.\nCust_ID\nOrder_No\nAmt")
    do_search = st.button("Run Fuzzy Search")

    st.sidebar.header("Search Options")
    search_scope = st.sidebar.selectbox("Scope", ["All source systems (auto-detect)"] + source_systems)
    systems_to_search = source_systems if search_scope.startswith("All") else [search_scope]
    top_k = st.sidebar.slider("Top matches per system", 1, 5, 2)
    threshold = st.sidebar.slider("Minimum match score (%)", 40, 100, 70)

    def parse_terms(text):
        return [p.strip() for p in text.replace(",", "\n").split("\n") if p.strip()]

    def build_match_records(df, term, systems, top_k, threshold):
        records = []
        for system in systems:
            choices = df[system].dropna().astype(str).unique().tolist()
            matches = process.extract(term, choices, scorer=fuzz.WRatio, limit=top_k)
            for match_name, score, _ in matches:
                if score >= threshold:
                    row = df[df[system] == match_name].iloc[0]
                    records.append({
                        "Search Term": term,
                        "Matched System": system,
                        "Matched Column": match_name,
                        "Match Score": int(score),
                        "Data Warehouse": row[dw_col],
                        "Definition": row.get(def_col, "")
                    })
        return records

    if do_search:
        terms = parse_terms(user_input)
        if not terms:
            st.warning("Please enter at least one search term.")
        else:
            all_records = []
            for term in terms:
                all_records.extend(build_match_records(df, term, systems_to_search, top_k, threshold))
            if all_records:
                results = pd.DataFrame(all_records)
                st.dataframe(results, use_container_width=True)
                csv = results.to_csv(index=False).encode("utf-8")
                st.download_button("Download results as CSV", csv, "mapping_lookup_results.csv", "text/csv")
            else:
                st.info("No matches found.")

# ====================================================================================
#  TAB 2: AI Definition Search
# ====================================================================================
with tab2:
    st.markdown("### Search by Definition (AI)")
    query = st.text_area("Describe what you're looking for", height=120, placeholder="e.g., Which column stores customer order dates?")
    do_ai_search = st.button("Run AI Search")

    if do_ai_search:
        if not query.strip():
            st.warning("Please enter a description to search.")
        else:
            try:
                client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

                dw_info = "\n".join([
                    f"{r[dw_col]}: {r.get(def_col, '')}"
                    for _, r in df.iterrows()
                    if pd.notna(r[dw_col])
                ])

                prompt = f"""
You are a data warehouse assistant. The following are column names and their technical definitions:

{dw_info}

User query: "{query}"

Return a ranked list (JSON array) of the top 5 matching Data Warehouse columns with reasoning.
Example:
[
  {{ "column": "Customer_Key", "reason": "Contains unique customer identifiers." }},
  {{ "column": "Order_Date", "reason": "Stores the timestamp when an order was placed." }}
]
"""

                model_list = ["gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
                response = None
                selected_model = None

                for model_name in model_list:
                    try:
                        with st.spinner(f"Using {model_name}..."):
                            response = client.chat.completions.create(
                                model=model_name,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.3,
                            )
                            selected_model = model_name
                            break
                    except Exception as e:
                        if "model_not_found" in str(e) or "does not exist" in str(e):
                            continue
                        else:
                            raise

                if response is None:
                    st.error("No available OpenAI model could be used.")
                else:
                    reply = response.choices[0].message.content.strip()
                    reply = re.sub(r"```(json)?", "", reply).strip()

                    parsed = None
                    try:
                        parsed = json.loads(reply)
                    except Exception:
                        lines = re.findall(r'([A-Za-z0-9_]+)\s*[-:]\s*(.+)', reply)
                        if lines:
                            parsed = [{"column": c.strip(), "reason": r.strip()} for c, r in lines]

                    if parsed and isinstance(parsed, list):
                        matched_columns = [p.get("column") for p in parsed if p.get("column")]
                        reasons = {p["column"]: p.get("reason", "") for p in parsed if "column" in p}
                        filtered_df = df[df[dw_col].isin(matched_columns)].copy()
                        filtered_df["AI Match Reason"] = filtered_df[dw_col].map(reasons)

                        top_3 = matched_columns[:3]
                        def highlight_rows(row):
                            if row[dw_col] == top_3[0]:
                                return ['background-color: #a5d6a7'] * len(row)
                            elif len(top_3) > 1 and row[dw_col] == top_3[1]:
                                return ['background-color: #c8e6c9'] * len(row)
                            elif len(top_3) > 2 and row[dw_col] == top_3[2]:
                                return ['background-color: #e8f5e9'] * len(row)
                            else:
                                return [''] * len(row)

                        styled = filtered_df.style.apply(highlight_rows, axis=1)
                        top_col = top_3[0] if top_3 else None
                        if top_col:
                            st.markdown(f"🏆 **Top Match:** `{top_col}` — {reasons.get(top_col, '')}")

                        st.dataframe(styled, use_container_width=True)
                        csv = filtered_df.to_csv(index=False).encode("utf-8")
                        st.download_button("Download results as CSV", csv, "ai_search_results.csv", "text/csv")
                    else:
                        st.warning("The AI response could not be parsed into structured results.")

            except Exception as e:
                st.error(f"AI search failed: {e}")

# ====================================================================================
#  DATA PREVIEW
# ====================================================================================
with st.expander("Preview mapping (first 20 rows)"):
    st.dataframe(df.head(20))
