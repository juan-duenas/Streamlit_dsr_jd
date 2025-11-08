import streamlit as st
import pandas as pd

# page config should be set before any UI elements
st.set_page_config(layout="wide")

st.header("Worldwide Analysis of Quality of Life and Economic Factors")
st.subheader(
    "This app enables you to explore the relationships between poverty, life expectancy, "
    "and GDP across various countries and years. Use the panels to select options and interact with the data."
)

# add tabs
tab1, tab2, tab3 = st.tabs(["Global Overview", "Country Deep Dive", "Data Explorer"])

with tab1:
    st.header("Global Overview")

with tab2:
    st.header("Country Deep Dive")

with tab3:
    st.header("Data Explorer")

    # Load dataset from remote URL (GitHub raw)
    DATA_URL = (
        "https://raw.githubusercontent.com/JohannaViktor/streamlit_practical/refs/heads/main/"
        "global_development_data.csv"
    )
    try:
        df = pd.read_csv(DATA_URL)
    except FileNotFoundError:
        st.error(f"Could not load data from '{DATA_URL}'. Make sure the URL is reachable and try again.")
        st.stop()
    except Exception as e:
        st.error(f"Error reading data from '{DATA_URL}': {e}")
        st.stop()

    # Determine available columns for country and year
    country_col = None
    year_col = None
    for candidate in ["country", "Country", "COUNTRY", "country_name"]:
        if candidate in df.columns:
            country_col = candidate
            break
    for candidate in ["year", "Year", "YEAR"]:
        if candidate in df.columns:
            year_col = candidate
            break

    if year_col is None:
        st.error("No year column found in dataset. Expected a column named 'year' (case-insensitive).")
        st.write(df.head())
        st.stop()

    # Country multiselect (optional if dataset has country column)
    if country_col is not None:
        countries = sorted(df[country_col].dropna().unique())
        selected_countries = st.multiselect("Select countries", options=countries, default=countries[:10])
    else:
        st.info("No country column found — showing data filtered only by year.")
        selected_countries = None

    # Year slider
    try:
        min_year = int(df[year_col].min())
        max_year = int(df[year_col].max())
    except Exception:
        st.error("Could not determine numeric year range from the dataset.")
        st.write(df.head())
        st.stop()

    year_range = st.slider("Year range", min_value=min_year, max_value=max_year, value=(min_year, max_year))

    # Filter dataframe
    filtered = df[(df[year_col] >= year_range[0]) & (df[year_col] <= year_range[1])]
    if selected_countries is not None:
        filtered = filtered[filtered[country_col].isin(selected_countries)]

    st.markdown(f"### Filtered data — {len(filtered)} rows")
    st.dataframe(filtered.reset_index(drop=True))

    # Make the filtered dataset downloadable as CSV
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name="filtered_global_development_data.csv",
        mime="text/csv",
    )

# end of file
