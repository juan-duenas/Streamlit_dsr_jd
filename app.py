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

    st.subheader("Input or modify the three fields below")
    # User inputs in three columns
    col1, col2, col3 = st.columns(3)
    with col1:
        gdp = st.number_input("GDP per capita", min_value=0.0, value=10000.0, step=100.0)
    with col2:
        headcount = st.number_input("Headcount ratio (upper mid income poverty line)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
    with col3:
        year = st.number_input("Year", min_value=1900, max_value=2100, value=2020, step=1)

    # Model prediction (load pre-trained model from pickle)
    import numpy as np
    import pickle
    import plotly.express as px
    FEATURES = ['GDP per capita', 'headcount_ratio_upper_mid_income_povline', 'year']

    @st.cache_resource
    def load_model():
        with open("rf_lifeexp_model.pkl", "rb") as f:
            return pickle.load(f)

    try:
        model = load_model()
        model_ready = True

        # Add prediction button here, between inputs and feature importance
        if st.button("Predict Life Expectancy"):
            if model_ready:
                input_array = np.array([[gdp, headcount, year]])
                pred = model.predict(input_array)[0]
                st.success(f"Predicted Life Expectancy: {pred:.2f} years")
            else:
                st.warning("Model is not available. Please train and save the model first by running model.py.")
        
        # Display feature importances below prediction
        st.subheader("Feature Importance")
        # Map feature names to user-friendly labels
        feature_label_map = {
            'GDP per capita': 'GDP per capita',
            'headcount_ratio_upper_mid_income_povline': 'Headcount ratio (upper mid income poverty line)',
            'year': 'Year'
        }
        importances = pd.DataFrame({
            'Feature': [feature_label_map[f] for f in FEATURES],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)

        # Create interactive plotly bar chart
        fig = px.bar(importances,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Random Forest Feature Importance',
                    template='simple_white')

        fig.update_layout(
            showlegend=False,
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            height=400,
            width=800,
            title_x=0.5
        )

        st.plotly_chart(fig)
                      
    except Exception as e:
        st.error(f"Model file not found or could not be loaded: {e}")
        model_ready = False

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
