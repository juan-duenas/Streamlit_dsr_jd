import plotly.express as px
import pandas as pd

def scatter_gdp_lifeexp(df: pd.DataFrame):
    """
    Create an interactive scatter plot (plotly) relating GDP per capita (log x) and Life Expectancy (IHME).
    Point size: headcount_ratio_upper_mid_income_povline
    Color: country (if available)
    Hover: all columns
    """
    gdp_col = 'GDP per capita'
    lifeexp_col = 'Life Expectancy (IHME)'
    size_col = 'headcount_ratio_upper_mid_income_povline'
    # Try to find a country column
    country_col = None
    for c in ['country', 'Country', 'COUNTRY', 'country_name']:
        if c in df.columns:
            country_col = c
            break
    
    hover_cols = []
    if country_col:
        hover_cols.append(country_col)
    hover_cols.append(gdp_col)
    fig = px.scatter(
        df,
        x=gdp_col,
        y=lifeexp_col,
        size=size_col if size_col in df.columns else None,
        color=country_col if country_col else None,
        hover_data=hover_cols,
        log_x=True,
        title="Life Expectancy vs GDP per Capita",
        labels={
            gdp_col: "GDP per capita (log scale)",
            lifeexp_col: "Life Expectancy (IHME)",
            size_col: "Headcount Ratio (upper mid income poverty line)",
            country_col: "Country" if country_col else None
        }
    )
    fig.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey')))
    fig.update_layout(
        height=500,
        width=800,
        legend_title_text=country_col if country_col else None,
        title_x=0.5
    )
    return fig


def map_countries(df: pd.DataFrame):
    """
    Create a world map with one point per country in the dataframe.
    Uses plotly.express.scatter_geo with locationmode='country names'.
    Aggregates values per country (mean) to avoid duplicate markers.
    """
    # detect country column
    country_col = None
    for c in ['country', 'Country', 'COUNTRY', 'country_name']:
        if c in df.columns:
            country_col = c
            break
    if country_col is None:
        raise ValueError('No country column found in dataframe')

    # choose some value columns to show in hover if present
    gdp_col = 'GDP per capita' if 'GDP per capita' in df.columns else None
    lifeexp_col = 'Life Expectancy (IHME)' if 'Life Expectancy (IHME)' in df.columns else None
    size_col = gdp_col

    # aggregate by country
    agg_dict = {}
    if gdp_col:
        agg_dict[gdp_col] = 'mean'
    if lifeexp_col:
        agg_dict[lifeexp_col] = 'mean'
    if agg_dict:
        agg = df.groupby(country_col).agg(agg_dict).reset_index()
    else:
        agg = df[[country_col]].drop_duplicates().reset_index(drop=True)

    hover_cols = [country_col]
    if gdp_col:
        hover_cols.append(gdp_col)
    if lifeexp_col:
        hover_cols.append(lifeexp_col)

    import plotly.express as px

    fig = px.scatter_geo(
        agg,
        locations=country_col,
        locationmode='country names',
        hover_name=country_col,
        hover_data=hover_cols,
        size=size_col if size_col in agg.columns else None,
        projection='natural earth',
        title='Countries in dataset (aggregated)'
    )
    fig.update_layout(height=600, title_x=0.5)
    return fig
