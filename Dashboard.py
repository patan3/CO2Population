import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title='CO2-Population Dashboard', layout='wide', initial_sidebar_state='auto') # page_icon=':flag-eu:', 

st.sidebar.markdown('# About\n The aim of this web application is to analyze the relation between the CO2 emission and the population growth of a given country. The interest is on whether there is a link between them and how it changes over time and among different parts of the world.')
st.sidebar.markdown("## Data Sources\n") 
st.sidebar.markdown("- [CO2 Emission From Fuel Combustion 2020 Edition](https://iea.blob.core.windows.net/assets/474cf91a-636b-4fde-b416-56064e0c7042/WorldCO2_Documentation.pdf) ")
st.sidebar.markdown("- [The World Bank](https://data.worldbank.org/indicator/SP.POP.GROW) ")

st.sidebar.markdown('# Sections')
st.sidebar.markdown('0. Introduction')
st.sidebar.markdown('1. Exploratory Data Analysis')
st.sidebar.markdown('2. Analysis')

st.sidebar.markdown(" # About the Author ")
st.sidebar.markdown("Angelo Patane'  \n angelpatane9@gmail.com")
st.sidebar.markdown("[Linkedin](https://www.linkedin.com/in/angelopatane/)    \n [Twitter](https://twitter.com/angel_patane3)")

@st.cache
def load_data():
    return pd.read_csv(r'https://raw.githubusercontent.com/patan3/CO2Population/master/df_final.csv', index_col = 0), pd.read_csv(r'https://raw.githubusercontent.com/patan3/CO2Population/master/df_merge.csv', index_col = 0)

df_final, df_merge = load_data()
# %%
st.header('0. Introduction')
st.markdown("The variables 'CO2 Emissions' and 'Population' are indices. The reference value is 100. The reference year is 1990 for all the countries.")
if st.checkbox('Display Cleaned Data'):
    st.dataframe(df_final)
    st.dataframe(df_merge)


# %%
st.header('1. Exploratory Data Analysis')
# Plot 1
# Country Slider Plot 1 
country1 = st.selectbox('Select the region', df_final['Country'].unique())
fig1 = px.line(df_final.loc[df_final['Country'] == country1], 
              x = "Year", y=["CO2 emissions", "Population"],
              template = 'plotly_white')
st.subheader('1.1 CO2 Emissions and Population Indeces Change of ' + str(country1) + ' (1971-2018)')
st.plotly_chart(fig1, use_container_width=True)

# %%

# Plot 2
# African Countries Checkbox Plot 2

african_countries = st.radio('Display African Countries', ('Yes', 'No'))
africa = df_final.Country.unique()[list(df_final.Country.unique()).index('Non-OECD Europe and Eurasia')+1:list(df_final.Country.unique()).index('Other Africa')+1]

if np.where(african_countries == 'Yes', True, False):
    fig2 = px.choropleth(df_final, locations = 'Country', locationmode="country names", color="CO2 emissions",
                     hover_name="Country", color_continuous_scale = 'Reds',
                     animation_frame="Year",
                     projection="natural earth")
    st.subheader("1.2 CO2 Emissions Index of the World (with African countries) for a given year")
else:
    fig2 = px.choropleth(df_final.loc[~df_final['Country'].isin(africa)], locations = 'Country', locationmode="country names", color="CO2 emissions",
                     hover_name="Country", color_continuous_scale = 'Reds',
                     animation_frame="Year",
                     projection="natural earth")
    st.subheader("1.2 CO2 Emissions Index of the World (without African countries) for a given year")
st.plotly_chart(fig2, use_container_width=True)

# %% 

# Plot 3 - CO2 Emissions vs Population Growth
drop_index = df_merge.loc[(df_merge['Country'] == 'Equatorial Guinea') | (df_merge['Country'] == 'Benin') | (df_merge['Country'] == 'Gibraltar')].index
df_merge_1 = df_merge.drop(drop_index).dropna().reset_index(drop = True)

fig3 = px.scatter(df_merge_1, 
                 x = "Population growth (annual %)", y="CO2 emissions",
                 template = 'plotly_white', hover_data = ['Country', 'Continent'], color = 'Continent',
                animation_frame="Year", range_x = [-15, 20], range_y = [-100, 1300])
st.subheader('1.3 CO2 Emissions Index vs Population Growth of the World (without Equatorial Guinea and Benin)')
st.plotly_chart(fig3, use_container_width=True)


# %%

# Plot 4 - Animated Bar Charts
col1, col2 = st.beta_columns(2)

df_merge_continents = df_merge.dropna().groupby(by = ['Continent', 'Year']).mean().reset_index()

fig41 = px.bar(df_merge_continents, y = 'Continent', x = 'CO2 emissions', color = 'Continent',
  animation_frame = 'Year', animation_group = 'Continent', template = 'plotly_white', orientation = 'h')

fig42 = px.bar(df_merge_continents, y = 'Continent', x = 'Population growth (annual %)', color = 'Continent',
  animation_frame = 'Year', animation_group = 'Continent', template = 'plotly_white', orientation = 'h')

col1.subheader('1.4.1 CO2 Emission Index Bar Chart of the World (1971-2018)')
col1.plotly_chart(fig41, use_container_width=True)
col2.subheader('1.4.2 Population Growth Bar Chart of the World (1971-2018)')
col2.plotly_chart(fig42, use_container_width=True)

# %%
st.header('2. Analysis')

# Pearson Correlation Coefficient 
year = st.slider('Select the year', min_value = 1971, max_value = 2018)

df_merge_year = df_merge.loc[df_merge['Year'] == year]
drop_index = df_merge_year.loc[(df_merge_year['Country'] == 'Equatorial Guinea') | (df_merge_year['Country'] == 'Benin') | (df_merge_year['Country'] == 'Nepal') | (df_merge_year['Country'] == 'Bahrain')].index
df_merge_year_1 = df_merge_year.drop(drop_index).dropna().reset_index(drop = True)

fig5 = px.scatter(df_merge_year_1, 
                 x = "Population growth (annual %)", y="CO2 emissions",
                 template = 'plotly_white', hover_data = ['Country', 'Continent'], color = 'Continent')
coeff, pvalue = stats.pearsonr(df_merge_year_1['CO2 emissions'], df_merge_year_1['Population growth (annual %)'])

st.subheader('2.1 CO2 Emissions Index vs Population Growth (annual %) - ' + str(year))
st.markdown('#### Pearson Correlation Coefficient $r$ =  ' + str(coeff))
st.markdown('#### $p$-value =  ' + str(pvalue) + ' ($H_0$: true correlation coefficient = 0)')
st.plotly_chart(fig5, use_container_width=True)

# %%

# Cluster Analysis
scaler = StandardScaler(with_mean=False)
scaled_columns = scaler.fit_transform(df_merge_year_1[['CO2 emissions', 'Population growth (annual %)']])
scaled_df = pd.DataFrame(scaled_columns, columns = ['CO2 emissions', 'Population growth (annual %)'])
scaled_df['Country'] = df_merge_year_1['Country']
scaled_df['Continent'] = df_merge_year_1['Continent']

clusters = st.radio('How many clusters?', (2, 3, 4, 5, 6))

# Initiate kMeans object
kmeans = KMeans(
     init = "random",
     n_clusters = clusters,
     n_init = 10,
     random_state = 42
)

# Fit kMeans object
kmeans.fit(scaled_df[['CO2 emissions', 'Population growth (annual %)']])

# Put kMeans predicted labels
scaled_df['kM Labels'] = kmeans.labels_.astype(object)

# Plot Clusters
fig6 = px.scatter(scaled_df, 
                 x = 'Population growth (annual %)', y='CO2 emissions',
                 template = 'plotly_white', hover_data = ['Country'], color = 'kM Labels')
st.subheader('2.2 CO2 Emissions Index vs Population Growth (annual %) with ' + str(clusters) + ' Clusters - ' + str(year))
st.plotly_chart(fig6, use_container_width=True)

