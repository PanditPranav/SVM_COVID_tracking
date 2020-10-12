import json
from datetime import date
from urllib.request import urlopen

import matplotlib.dates as dates
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator, WeekdayLocator
from matplotlib.ticker import NullFormatter
from pandas.io.json import json_normalize

today = date.today()
#sns.set_style('whitegrid')
# style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1
dpi = 1000
plt.rcParams['font.size'] = 13
#plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
plt.rcParams['axes.titlesize'] = plt.rcParams['font.size']
plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
plt.rcParams['figure.figsize'] = 8, 8


st.beta_set_page_config(page_title="COVID19: EpiCenter for Disease Dynamics",
                   layout='wide',
                   initial_sidebar_state='auto')

st.sidebar.write('')
sidebar_selection = st.sidebar.radio('Select data:',['Select Counties','Tri-County Area','California'])
#@st.cache(suppress_st_warning=True)
def plot_county(county):
    import numpy as np
    FIPSs = confirmed.groupby(['Province_State', 'Admin2']).FIPS.unique().apply(pd.Series).reset_index()
    FIPSs.columns = ['State', 'County', 'FIPS']
    FIPSs['FIPS'].fillna(0, inplace = True)
    FIPSs['FIPS'] = FIPSs.FIPS.astype(int).astype(str).str.zfill(5)
    @st.cache(ttl=3*60*60, suppress_st_warning=True)
    def get_testing_data(County):
        if len(County) == 1:
            #print(len(County))
            f = FIPSs[FIPSs.County == County[0]].FIPS.values[0]
            #print(f)
            path1 = 'https://data.covidactnow.org/latest/us/counties/'+f+'.OBSERVED_INTERVENTION.timeseries.json'
            #print(path1)
            df = json.loads(requests.get(path1).text)
            #print(df.keys())
            data = pd.DataFrame.from_dict(df['actualsTimeseries'])
            data['Date'] = pd.to_datetime(data['date'])
            data = data.set_index('Date')
            #print(data.tail())
            try:
                data['new_negative_tests'] = data['cumulativeNegativeTests'].diff()
                data.loc[(data['new_negative_tests'] < 0)] = np.nan
            except:
                data['new_negative_tests'] = np.nan
                st.text('Negative test data not avilable')
            data['new_negative_tests_rolling'] = data['new_negative_tests'].fillna(0).rolling(14).mean()


            try:
                data['new_positive_tests'] = data['cumulativePositiveTests'].diff()
                data.loc[(data['new_positive_tests'] < 0)] = np.nan
            except:
                data['new_positive_tests'] = np.nan
                st.text('test data not avilable')
            data['new_positive_tests_rolling'] = data['new_positive_tests'].fillna(0).rolling(14).mean()
            data['new_tests'] = data['new_negative_tests']+data['new_positive_tests']
            data['new_tests_rolling'] = data['new_tests'].fillna(0).rolling(14).mean()
            data['testing_positivity_rolling'] = (data['new_positive_tests_rolling'] / data['new_tests_rolling'])*100
            #data['testing_positivity_rolling'].tail(14).plot()
            #plt.show()
            return data['new_tests_rolling'], data['testing_positivity_rolling'].iloc[-1:].values[0]
        elif (len(County) > 1) & (len(County) < 5):
            new_positive_tests = []
            new_negative_tests = []
            new_tests = []
            for c in County:
                f = FIPSs[FIPSs.County == c].FIPS.values[0]
                path1 = 'https://data.covidactnow.org/latest/us/counties/'+f+'.OBSERVED_INTERVENTION.timeseries.json'
                df = json.loads(requests.get(path1).text)
                data = pd.DataFrame.from_dict(df['actualsTimeseries'])
                data['Date'] = pd.to_datetime(data['date'])
                data = data.set_index('Date')
                try:
                    data['new_negative_tests'] = data['cumulativeNegativeTests'].diff()
                    data.loc[(data['new_negative_tests'] < 0)] = np.nan
                except:
                    data['new_negative_tests'] = np.nan
                    #print('Negative test data not avilable')

                try:
                    data['new_positive_tests'] = data['cumulativePositiveTests'].diff()
                    data.loc[(data['new_positive_tests'] < 0)] = np.nan
                except:
                    data['new_positive_tests'] = np.nan
                    #print('Negative test data not avilable')
                data['new_tests'] = data['new_negative_tests']+data['new_positive_tests']

                new_positive_tests.append(data['new_positive_tests'])
                #new_negative_tests.append(data['new_tests'])
                new_tests.append(data['new_tests'])

            new_positive_tests_rolling = pd.concat(new_positive_tests, axis = 1).sum(axis = 1)
            new_positive_tests_rolling = new_positive_tests_rolling.fillna(0).rolling(14).mean()

            new_tests_rolling = pd.concat(new_tests, axis = 1).sum(axis = 1)
            new_tests_rolling = new_tests_rolling.fillna(0).rolling(14).mean()

            data_to_show = (new_positive_tests_rolling / new_tests_rolling)*100
            return new_tests_rolling, data_to_show.iloc[-1:].values[0]
        else:
            st.text('Getting testing data for California State')
            path1 = 'https://data.covidactnow.org/latest/us/states/CA.OBSERVED_INTERVENTION.timeseries.json'
            df = json.loads(requests.get(path1).text)
            data = pd.DataFrame.from_dict(df['actualsTimeseries'])
            data['Date'] = pd.to_datetime(data['date'])
            data = data.set_index('Date')

            try:
                data['new_negative_tests'] = data['cumulativeNegativeTests'].diff()
                data.loc[(data['new_negative_tests'] < 0)] = np.nan
            except:
                data['new_negative_tests'] = np.nan
                print('Negative test data not avilable')
            data['new_negative_tests_rolling'] = data['new_negative_tests'].fillna(0).rolling(14).mean()


            try:
                data['new_positive_tests'] = data['cumulativePositiveTests'].diff()
                data.loc[(data['new_positive_tests'] < 0)] = np.nan
            except:
                data['new_positive_tests'] = np.nan
                st.text('test data not avilable')
            data['new_positive_tests_rolling'] = data['new_positive_tests'].fillna(0).rolling(14).mean()
            data['new_tests'] = data['new_negative_tests']+data['new_positive_tests']
            data['new_tests_rolling'] = data['new_tests'].fillna(0).rolling(14).mean()
            data['testing_positivity_rolling'] = (data['new_positive_tests_rolling'] / data['new_tests_rolling'])*100
            return data['new_tests_rolling'], data['testing_positivity_rolling'].iloc[-1:].values[0]


    testing_df, testing_percent = get_testing_data(County=county)
    county_confirmed = confirmed[confirmed.Admin2.isin(county)]
    #county_confirmed = confirmed[confirmed.Admin2 == county]
    county_confirmed_time = county_confirmed.drop(county_confirmed.iloc[:, 0:12], axis=1).T #inplace=True, axis=1
    county_confirmed_time = county_confirmed_time.sum(axis= 1)
    county_confirmed_time = county_confirmed_time.reset_index()
    county_confirmed_time.columns = ['date', 'cases']
    county_confirmed_time['Datetime'] = pd.to_datetime(county_confirmed_time['date'])
    county_confirmed_time = county_confirmed_time.set_index('Datetime')
    del county_confirmed_time['date']
    #print(county_confirmed_time.head())
    incidence= pd.DataFrame(county_confirmed_time.cases.diff())
    incidence.columns = ['incidence']

    #temp_df_time = temp_df.drop(['date'], axis=0).T #inplace=True, axis=1
    county_deaths = deaths[deaths.Admin2.isin(county)]
    population = county_deaths.Population.values.sum()

    del county_deaths['Population']
    county_deaths_time = county_deaths.drop(county_deaths.iloc[:, 0:11], axis=1).T #inplace=True, axis=1
    county_deaths_time = county_deaths_time.sum(axis= 1)

    county_deaths_time = county_deaths_time.reset_index()
    county_deaths_time.columns = ['date', 'deaths']
    county_deaths_time['Datetime'] = pd.to_datetime(county_deaths_time['date'])
    county_deaths_time = county_deaths_time.set_index('Datetime')
    del county_deaths_time['date']

    cases_per100k  = ((county_confirmed_time)*100000/population)
    cases_per100k.columns = ['cases per 100K']
    cases_per100k['rolling average'] = cases_per100k['cases per 100K'].rolling(7).mean()

    deaths_per100k  = ((county_deaths_time)*100000/population)
    deaths_per100k.columns = ['deaths per 100K']
    deaths_per100k['rolling average'] = deaths_per100k['deaths per 100K'].rolling(7).mean()


    incidence['rolling_incidence'] = incidence.incidence.rolling(7).mean()
    metric = (incidence['rolling_incidence']*100000/population).iloc[[-1]]

    #print(county_deaths_time.tail(1).values[0])
    #print(cases_per100k.head())
    fig, ((ax4, ax3),(ax1, ax2)) = plt.subplots(2,2, figsize=(12,8))
    #fig, ((ax4, ax3),(ax1, ax2)) = plt.subplots(2,2, figsize=(6,4))

    county_confirmed_time.plot(ax = ax1,  lw=4, color = '#377eb8')
    county_deaths_time.plot(ax = ax1,  lw=4, color = '#e41a1c')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Number of individuals')



    testing_df.plot(ax = ax2,  lw=4, color = '#377eb8')
    #cases_per100k['cases per 100K'].plot(ax = ax2,  lw=4, linestyle='--', color = '#377eb8')
    #cases_per100k['rolling average'].plot(ax = ax2, lw=4, color = '#377eb8')

    #deaths_per100k['deaths per 100K'].plot(ax = ax2,  lw=4, linestyle='--', color = '#e41a1c')
    #deaths_per100k['rolling average'].plot(ax = ax2, lw=4, color = '#e41a1c')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Number of new tests')

    incidence.incidence.plot(kind ='bar', ax = ax3, width=1)
    ax3.set_xticklabels(incidence.index.strftime('%b %d'))
    for index, label in enumerate(ax3.xaxis.get_ticklabels()):
        if index % 7 != 0:
            label.set_visible(False)
    for index, label in enumerate(ax3.xaxis.get_major_ticks()):
        if index % 7 != 0:
            label.set_visible(False)




    (incidence['rolling_incidence']*100000/population).plot(ax = ax4, lw = 4)
    ax4.axhline(y = 5,  linewidth=2, color='r', ls = '--', label="Threshold for Phase 2:\nInitial re-opening")
    ax4.axhline(y = 1,  linewidth=2, color='b', ls = '--', label="Threshold for Phase 3:\nEconomic recovery")
    ax4.legend(fontsize = 10)
    if (incidence['rolling_incidence']*100000/population).max()< 5.5:
        ax4.set_ylim(0,5.5)

    #print(metric)

    #incidence['rolling_incidence']
    #ax3.grid(which='both', alpha=1)
    ax1.set_title('(C) Cumulative cases and deaths')
    ax2.set_title('(D) Daily new tests')
    ax3.set_title('(B) Daily incidence (new cases)')
    ax4.set_title('(A) Weekly rolling mean of incidence per 100k')
    ax3.set_ylabel('Number of individuals')
    ax4.set_ylabel('per 100 thousand')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.figure(linewidth=1, edgecolor="#cccccc")


    if len(county) == 1:
        st.subheader('Current situation of COVID-19 cases in '+', '.join(map(str, county))+' county ('+ str(today)+')')

    else:
        st.subheader('Current situation of COVID-19 cases in '+', '.join(map(str, county))+' counties ('+ str(today)+')')

    # a1, a2, a3 = st.beta_columns(3)
    #
    # with a1:
    #     st.text('# new cases averaged over last 7 days = %s' %'{:,.1f}'.format(metric.values[0]))
    #     st.text("Population under consideration = %s"% '{:,.0f}'.format(population))
    #
    # with a2:
    #     st.text("Total cases = %s"% '{:,.0f}'.format(county_confirmed_time.tail(1).values[0][0]))
    #     st.text("Total deaths = %s"% '{:,.0f}'.format(county_deaths_time.tail(1).values[0][0]))
    #
    # with a3:
    #     st.text("% test positivity (14 day average)= "+"%.2f" % testing_percent)


    c1, c_space, c2 = st.beta_columns((8,.25,4))
    c3 = st.beta_container()

    with c1:
        st.write('')
        st.pyplot(fig)

    # with c2:
    #     sub1, sub2 = st.beta_columns((.6,50))
    #     with sub2

    import streamlit.components.v1 as components
    if len(county)==1:
        for C in county:
            with c2:
                # st.text(C)
                st.write('')
                st.write('')
                f = FIPSs[FIPSs.County == C].FIPS.values[0]
                components.iframe("https://covidactnow.org/embed/us/county/"+f, width=350, height=365, scrolling=False)
    elif len(county)<=3:
        for C in county:
            with c3:
                # st.text(C)
                st.write('')
                st.write('')
                f = FIPSs[FIPSs.County == C].FIPS.values[0]
                components.iframe("https://covidactnow.org/embed/us/county/"+f, width=350, height=365, scrolling=False)

    with c2:
        # st.text('# new cases averaged over last 7 days = %s' %'{:,.1f}'.format(metric.values[0]))
        st.write('')
        st.write('')
        st.markdown("Population under consideration = %s"% '{:,.0f}'.format(population))
        st.markdown("Total cases = %s"% '{:,.0f}'.format(county_confirmed_time.tail(1).values[0][0]))
        st.markdown("Total deaths = %s"% '{:,.0f}'.format(county_deaths_time.tail(1).values[0][0]))
        st.markdown("% test positivity (14 day average)= "+"%.2f" % testing_percent)

def plot_state():
    import numpy as np

    #FIPSs = confirmed.groupby(['Province_State', 'Admin2']).FIPS.unique().apply(pd.Series).reset_index()
    #FIPSs.columns = ['State', 'County', 'FIPS']
    #FIPSs['FIPS'].fillna(0, inplace = True)
    #FIPSs['FIPS'] = FIPSs.FIPS.astype(int).astype(str).str.zfill(5)
    @st.cache(ttl=3*60*60, suppress_st_warning=True)
    def get_testing_data_state():
            st.text('Getting testing data for California State')
            path1 = 'https://data.covidactnow.org/latest/us/states/CA.OBSERVED_INTERVENTION.timeseries.json'
            df = json.loads(requests.get(path1).text)
            data = pd.DataFrame.from_dict(df['actualsTimeseries'])
            data['Date'] = pd.to_datetime(data['date'])
            data = data.set_index('Date')

            try:
                data['new_negative_tests'] = data['cumulativeNegativeTests'].diff()
                data.loc[(data['new_negative_tests'] < 0)] = np.nan
            except:
                data['new_negative_tests'] = np.nan
                print('Negative test data not avilable')
            data['new_negative_tests_rolling'] = data['new_negative_tests'].fillna(0).rolling(14).mean()


            try:
                data['new_positive_tests'] = data['cumulativePositiveTests'].diff()
                data.loc[(data['new_positive_tests'] < 0)] = np.nan
            except:
                data['new_positive_tests'] = np.nan
                st.text('test data not avilable')
            data['new_positive_tests_rolling'] = data['new_positive_tests'].fillna(0).rolling(14).mean()
            data['new_tests'] = data['new_negative_tests']+data['new_positive_tests']
            data['new_tests_rolling'] = data['new_tests'].fillna(0).rolling(14).mean()
            data['testing_positivity_rolling'] = (data['new_positive_tests_rolling'] / data['new_tests_rolling'])*100
            return data['new_tests_rolling'], data['testing_positivity_rolling'].iloc[-1:].values[0]


    testing_df, testing_percent = get_testing_data_state()
    county_confirmed = confirmed[confirmed.Province_State == 'California']
    #county_confirmed = confirmed[confirmed.Admin2 == county]
    county_confirmed_time = county_confirmed.drop(county_confirmed.iloc[:, 0:12], axis=1).T #inplace=True, axis=1
    county_confirmed_time = county_confirmed_time.sum(axis= 1)
    county_confirmed_time = county_confirmed_time.reset_index()
    county_confirmed_time.columns = ['date', 'cases']
    county_confirmed_time['Datetime'] = pd.to_datetime(county_confirmed_time['date'])
    county_confirmed_time = county_confirmed_time.set_index('Datetime')
    del county_confirmed_time['date']
    #print(county_confirmed_time.head())
    incidence= pd.DataFrame(county_confirmed_time.cases.diff())
    incidence.columns = ['incidence']

    #temp_df_time = temp_df.drop(['date'], axis=0).T #inplace=True, axis=1
    county_deaths = deaths[deaths.Province_State == 'California']
    population = county_deaths.Population.values.sum()

    del county_deaths['Population']
    county_deaths_time = county_deaths.drop(county_deaths.iloc[:, 0:11], axis=1).T #inplace=True, axis=1
    county_deaths_time = county_deaths_time.sum(axis= 1)

    county_deaths_time = county_deaths_time.reset_index()
    county_deaths_time.columns = ['date', 'deaths']
    county_deaths_time['Datetime'] = pd.to_datetime(county_deaths_time['date'])
    county_deaths_time = county_deaths_time.set_index('Datetime')
    del county_deaths_time['date']

    cases_per100k  = ((county_confirmed_time)*100000/population)
    cases_per100k.columns = ['cases per 100K']
    cases_per100k['rolling average'] = cases_per100k['cases per 100K'].rolling(7).mean()

    deaths_per100k  = ((county_deaths_time)*100000/population)
    deaths_per100k.columns = ['deaths per 100K']
    deaths_per100k['rolling average'] = deaths_per100k['deaths per 100K'].rolling(7).mean()


    incidence['rolling_incidence'] = incidence.incidence.rolling(7).mean()
    metric = (incidence['rolling_incidence']*100000/population).iloc[[-1]]

    #print(county_deaths_time.tail(1).values[0])
    #print(cases_per100k.head())
    fig, ((ax4, ax3),(ax1, ax2)) = plt.subplots(2,2, figsize=(12,8))
    #fig, ((ax4, ax3),(ax1, ax2)) = plt.subplots(2,2, figsize=(6,4))

    county_confirmed_time.plot(ax = ax1,  lw=4, color = '#377eb8')
    county_deaths_time.plot(ax = ax1,  lw=4, color = '#e41a1c')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Number of individuals')



    testing_df.plot(ax = ax2,  lw=4, color = '#377eb8')
    #cases_per100k['cases per 100K'].plot(ax = ax2,  lw=4, linestyle='--', color = '#377eb8')
    #cases_per100k['rolling average'].plot(ax = ax2, lw=4, color = '#377eb8')

    #deaths_per100k['deaths per 100K'].plot(ax = ax2,  lw=4, linestyle='--', color = '#e41a1c')
    #deaths_per100k['rolling average'].plot(ax = ax2, lw=4, color = '#e41a1c')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Number of new tests')

    incidence.incidence.plot(kind ='bar', ax = ax3, width=1)
    ax3.set_xticklabels(incidence.index.strftime('%b %d'))
    for index, label in enumerate(ax3.xaxis.get_ticklabels()):
        if index % 7 != 0:
            label.set_visible(False)
    for index, label in enumerate(ax3.xaxis.get_major_ticks()):
        if index % 7 != 0:
            label.set_visible(False)




    (incidence['rolling_incidence']*100000/population).plot(ax = ax4, lw = 4)
    ax4.axhline(y = 5,  linewidth=2, color='r', ls = '--', label="Threshold for Phase 2:\nInitial re-opening")
    ax4.axhline(y = 1,  linewidth=2, color='b', ls = '--', label="Threshold for Phase 3:\nEconomic recovery")
    ax4.legend(fontsize = 10)
    if (incidence['rolling_incidence']*100000/population).max()< 5.5:
        ax4.set_ylim(0,5.5)

    #print(metric)

    #incidence['rolling_incidence']
    #ax3.grid(which='both', alpha=1)
    ax1.set_title('(C) Cumulative cases and deaths')
    ax2.set_title('(D) Daily new tests')
    ax3.set_title('(B) Daily incidence (new cases)')
    ax4.set_title('(A) Weekly rolling mean of incidence per 100k')
    ax3.set_ylabel('Number of individuals')
    ax4.set_ylabel('per 100 thousand')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.figure(linewidth=1, edgecolor="#cccccc")

    st.subheader('Current situation of COVID-19 cases in California ('+ str(today)+')')

    c1, c2 = st.beta_columns((11,4))

    with c1:
        st.write('')
        st.pyplot(fig)

    with c2:
        # st.text('# new cases averaged over last 7 days = %s' %'{:,.1f}'.format(metric.values[0]))
        st.write('')
        st.write('')
        st.markdown("Population under consideration = %s"% '{:,.0f}'.format(population))
        st.markdown("Total cases = %s"% '{:,.0f}'.format(county_confirmed_time.tail(1).values[0][0]))
        st.markdown("Total deaths = %s"% '{:,.0f}'.format(county_deaths_time.tail(1).values[0][0]))
        st.markdown("% test positivity (14 day average)= "+"%.2f" % testing_percent)

@st.cache(ttl=3*60*60, suppress_st_warning=True)
def get_data():
    US_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
    US_deaths = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
    confirmed = pd.read_csv(US_confirmed)
    deaths = pd.read_csv(US_deaths)
    return confirmed, deaths

confirmed, deaths = get_data()

t1, t2 = st.beta_columns(2)
with t1:
    st.markdown('# COVID-19 Data and Reporting')

with t2:
    st.header("")
    st.write("""
    **EpiCenter for Disease Dynamics** | School of Veterinary Medicine - UC Davis
    """)

st.write("")
st.markdown("""
COVID-Local provides basic key metrics against which to assess pandemic response and progress toward reopening.
See more at https://www.covidlocal.org/metrics/
For additional information please contact *epicenter@ucdavis.edu* or visit https://ohi.vetmed.ucdavis.edu/centers/epicenter-disease-dynamics""")
# explanation = st.beta_expander("Show more details")
# with explanation:
#     st.markdown("One of the key metrics for which data are widely available is the estimate of **daily new cases per 100,000 population**. Here, in following graphics, we will track")
#
#     st.markdown("(A) Estimates of daily new cases per 100,000 population (averaged over the last seven days)")
#     st.markdown("(B) Daily incidence (new cases)")
#     st.markdown("(C) Cumulative cases and deaths")
#     st.markdown("(D) Daily new tests")
#
#     st.markdown("Data source: Data for cases are procured automatically from **COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University**.")
#     st.markdown("The data is updated at least once a day or sometimes twice a day in the COVID-19 Data Repository.  https://github.com/CSSEGISandData/COVID-19")
#     st.markdown("Infection rate, positive test rate, ICU headtoom and contacts traceed from https://covidactnow.org/")
#     st.text('Report updated on '+ str(today))


if sidebar_selection == 'Select Counties':
    # st.markdown('## Select counties of interest')
    CA_counties = confirmed[confirmed.Province_State == 'California'].Admin2.unique().tolist()

    COUNTIES_SELECTED = st.multiselect('Select counties', CA_counties, default=['Yolo'])

    # st.markdown(COUNTIES_SELECTED)
    plot_county(COUNTIES_SELECTED)

if sidebar_selection == 'Tri-County Area':
    plot_county(['Yolo', 'Solano', 'Sacramento'])
    plot_county(['Yolo'])
    plot_county(['Sacramento'])
    plot_county(['Solano'])

if sidebar_selection == 'California':
    plot_state()