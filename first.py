import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import statsmodels.api as sm
import plotly.express as px
import pickle

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
import streamlit as st

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from PIL import Image
from datetime import datetime
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import r2_score, mean_squared_error

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_option('deprecation.showPyplotGlobalUse', False)
# st.set_page_config(layout='wide')

st.title("Greenhouse Gas Emission Predictor")

with st.beta_expander("About"):
    st.write("""
    * Visualizes emissions of greenhouse gases including carbon dioxide (CO$_2$),
     methane (CH$_4$), and nitrous oxide (N$_2$O) for all countries
    * Forecasts CO$_2$ emission of 16 countries leading the global economic output for 2020-2025.
    """)

    image=Image.open ("Global_warming.jpg")
    st.image(image)


# df=pd.read_csv(r'C:\Users\Administrator\Documents\GA_Work\Capstone\Links\1\co2-data-master\co2-data-master\owid-co2-data.csv',parse_dates=['year'])
df=pd.read_csv('owid-co2-data.csv',parse_dates=['year'])


continents=['Africa','Asia (excl. China & India)','EU-27','EU-28',
'Europe','Europe (excl. EU-27)','Europe (excl. EU-28)',
'International transport','North America','North America (excl. USA)',
'Oceania','South America','World']
array_to_list=df['country'].unique().tolist()
only_countries=[item for item in array_to_list if item not in continents]

check_1 = st.sidebar.checkbox('GHG emission for all countries')
if check_1:
     country_name=st.selectbox("Country",only_countries,key = "First")
     def plot_series(df,country_name,steps=10):
             A=df.loc[df['country']==country_name,['year','co2','population',
             'gdp','methane','nitrous_oxide']]
             A.set_index('year',inplace=True)
             fig,axes=plt.subplots(nrows=1,ncols=3,figsize=(10*3,10))
             axes[0].plot(A['co2'],linewidth=3,color='r',linestyle='dashed',
             marker='o',markersize=10)
             axes[0].set_title("CO${_2}$ emission",fontsize=50)
             axes[0].set_xlabel('Year', fontsize=50)
             axes[0].set_ylabel('CO$_2$, million tonnes',fontsize=50)
             axes[0].set_xticklabels(A.index[0::steps],rotation=75)
             for label in (axes[0].get_xticklabels() + axes[0].get_yticklabels()):
                 label.set_fontsize(40)
             axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
             for direction in ['top','right','left','bottom']:
                axes[0].spines[direction].set_linewidth(6)

             axes[1].plot(A['methane'],linewidth=3,color='blue',
             linestyle='dashed',marker='o',markersize=10)
             axes[1].set_title("CH${_4}$ emission",fontsize=50)
             axes[1].set_xlabel('Year', fontsize=50)
             axes[1].set_xticklabels(A.index[0::steps],rotation=75)
             for label in (axes[1].get_xticklabels() + axes[1].get_yticklabels()):
                label.set_fontsize(40)
             axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
             for direction in ['top','right','left','bottom']:
                axes[1].spines[direction].set_linewidth(6)

             axes[2].plot(A['nitrous_oxide'],linewidth=3,color='green',linestyle='dashed',marker='o',markersize=10)
             axes[2].set_title("N$_2$O emission",fontsize=50)
             axes[2].set_xlabel('Year', fontsize=50)
             axes[2].set_xticklabels(A.index[0::steps],rotation=75)
             for label in (axes[2].get_xticklabels() + axes[2].get_yticklabels()):
                label.set_fontsize(40)
             axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
             for direction in ['top','right','left','bottom']:
                axes[2].spines[direction].set_linewidth(6)
             fig.tight_layout(pad=3);
     st.pyplot(plot_series(df,country_name))


check_2 = st.sidebar.checkbox('Carbon dioxide emission for continents')
if check_2:
    continents_pure=['Africa','Europe','Asia','North America','South America',
    'World']
    continent_name=st.selectbox("Continent",continents_pure)
    A_continent=df.loc[df['country']==continent_name,['year','co2',
    'population','gdp']]
    A_continent.set_index('year',inplace=True)

    fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(5,5))
    axes.plot(A_continent['co2'],linewidth=1,color='r',
    linestyle='dashed',marker='o',markersize=3,alpha=0.5)
    axes.set_title("CO${_2}$ emission",fontsize=15)
    axes.set_xlabel('Year', fontsize=15)
    axes.set_ylabel('CO$_2$, million tonnes',fontsize=15)
    axes.set_xticklabels(A_continent.index[0::10],rotation=75)
    for label in (axes.get_xticklabels() + axes.get_yticklabels()):
        label.set_fontsize(10)
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    for direction in ['top','right','left','bottom']:
        axes.spines[direction].set_linewidth(1)
    st.pyplot(fig)


check_3 = st.sidebar.checkbox('Carbon dioxide emission for the world in 1900-2019')
if check_3:
    array_to_list=df['country'].unique().tolist()
    only_countries=[item for item in array_to_list if item not in continents]
    df_only_countries=df[df["country"].isin (only_countries)]
    continents_plot=['africa','europe','asia','north america',
    'south america','world']
    continent_name_plot=st.selectbox("Continent",continents_plot)
    selected_year=st.slider('Year',min_value=1900,max_value=2019,
value=2000,step=1,
help='CO$_2$ emission will be rendered based on year and location of countries')
    selected_year_dtime=datetime.strptime(str(selected_year)+"-01-01","%Y-%m-%d")

    countries_year=df_only_countries[df_only_countries['year']==selected_year_dtime]
    countries_year['log10_co2']=np.log(countries_year['co2'])

    fig=px.choropleth(countries_year, locations="iso_code",
                      color='log10_co2',
                      range_color=(0,10),
                      labels={'log10_co2':"log10(CO_2)"},
                      hover_name="country",
                      hover_data={'co2':True, 'co2_per_capita':True, 'cement_co2':True,
                                   'coal_co2':True, 'flaring_co2':True, 'gas_co2':True, 'oil_co2':True,
                                   'other_industry_co2':True,
                                   'population':True,},
                     scope=continent_name_plot,
                    color_continuous_scale=px.colors.sequential.Reds)
    fig.update_layout(title={"text":' Carbon dioxide emission for countries in '+continent_name_plot.upper()+" - "+str(selected_year),'xanchor':"center",
                             'x':0.5},font_size=18,margin={"r":0,"t":70,"l":0,"b":10},height=800,width=800,
                     )
    st.plotly_chart(fig)
    df_info=countries_year[['country','year','co2','co2_per_capita',
    'cement_co2','coal_co2', 'flaring_co2', 'gas_co2', 'oil_co2',
    'other_industry_co2','population','gdp']]
    df_info.set_index('year',inplace=True)
    st.dataframe(data=df_info,width=1000,height=500)


check_4 = st.sidebar.checkbox('Prediction of carbon dioxide emission for countries 2020-2025')
if check_4:
    g20=['Argentina', 'Australia', 'Brazil', 'Canada', 'China', 'Germany',
    'France', 'India', 'Indonesia', 'Italy', 'Japan', 'Mexico', 'Saudi Arabia',
    'Turkey',
    'United Kingdom', 'United States']

    country_name=st.selectbox("Country",g20,key = "Second",
    help='16  countries representing 85% of global economic output')

    # loaded_model=pickle.load(open('finalized_model.pkl','rb'))

    A=df.loc[df['country']==country_name,['year','co2','population','gdp','methane','nitrous_oxide']]
    A.set_index('year',inplace=True)
    A['population_lag_1']=A['population'].shift(1)
    A['co2_lag_1']=A['co2'].shift(1)
    A['time'] =range(0,A.shape[0])
    X_train, X_test, y_train, y_test = train_test_split(A.drop(columns=['co2',
    'population','gdp','methane','nitrous_oxide']), A['co2'],test_size = 0.2,
    shuffle=False)
    X_train.dropna(inplace=True)
    y_train=y_train[X_train.index]

    X_test.dropna(inplace=True)
    y_test=y_test[X_test.index]
    # st.write(X_train.shape)
    # st.write(X_test.shape)
    #
    # st.write(y_train.shape)
    # st.write(y_test.shape)

    X_train=sm.add_constant(X_train)
    X_test=sm.add_constant(X_test)
    lm =sm.OLS(y_train,X_train)
    lm_results =lm.fit()
    preds=lm_results.predict(X_test).dropna()


    df=pd.read_csv('owid-co2-data.csv',parse_dates=['year'])

    df_plus=df_plus[['population','year','country']]
    df_plus=df_plus.loc[df_plus['country']==country_name,['population','year','country']]
    df_plus.dropna(inplace=True)
    X=df_plus[['year']]
    y=df_plus['population']
    X_plus_train,X_plus_test,y_plus_train,y_plus_test=train_test_split(X,y,test_size=0.3)
    ss=StandardScaler()
    X_plus_train_sc=ss.fit_transform(X_plus_train)
    X_plus_test_sc=ss.transform(X_plus_test)
    lr=LinearRegression()
    cross_val_score(lr,X_plus_train_sc,y_plus_train,cv=5).mean()
    lr.fit(X_plus_train_sc,y_plus_train)
    preds_plus=lr.predict(X_plus_test_sc)
    new_x=[i for i in range(2019,2026,1)]
    new_x_df=pd.DataFrame(new_x)
    new_x_sc=ss.transform(new_x_df)
    pred_new_years=lr.predict(new_x_sc)
    forecast_df=pd.DataFrame(pred_new_years)
    forecast_year_dtime=[datetime.strptime(str(item)+"-01-01","%Y-%m-%d") for item in new_x]
    forecast_year_dtime_df=pd.DataFrame(forecast_year_dtime)
    merged=pd.concat([forecast_year_dtime_df,forecast_df],axis=1)
    merged.columns=['year','population']
    merged.set_index('year',inplace=True)
    merged['population']=merged['population'].shift(1)
    merged.dropna(inplace=True)

    time=X_test['time'][-1]
    pred_new_years_list=list(pred_new_years)
    predict_first=list(preds)[-1]
    prediction=[]

    for i in range(len(pred_new_years_list)):
        X_test_plus=[1,pred_new_years_list[i],predict_first,time+i]
        new_pred=lm_results.predict(X_test_plus)
        prediction.append(new_pred[0])
        predict_first=new_pred[0]

    prediction=pd.DataFrame(prediction)
    Final=pd.concat([forecast_year_dtime_df,prediction],axis=1)
    Final.columns=['year','forecast-co2']
    Final.set_index('year',inplace=True)





    st.write("## **_R$^2$_** calculated for test data set of "+country_name+' is '+str(round(r2_score(y_test,preds),3)))

    forecast_year=[i for i in range(2020,2026,1)]
    forecast_year_dtime=[datetime.strptime(str(item)+"-01-01","%Y-%m-%d") for item in forecast_year ]

    figure=plt.figure(figsize=(10,10))
    train_plt=plt.plot(y_train.index, y_train.values,linewidth=4,
    color = 'r',label="Train")
    test_plt=plt.plot(y_test.index, y_test.values,linewidth=4,
    color = 'blue',label='Test')
    predict_plt=plt.scatter(y_test.index,lm_results.predict(X_test),s=100,
    color='green', alpha = 0.7,label='Predicted test data')
    forecast_plt=plt.scatter(Final.index,Final.values,s=100,
    color='black', alpha = 0.7,label='Forecast for 2020-2025')
    plt.legend(fontsize=20,loc='best',markerscale=1.5)
    plt.xlabel('Year', fontsize=30)
    plt.ylabel('CO$_2$, million tonnes',fontsize=30)
    plt.title("CO${_2}$ emission",fontsize=35)
    plt.xticks(fontsize=25,rotation=75)
    plt.yticks(fontsize=25)
    st.pyplot(figure)

    # figure=plt.figure(figsize=(10,10))
    # train_plt=plt.plot(y_train.index, y_train.values,linewidth=4,
    # color = 'r',label="Train")
    # test_plt=plt.plot(y_test.index, y_test.values,linewidth=4,
    # color = 'blue',label='Test')
    # predict_plt=plt.scatter(y_test.index,lm_results.predict(X_test),s=100,
    # color='green', alpha = 0.7,label='Predicted test data')
    # forecast_plt=plt.scatter(Final.index,Final.values,s=100,
    # color='black', alpha = 0.7,label='Forecast for 2020-2025')
    # plt.legend(fontsize=20,loc='best',markerscale=3)
    # plt.xlabel('Year', fontsize=30)
    # plt.ylabel('CO$_2$, million tonnes',fontsize=30)
    # plt.title("CO${_2}$ emissions",fontsize=35)
    # plt.xticks(fontsize=25,rotation=75)
    # plt.yticks(fontsize=25)
    # plt.xlim(pd.Timestamp('2015-01-01'),pd.Timestamp('2025-01-01'))
    # plt.ylim(min())
    # st.pyplot(figure)


# options = st.multiselect('What are your favorite colors',
# ['Green', 'Yellow', 'Red', 'Blue'],['Yellow', 'Red'])
#
# st.write('You selected:', options)
