#!/usr/bin/env python
# coding: utf-8
# %%

# ### Import Libraries

# %%


#Import Libraries
import numpy as np #Basic operations
import pandas as pd #For dataframe manipulations
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt #For data visualization
import seaborn as sns
import plotly.express as px #For plotting graphs
from sklearn.utils import shuffle
from category_encoders import TargetEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, mean_absolute_error, mean_squared_error


# ### Read the data

# %%


df= pd.read_csv('D:/KDU BLENDER LEARNING/Introduction to Data Science/New datasetr/API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_4250827.csv')
df.head(5)


# ### Looking at the columns

# %%


df.columns


# - As there are some column categories that do not have any relation with what we are doing, we are going to drop Country Code, Indicator Name, Indicator Code

# ### Dropping Unnecesary columns in the dataset

# %%


drop_cols = ["Country Code", "Indicator Name", "Indicator Code", "Unnamed: 66"]
df.drop(drop_cols, axis=1, inplace=True)


# %%


df.columns


# ### Looking at the values in the dataset

# %%


#Setting the graphs
plt.rcParams.update({'font.size': 10})
sns.set_style("darkgrid")


# %%


plt.figure(figsize = (15,8))
plt.xticks(rotation=90)
sns.dark_palette("#69d", reverse=True, as_cmap=True)
sns.barplot(x=df.columns, y=df.isna().sum(),linewidth=2.5, edgecolor=".2",palette='YlGnBu_r')
plt.xlabel("Columns name")
plt.ylabel("Number of missing values in the Inflation dataset")
plt.show()


# - Some of the countries have missing values in the dataset, most countries have missing data as early as 1960, we will need to clean the dataset first in order to have an accurate representation.

# #### How many countries have missing values?Which ones?

# %%


print("The amount of countries that have null values are: ",df.isnull().any(axis = 1).sum())


# - There are a total of 202 countries that have missing data since 1960, we are going to isolate those countries and create a new dataset with them.

# %%


df[df.isna().any(axis=1)]


# ### Dropping the columns with Null values

# %%


da = df.dropna(axis = 0) 
da


# %%


print("The amount of countries that have null values are: ",da.isnull().any(axis = 1).sum())


# - We have clean the dataset of all null values, now let's see how many countries have a full data completion.

# %%


da.count()


# - We have 64 countries with full data starting from 1960

# %%


# to expand the maximum output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# ### Transpose the Dataset and Reindex

# %%


def da_null(num):
    null=da.loc[[num]].isna().sum().sum()
    name= da.loc[num,'Country Name']
    
    return  name, null


# %%


transp= da.T
transp.columns = transp. iloc[0]

transp=transp.drop(index='Country Name')
transp.reset_index(inplace=True)
transp=transp.rename(columns = {'index':'Year'})
transp


# - Now we have a easier look for  the dataset

# %%


columns= transp.columns[1:]
columns


# ### Line plot for Inflation Rate of countries

# %%


plt.rcParams['figure.figsize']=15,8
plt.ylabel("inflation rate")
plt.xticks(rotation=90)

for country in columns:
    sns.lineplot(data=transp,x='Year',y=country)
    
plt.legend(columns)
    
    
plt.show()


# - The lineplot of countries does not give an accurate representation of the countries, therefore we are going to split them up and look at their Inflation rate individually.

# %%


import warnings
warnings.filterwarnings("ignore")
fig, axes = plt.subplots(9,2, figsize=(25,25))
fig.subplots_adjust(hspace=.3, wspace=.175)
for ax, col in zip(axes.flat,columns):
    
    sns.lineplot(data=transp,x='Year',y=col,color='Blue',ax=ax)
    ax.set_title(col,fontweight="bold")
    ax.set_xticklabels(transp['Year'].tolist(),rotation=90)
    
fig.tight_layout()
fig.show()


# This LinePlot tells us that Brazil had got the highest amount of inflation were Bolivia, Costa Rice and Ecuador. Next, We are going to look into the Inflation rate details.

# ### Countries with the most and least mean inflations rates.

# %%


#Creating the new value Rate
da = pd.melt(da, id_vars='Country Name', var_name='Year', value_name='Rate')
print(f'Total number of countries and territories: ', da['Country Name'].nunique())
da.head()


# %%


da['Year'] = da['Year'].astype('int32')
da.info()


# ### The min, max and mean of countries

# %%


da.groupby('Country Name')[['Rate']].agg(['min','max','mean'])


# #### Highest average Inflation Rate 

# %%


#From the highest Inflation Rate since 1960
da[da['Year']>=1960].groupby('Country Name').mean().nlargest(10, 'Rate')


# - we can see the average highest Inflation rates are Bolivia, Peru and Indonesia. This is a different result from the highest Inflation Rate ever which were Bolivia, Costa Rice and Ecuador. Although Costa Rice and Ecuador have the highest ever Inflation Rate recorded, it was only for a short period of time compared to Peru and Indonesia which has the highest average.

# #### Lowest average Inflation Rate

# %%


#From the lowest Inflation Rate since 1960
da[da['Year']>=1960].groupby('Country Name').mean().nsmallest(10, 'Rate')


# - The lowest average Inflation Rates are Switzerland, Germany and Panama. These Countries have a lower Inflation since 1960.

# ### Graph for the average Inflation Rate between the Countries

# %%


plt.figure(figsize = (20, 10))
da.groupby('Country Name')['Rate'].mean().plot(kind = 'bar', color='g')
plt.title("The Average Inflation Rate of Countries over the years", fontsize = 20)
plt.show()


# ### The highest Inflation value from each country

# %%


transp.max()


# ### Highest Inflation rate of alltime

# %%


da[da['Rate'] > 350]


# These values reveal the following as the top five highest ever inflation rates:
#     1) Bolivia - 11749.63%
#     2) Peru - 7481.66%
#     3) Indonesia - 1136.25%
#     4) Sudan - 382.81%
#     5) Isreal - 373.21
# These five countries have the highest maximum inflation in history.

# These countries had the lowest Inflation rate compared to other countries, Although it the the highest ever in their own countries: 
#     1. Germany - 7%
#     2. Austria - 9.5%
#     3. Switzerland - 9.8%
#     4. Netherlands - 10.2%
#     5. Luxembourg - 10.7%

# ### Lowest Inflation rate of all time

# %%


da[da['Rate'] < -5]


# - These countries contained the lowest Inflation

# ### Inflation Analysis

# We will now begin to analyze the evolution of the Inflation and some characteristics in some relevant years of different Countries.

# Now we are going to compare side to side the countries with the most Inflation rates. Bolivia, Peru and Indonesia

# %%


def plot_country(da:pd.DataFrame, country:str):
    sns.set_style("white")
    data = da[da["Country Name"] == country]
    fig, ax = plt.subplots(1,1,figsize=(15,4))
    ax.plot(data.Year, data.Rate, color = "slategrey", marker="o", linewidth=2)
    ax.fill_between(data.Year, 0, data.Rate, alpha=0.2, color='g', where=(data.Rate >= 0))
    ax.fill_between(data.Year, 0, data.Rate, alpha=0.2, color='r', where=(data.Rate < 0))
    ax.set_xticks(data.Year)
    ax.set_xticklabels(data.Year, rotation=90)
    ax.set_title("Inflation evolution in "+country, loc="left", fontsize=20)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Inflation in %", fontsize=14)
    sns.despine()


# %%


for country in ["Bolivia", "Peru", "Indonesia"]:
    plot_country(da, country)


# - These 3 countries have a common point, they all have A year where they peaked in Inflation Rate then it drops dramatically over the next years. The 3 countries were all suffering from political instability.

# Lets take a look into the Countries with the least Inflation.

# %%


for country in ["Switzerland", "Germany", "Austria"]:
    plot_country(da, country)


# - Unlike the countries that have hyperinflation over short periods of time, the Inflation Rate of Switzerland, Germany and Germany rose over a period of time and Lowers over the years.These countries never had an Inflation Rate higher then 10% since 1970.

# ### Heatmap

# %%


def countriesCorr(da, countries):
    table = da.pivot(index='Year', columns='Country Name', values='Rate')[countries]
    f, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(table.corr(), cmap="YlGnBu", annot=True, linewidths=1, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title("Country inflation correlation", fontsize=18)
    plt.show()

countries = ["Bolivia", "Peru", "Indonesia","Sudan", "Israel", "Germany", "Austria", "Switzerland", "Netherlands", "Luxembourg"]
countriesCorr(da, countries)


# ### The overall Inflation rate of countries since 1960

# %%


plt.figure(figsize = (20, 20))
da.groupby('Year').median()['Rate'].plot(kind ='barh', fontsize = 15, color='brown')


# We can see from this graph that most countries have a high Inflation Rate from 1974, a total of 16 countries has seen a high Inflation rate that year, compared to recent years where contries have a fairly low Inflation rate.

# ### Calculating percentiles of countries that were colonized and non colonized

# For non colonized country we are going to use Germany, while for colonized Country we are going to use India.

# %%


for i in range(90, 100): print("The {:.1f}th percentile value is {:.2f}".format(i, np.percentile(transp['Germany'],i)))


# %%


for i in range(90, 100): print("The {:.1f}th percentile value is {:.2f}".format(i, np.percentile(transp['India'],i)))


# ### Inflation rates affecting both colonized and non-colonized countries

# For this we will split up the countries into seperate categories, Colonized and non-colonized countries. 

# %%


# non-colonized countries
non_country = transp[[ "Austria", "Belgium",  "Switzerland", "Cyprus", "Denmark", "Dominican Republic", "Finland", "France", "Germany","Italy", "United Kingdom", "Spain", "Greece","Ireland", "Iceland", "Japan", "Luxembourg", "Mexico", "Malta", "Netherlands", "Norway", "Portugal", "Sweden", "Thailand", "Turkiye"]]
non_country


# - The countries that were not colonized are Austria, Belgium,  Switzerland, Cyprus, Denmark, Dominican Republic, Finland, Germany, France, Italy, United Kingdom, Spain, Greece, Ireland, Iceland, Japan, Luxembourg, Mexico, Malta, Netherlands, Norway, Portugal, Sweden, Thailand, Turkiye.

# %%


#colonized country
colo_country = transp[["Australia","Burkina Faso", "Bolivia", "Canada","Colombia", "Costa Rica", "Ecuador", "Egypt, Arab Rep.", "Dominican Republic", "Honduras", "Haiti", "Indonesia", "India", "Israel","Italy", "Jamaica", "Kenya", "Korea, Rep.", "Latin America & Caribbean (excluding high income)", "Sri Lanka", "Morocco", "Malaysia", "North America", "Nigeria", "New Zealand", "Pakistan", "Panama", "Peru", "Philippines", "Paraguay", "Sudan", "El Salvador", "Uruguay", "South Africa", "United States"  ]]
colo_country


# - The countries that were colonized before were Australia,Burkina Faso, Bolivia, Canada, Colombia, Costa Rica, Ecuador, Egypt, Arab Rep., Dominican Republic, Honduras, Haiti, Indonesia, India, Israel, Jamaica, Kenya, Korea, Rep., Latin America & Caribbean, Sri Lanka, Morocco, Malaysia, North America, Nigeria, New Zealand, Pakistan, Panama, Peru, Philippines, Paraguay, Sudan, El Salvador, Uruguay, South Africa, United States.

# %%


def compare_countries(transp:pd.DataFrame, countries:list):
    if len(countries) > 2:
        raise ValueError('More than two countries passed')
    sns.set_style("white")
    data = transp[transp["Country Name"].isin(countries)]
    fig, ax = plt.subplots(1,1,figsize=(25,4))
    sns.lineplot(x='Year', y='Rate', hue='Country Name', data=data, ax=ax, marker="o", linewidth=2, markersize=8)
    ax.set_xticks(transp.Year)
    ax.set_xticklabels(transp.Year, rotation=90)
    ax.set_title(f"Inflation comparison between {countries[0]} and {countries[1]}", loc="left", fontsize=20)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Inflation in %", fontsize=14)
    sns.despine()


# We will now start to compare some of the Inflation Rates of colonized countries and non-colonized countries to see their effects and can colonized countries recover from the Inflation rate when they declared Independence as well as can colonized countries do better then non-colonized countries in terms of growth.

# ### Comparing colonized and non colonized countries

# %%


compare_countries(da, ['Sweden', 'Australia'])


# Sweden was a country that was never colonized while Australia was colonized by British and later gain Independence. This graph shows that both countries had a similar Inflation Rate throughout the years.

# ### Colonized countries with different Inflation Rate over the years.

# %%


compare_countries(da, ['Malaysia', 'Philippines'])


# %%


compare_countries(da, ['South Africa', 'Kenya'])


# Countries that were colonized have different Inflation rate progressions, even though they were colonized, some countries deal with their Inflation rate better then others. Malaysia and Philippines are both countries in Asia and were colonized, Malaysia has a lower inflation Rate compared to the Philipines even though they gain their independence earlier then Malaysia. South Africa and Kenya were colonized by the same United Kingdom but South Africa had a more stable Inflation then Kenya.

# ### The influence of colonized countries  on countries they colonize

# %%


def compare_mcountries(transp:pd.DataFrame, countries:list):
    if len(countries) > 3:
        raise ValueError('More than three countries passed')
    sns.set_style("white")
    data = transp[transp["Country Name"].isin(countries)]
    fig, ax = plt.subplots(1,1,figsize=(25,4))
    sns.lineplot(x='Year', y='Rate', hue='Country Name', data=data, ax=ax, marker="o", linewidth=2, markersize=8)
    ax.set_xticks(transp.Year)
    ax.set_xticklabels(transp.Year, rotation=90)
    ax.set_title(f"Inflation comparison between {countries[0]} and {countries[1]} and {countries[2]}", loc="left", fontsize=20)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Inflation in %", fontsize=14)
    sns.despine()


# %%


compare_mcountries(da, ['Spain', 'Costa Rica', "Colombia"])


# - The graph shows the relation between the colonizing countries which are Costa Rica and Colombia, and the country that colonized them which was Spain. Spain has a lower Inflation rate compared to the countries that it has colonized.

# %%


compare_mcountries(da, ['France', 'Burkina Faso', "Haiti"])


# - Another example is France which is the colonizing country, and the ones that were colonized are Burkina Faso and Haiti. France has a stable economy Inflation over the years without any massive changes. While Haiti and Burkina Faso has seen some dramatical changes over the years.

# ### Colonized countries that have better Inflation Rate then non colonized countries

# %%


compare_mcountries(da, ['United Kingdom', 'United States', "Canada"])


# The United states and Canada gained their independence before 1960, this graph shows that the United kingdom which were the ones who colonized the United States and Canada, have a lower Inflation Rate throughout the years. The graph shows a Increasing Inflation Rate in the 1974 due to the world being unprepared for higher oil prices, cars were not fuel efficient and there were fewer alternatives to oil. Canada and United States handled their Inflation better then the country which colonized them.

# - This proves that not all colonized countries are the same with dealing with Inflation, being colonized does not mean the country will have a high Inflation Rate.

# Hypothesis : Countries colonized have a higher inflation rate than countries without historical colonization. Research Question : Does the history of colonization influence the inflation rate of colonized countries?

# Conclusion: Colonized Countries have on average higher Inflation Rates compared to their non-colonized counterparts. Colonizing countries does have some impact on their Inflation Rate but it does not remain the same for others. Some countries can have a lower Inflation Rate then the ones that colonized them. Variously factors contribute to Inflation, although colonizing has some Impact, it is not the deciding factor of Inflation. 

# %%




