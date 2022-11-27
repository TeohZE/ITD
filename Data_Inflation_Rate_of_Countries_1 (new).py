#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[2]:


#Read dataset
df= pd.read_csv('D:/KDU BLENDER LEARNING/Introduction to Data Science/New datasetr/API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_4250827.csv')
df.head(5)


# In[3]:


#Looking at the columns
df.columns


# In[4]:


#Drop innecesary columns
drop_cols = ["Country Code", "Indicator Name", "Indicator Code", "Unnamed: 66"]
df.drop(drop_cols, axis=1, inplace=True)


# In[5]:


#Creating the new value Rate
inflation_df = pd.melt(df, id_vars='Country Name', var_name='Year', value_name='Rate')
print(f'Total number of countries and territories: ', inflation_df['Country Name'].nunique())
inflation_df.head()


# In[6]:


#looking at the null values and Dtype
inflation_df.info()


# # General Plot

# In[7]:


#Setting the graphs
plt.rcParams.update({'font.size': 10})
sns.set_style("darkgrid")


# In[8]:


plt.figure(figsize = (15,8))
plt.xticks(rotation=90)
sns.dark_palette("#69d", reverse=True, as_cmap=True)
sns.barplot(x=df.columns, y=df.isna().sum(),linewidth=2.5, edgecolor=".2",palette='prism_r')
plt.xlabel("Columns name")
plt.ylabel("Number of missing values in inflation_df")
plt.show()


# # Inflation Analysis

# We will now begin to analysis analyze the evolution of the Inflation and some characteristics in some relevant years.

# In[9]:


#The Dataframe
g20 = df.loc[df['Country Name'].isin(["Argentina", "Australia", "Brazil", "Canada", "China", "France", "Germany", "India", "Indonesia", "Italy", "Japan", "Mexico", "Russian Federation","South Africa", "Saudi Arabia", "Korea, Rep.", "Turkiye", "United Kingdom", "United States"])].copy()
g20


# #### Since G20 holds a strategic role in securing future global economic growth and prosperity. Together, the G20 members represent more than 80 percent of world GDP, 75 percent of international trade and 60 percent of the world population. And the data is obtained from thw world bank,we will name our dataframe as G20.

# #### MISSING VALUE G20 BAR PLOT

# In[10]:


plt.figure(figsize = (15,8))
plt.xticks(rotation=90)
sns.barplot(x=g20.columns, y=g20.isna().sum(),linewidth=2.5, edgecolor=".2",palette='mako')
plt.xlabel("Columns name")
plt.ylabel("Number of missing values in g20 dataset")
plt.show()


# #### How many countries have missing values?Which ones?

# In[11]:


print("The amount of countries that have null values are: ",g20.isnull().any(axis = 1).sum())


# In[12]:


g20[g20.isna().any(axis=1)]


# As we can see the countries that have null values are: Argentina, Brazil (the South America Countries that are part of the G20), China and Russian Federation(that are the world opposition to the United States) and Saudi Arabia.
# Now we are going to see how many null values have.

# # Transpose the Dataset and Reindex

# In[13]:


def g20_null(num):
    null=g20.loc[[num]].isna().sum().sum()
    name= g20.loc[num,'Country Name']
    
    return  name, null


# In[14]:


print(g20_null(9)[0],"has got",g20_null(9)[1],".This means that has all missing values.")
print(g20_null(29)[0],"has got",g20_null(29)[1], "missing values.")
print(g20_null(40)[0],"has got",g20_null(40)[1], "missing values.")
print(g20_null(202)[0],"has got",g20_null(202)[1],"missing values.")
print(g20_null(205)[0],"has got",g20_null(205)[1], "missing values.")


# Since Argentina has no data, we will be removing it as it makes no sense to have it.

# In[15]:


transp= g20.T
transp.columns = transp. iloc[0]

transp=transp.drop(index='Country Name')
transp.reset_index(inplace=True)
transp=transp.rename(columns = {'index':'Year'})
transp.drop(['Argentina'], inplace=True, axis=1)
transp


# In[16]:


columns= transp.columns[1:]
columns


# # List of G20 Countries 

# In[17]:


plt.rcParams['figure.figsize']=15,8
plt.ylabel("inflation rate")
plt.xticks(rotation=90)

for country in columns:
    sns.lineplot(data=transp,x='Year',y=country)
    
plt.legend(columns)
    
    
plt.show()


# This LinePlot tells us that Brazil had got the highest amount of inflation,Indonesia and Russian Federation follows.Next,
# We are going to plot the countries individually to see their Inflation rate.

# In[18]:


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


# ## Bar plot

# In[31]:


fig, axes = plt.subplots(9,2, figsize=(25,25))
fig.subplots_adjust(hspace=.3, wspace=.175)
for ax, col in zip(axes.flat,columns):
    sns.barplot(data=transp,x='Year',y=col,color='Blue',ax=ax)
    ax.set_title(col,fontweight="bold")
    ax.set_xticklabels(transp['Year'],rotation=90)
    
fig.tight_layout()
fig.show()


# Now we are going to compare side to side the countries with the most Inflation rates.Brazil, Indonesia and Russian Federation.

# In[32]:


three= ['Brazil','Indonesia','Russian Federation']


# In[33]:


fig, axes = plt.subplots(3,1, figsize=(10,10))
fig.subplots_adjust(hspace=.3, wspace=.175)
for ax, col in zip(axes.flat,three):
    sns.lineplot(data=transp,x='Year',y=col,color='Green',linewidth=2.5,marker="o",ax=ax)
    sns.barplot(data=transp,x='Year',y=col,color='Orange',ax=ax)
    ax.set_title(col,fontweight="bold")
    ax.set_xticklabels(transp['Year'],rotation=90)
    
fig.tight_layout()
fig.show()


# Here we can see that brazil has an Long-lasting episodes of high inflation beginning in the 1990s, Indonesia has an Long-lasting episodes of high inflation beginning in 1960s and Russia has an Long-lasting episodes of high inflation after the Fall of the Soviet Union.

# Next we can see some countries have their inflation rate increased in the last years like United States, Germany, Canada,Turkiye. This is what is call Pandemic Inflation Rate.

# In[ ]:


four=['United States','Germany','Canada','Turkiye']


# In[ ]:


fig, axes = plt.subplots(4,1, figsize=(10,10))
fig.subplots_adjust(hspace=.3, wspace=.175)
for ax, col in zip(axes.flat,four):
    sns.lineplot(data=transp,x='Year',y=col,color='Green',linewidth=2.5,marker="o",ax=ax)
    sns.barplot(data=transp,x='Year',y=col,color='Orange',ax=ax)
    ax.set_title(col,fontweight="bold")
    ax.set_xticklabels(transp['Year'],rotation=90)
    
fig.tight_layout()
fig.show()


# As we can see United States, Germany and Canada have got the highest inflation rate since 90s and Turkiye has got the highest since 2003 .

# In[ ]:





# In[ ]:


inflation_df['Year'] = inflation_df['Year'].astype('int32')
inflation_df.info()


# In[ ]:


#From the highest Inflation Rate since 1960
inflation_df[inflation_df['Year']>=1960].groupby('Country Name').mean().nlargest(10, 'Rate')


# In[ ]:


#From the lowest Inflation Rate since 1960
inflation_df[inflation_df['Year']>=1960].groupby('Country Name').mean().nsmallest(10, 'Rate')


# In[ ]:


def plot_country(inflation_df:pd.DataFrame, country:str):
    sns.set_style("white")
    data = inflation_df[inflation_df["Country Name"] == country]
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


# In[ ]:


for country in ["Spain", "France", "Italy", "Germany", 'Venezuela, RB', "Malaysia", 'Singapore']:
    plot_country(inflation_df, country)


# In[ ]:


def compare_countries(inflation_df:pd.DataFrame, countries:list):
    if len(countries) > 2:
        raise ValueError('More than two countries passed')
    sns.set_style("white")
    data = inflation_df[inflation_df["Country Name"].isin(countries)]
    fig, ax = plt.subplots(1,1,figsize=(15,4))
    sns.lineplot(x='Year', y='Rate', hue='Country Name', data=data, ax=ax, marker="o", linewidth=2, markersize=8)
    ax.set_xticks(data.Year)
    ax.set_xticklabels(data.Year, rotation=90)
    ax.set_title(f"Inflation comparison between {countries[0]} and {countries[1]}", loc="left", fontsize=20)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Inflation in %", fontsize=14)
    sns.despine()


# In[ ]:


compare_countries(inflation_df, ['Spain', 'France'])


# In[ ]:


compare_countries(inflation_df, ['Malaysia', 'Singapore'])


# ### Heatmap

# In[ ]:


def countriesCorr(inflation_df, countries):
    table = inflation_df.pivot(index='Year', columns='Country Name', values='Rate')[countries]
    f, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(table.corr(), cmap="YlGnBu", annot=True, linewidths=1, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title("Country inflation correlation", fontsize=18)
    plt.show()

countries = ['Spain', 'France', 'Germany', 'Italy', 'Norway', 'Portugal', 'Poland', 'Romania', 'Denmark', 'United Kingdom']
countriesCorr(inflation_df, countries)


# In[ ]:


new_data = df.dropna(axis = 0) 
new_data


# In[ ]:


print("The amount of countries that have null values are: ",new_data.isnull().any(axis = 1).sum())


# In[ ]:


def new_data_null(num):
    null=new_data.loc[[num]].isna().sum().sum()
    name= new_data.loc[num,'Country Name']
    
    return  name, null


# In[ ]:


transp= new_data.T
transp.columns = transp. iloc[0]

transp=transp.drop(index='Country Name')
transp.reset_index(inplace=True)
transp=transp.rename(columns = {'index':'Year'})
transp


# In[ ]:


columns= transp.columns[1:]
columns

Not been conolized: Austria, Belgium,  Switzerland, Cyprus, Denmark, Dominican Republic,  Spain,  Finland(Sweden), France, United Kingdom, Greece, Ireland, Iceland, Japan, Luxembourg, Mexico, Malta, Netherlands, Norway, Portugal, Sweden, Thailand, Turkiye,    


Been Conolized: Australia,Burkina Faso(French),Canada, Bolivia(Spanish), Colombia(Spanish), Costa Rica(Spain), Germany(Britain, Holland, France and Spain), Ecuador(Spain), Egypt, Arab Rep.(French, British), Guatemala(Spain), Honduras(Spain), Haiti(Span), Indonesia(Dutch), India(Europe), Israel(British), Italy(British,Spain Russia and Spain), Jamaica(British), Kenya(British),
Korea, Rep.(japan), Latin America & Caribbean(Spain, France, Protugal), Sri Lanka(Portugal, the Netherlands and Great Britain), Morocco(France and Spain), Malaysia(Portuguese, Dutch, British), North America(Britain, France, Spain, and the Netherlands), Nigeria(British), New Zealand(British), Pakistan(British), Panama(Spanish), Peru(Spanish),Philippines(Spanish), Paraguay(Spanish), Sudan(British), El Salvador(Spanish), Uruguay(Europeans), South Africa (Netherlands and british),United States,

 
Other: OECD members, Post-demographic dividend, Latin America & the Caribbean, Euro area, European Union,
# ### Non Colonized Country

# In[29]:


non_country = df.loc[df['Country Name'].isin([ "Austria", "Belgium",  "Switzerland", "Cyprus", "Denmark", "Dominican Republic", "Finland", "France", "United Kingdom", "Spain", "Greece","Ireland", "Iceland", "Japan", "Luxembourg", "Mexico", "Malta", "Netherlands", "Norway", "Portugal", "Sweden", "Thailand", "Turkiye"  ])].copy()
non_country


# ### Colonized Country

# In[ ]:


Been Conolized: Burkina Faso(French), Bolivia(Spanish), Colombia(Spanish), Costa Rica(Spain), Germany(Britain, Holland, France and Spain), Ecuador(Spain), Egypt, Arab Rep.(French, British), Guatemala(Spain), Honduras(Spain), Haiti(Span), Indonesia(Dutch), India(Europe), Israel(British), Italy(British,Spain Russia and Spain), Jamaica(British), Kenya(British),
Korea, Rep.(japan), Latin America & Caribbean(Spain, France, Protugal), Sri Lanka(Portugal, the Netherlands and Great Britain), Morocco(France and Spain), Malaysia(Portuguese, Dutch, British), North America(Britain, France, Spain, and the Netherlands), Nigeria(British), New Zealand(British), Pakistan(British), Panama(Spanish), Peru(Spanish),Philippines(Spanish), Paraguay(Spanish), Sudan(British), El Salvador(Spanish), Uruguay(Europeans), South Africa (Netherlands and british)


# In[30]:


colo_country = df.loc[df['Country Name'].isin(["Australia","Burkina Faso", "Bolivia", "Canada","Colombia", "Costa Rica", "Germany", "Ecuador", "Egypt, Arab Rep.", "Dominican Republic", "Honduras", "Haiti", "Indonesia", "India", "Israel","Italy", "Jamaica", "Kenya", "Korea, Rep.", "Latin America & Caribbean", "Sri Lanka", "Morocco", "Malaysia", "North America", "Nigeria", "New Zealand", "Pakistan", "Panama", "Peru", "Philippines", "Paraguay", "Sudan", "El Salvador", "Uruguay", "South Afric", "United States"  ])].copy()
colo_country


# In[25]:


plt.rcParams['figure.figsize']=15,8
plt.ylabel("inflation rate")
plt.xticks(rotation=90)

for country in columns:
    sns.lineplot(data=transp,x='Year',y=country)
    
plt.legend(columns)
    
    
plt.show()


# In[ ]:


fig, axes = plt.subplots(9,2, figsize=(25,25))
fig.subplots_adjust(hspace=.3, wspace=.175)
for ax, col in zip(axes.flat,columns):
    sns.barplot(data=transp,x='Year',y=col,color='Blue',ax=ax)
    ax.set_title(col,fontweight="bold")
    ax.set_xticklabels(transp['Year'],rotation=90)
    
fig.tight_layout()
fig.show()


# In[26]:


transp['Bolivia'].isnull().sum()


# In[ ]:





# In[ ]:





# In[ ]:




