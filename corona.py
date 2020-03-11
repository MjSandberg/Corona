from scipy import stats
import numpy as np
import sys
import matplotlib.pyplot as plt
from probfit import Chi2Regression, BinnedLH, UnbinnedLH
from iminuit import Minuit
from scipy.optimize import curve_fit
import datetime as dt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from bs4 import BeautifulSoup
import datetime
from requests import get
from astropy.io import ascii

# %%
url = "https://politi.dk/coronavirus-i-danmark/foelg-smittespredningen-globalt-regionalt-og-lokalt"
request = get(url)
soup = BeautifulSoup(request.text,"html.parser")
first = soup.find(class_="table-overflow")
first_2 = first.find_all('tr')

newy = float(re.findall('\d+', str(first_2[1]))[0])


d0 = 5
newx = (pd.datetime.now().day+pd.datetime.now().hour/24)-d0


# %%

y = np.loadtxt("smittede.txt",delimiter=",")
x = np.loadtxt("dage.txt",delimiter=",")

if ~np.isin(newy,y):
    y = np.append(y,newy)
    x = np.append(x, newx)

np.savetxt("dage.txt",x,delimiter=",")
np.savetxt("smittede.txt",y,delimiter=",")

def exp(x,a,b,c):
    return a*np.exp(b*x)+c
# %%
chi2_object = Chi2Regression(exp, x, y, error=None)
minuit = Minuit(chi2_object, pedantic=False, a = 2, b = 1, c = 1, print_level=0)
minuit.migrad()  # perform the actual fit
chi2 = sum((exp(x,*minuit.args) - y)**2)

NDOF = len(x) - len(minuit.args)
chi2_prob =  stats.chi2.sf(chi2, NDOF)

print(minuit.args)
print("chi2 = ", chi2)
print("chi2_prob = ", chi2_prob)
print(y)
# %%

max(x)
lin = np.linspace(0,max(x)+2,200)

fig = go.Figure()

fig.add_trace(go.Scatter(x=lin,y=exp(lin,*minuit.args), name = "fit: y=a*exp(b*x)+c", line = dict(color="royalblue",width=2.6) ) )
fig.add_trace(go.Scatter(x=x,y=y,name ="data",mode="lines+markers", line = dict(color="firebrick",width=2), marker=dict(color="firebrick", size=7 )) )

fig.update_layout(title='Smittede i Danmark plus fit ',
                   xaxis_title='Dage siden Torsdag d. 5/2-2020',
                   yaxis_title='Antal smittede',
                   legend=dict(font_size=16))

fig.show()

# %%
r =1 - sum( (np.mean(y)-exp(x,*minuit.args))**2) /sum( (y-np.mean(y))**2)
