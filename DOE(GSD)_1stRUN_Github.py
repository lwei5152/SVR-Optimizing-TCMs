# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:23:30 2018

@authors: LW and JWA
"""

# Basic packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd               # this deals with dataframes

# stats
import statsmodels.api as sm
from statsmodels.formula.api import ols   # ordinary least squares model 

# The machine learning modules
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm


# Get the data into a dateframe.  
df = pd.read_csv("DOE(GSD)_1stRUN.csv")

# this makes a box plot sorted by experiment
df.boxplot('FoM', by='Set', 
               vert=True, 
               boxprops = {'color':'blue'},    
               medianprops = {'linestyle':'-','color':'red'},
               figsize=(12,8)) 

plt.style.use('seaborn')
plt.xticks(fontsize=18) 
plt.yticks(fontsize=18)
font = {'size': 18}
plt.xlabel('Set', font)
plt.ylabel(r'FoM ($\mu$S)', font)
plt.title('2nd_Run FOM')

model = ols('FoM ~ C(Cu_content) + C(Temperature) + C(Time) + C(EDTA)', df).fit()

model.summary()
res = sm.stats.anova_lm(model, typ= 1)   # Need type 1 to agree with paper and with Matlab
print(res)     #  this write the model summary to the console

#  Now make the bar graph
objects = (df.columns[3],   # These are the column labels (factors)
           df.columns[4],
           df.columns[5],
           df.columns[6])

y_pos = np.arange(len(objects))  # this just makes an array [0, 1, 2, 3]
                                 # arrange makes evenly spaced values on a 
                                 # given interval.  Sort of expects integers

totalSSRnoRes = sum(res.sum_sq)-res.sum_sq[-1]  # for normalizing

performance = [res.sum_sq[0]/totalSSRnoRes,     # these are the bar lengths
               res.sum_sq[1]/totalSSRnoRes, 
               res.sum_sq[2]/totalSSRnoRes, 
               res.sum_sq[3]/totalSSRnoRes]

plt.figure(2)                   # can number them but they will not overwrite unless you close them
plt.bar(y_pos, performance, 
        align='center', 
        width=0.8,              # default is 0.8
        alpha=1.0,              # this is transparency, 1.0 is solid
        color=['skyblue', 'peru', 'yellowgreen', 'gold'])
plt.xticks(y_pos, objects)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
font = {'size': 18}
plt.xlabel('Process condition', font)
plt.ylabel('Fraction of total variance', font)
plt.title('1st run ANOVA results', font)

plt.style.use('seaborn')        # this makes a lightgrey background with a nice grid

plt.show()

# Make a list of variables for machine learning
names = ('Cu_content', 'Temperature',  'Time', 'EDTA')   
variables = df.loc[:, names]           

reg_FoM = Pipeline([('scl', StandardScaler()),
                    ('clf', svm.SVR(kernel='rbf', gamma=0.5, 
                                    C=40, epsilon = 0.1,
                                    verbose=True))])

# Fit the variables to the FoM
reg_FoM.fit(variables, df.FoM)     # df.cfe gets the column with FoM 

df['FoM_pred_svm'] = reg_FoM.predict(variables)


colors = plt.cm.tab20(np.linspace(0, 1, 35)[0:len(df.Set.unique())])
color_dic = {label: color for label, color in zip(df.Set.unique(), colors)}
df['color'] = df.Set.map(color_dic)    # associates a colar with the labels in the dataframe.  

#  Now make the plot of predicted vs. measured FoM
fig, ax1 = plt.subplots(1, 1, 
                        clear=True,      
                        num='Predicted vs measured FoM',   # sets the label at the top of the window
                        figsize=(10, 8))   # 
for label, data in df.groupby('Set'):   # loop through to catch each experimental condition
    plt.plot('FoM_pred_svm', 'FoM', 'o',  
             color=data['color'].iloc[0],
             data=data,                   # refers back to reference to dataframe
             label=label)
plt.legend(loc='lower right', frameon=True)     # make a legend
plt.plot([-1, 220], [0, 220], ls="--", c=".3")  # This draws a dotted line
plt.autoscale(enable=False)                    # turn off autoscale
plt.xlim(0,220)                             # setting plot range here
plt.ylim(0,220)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
font = {'size': 24}
ax1.set_xlabel(r'Predicted FOM ($\mu$S)', font)
ax1.set_ylabel(r'Measured FOM ($\mu$S)', font)

plt.style.use('seaborn')                 # Best style for this plot

plt.tight_layout()                       # finally, show the plot, 
plt.show()                               # tight means push plot to edges of the window

#  this is to make plot of Normalized measured-predicted vs. measured 
plt.figure(3)  
fig, ax1 = plt.subplots(1, 1, 
                        clear=True,       
                        num='Predicted vs measured FoM',   # sets the label at the top of the window
                        figsize=(10, 8))   # 
df['min'] = (df.FoM-df.FoM_pred_svm)/df.FoM
for label, data in df.groupby('Set'):   # loop through to catch each experimental condition
    plt.plot('FoM_pred_svm', 'min', 'o',  
             color=data['color'].iloc[0],
             data=data,                   # refers back to reference to dataframe
             label=label)
plt.legend(loc='lower right', frameon=True)    
plt.xlim(0,220)
plt.ylim(-0.6,0.6)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
font = {'size': 24}
ax1.set_ylabel('Normalization', font)
ax1.set_xlabel(r'Predicted FOM ($\mu$S)', font)

#plt.style.use('ggplot')
plt.style.use('seaborn')                 

plt.tight_layout()                       # finally, show the plot, 
plt.show()                               # tight means push plot to edges of the window

u_len = 4    #slice for 4 temperature levels
v_len = 2  # two Cu_content
us = np.array([60, 70, 80, 90])
vs = np.array([0.65,0.85]) # for the 2 Cu contents

x_len, y_len = 100, 100 # number of points to make the contour plot
xs = np.linspace(10, 130, x_len) # vary time between 60 and 120
ys = np.linspace(0.01, 0.11, y_len) # Na2EDTA range, this will be y axis

vi, ui, xi, yi = names 
# names = ('Cu_content', 'Temperature',  'Time', 'EDTA')

fig, axs = plt.subplots(nrows=1, ncols=4,      # 2 row, 4 columns, present position=1
                        sharex=True, sharey=True,   # Same x and y axes
                        clear=True, 
                        num='Support Vector Machine Regression, FOM',        # text at top of window
                        figsize=(20, 4.5))  

for ax, u in zip(axs, us): 
    # zip puts column vectors together, two loops: ax = axs[0,0] and v=0.65; ax = axs[0,1] and v=0.85 
    xm,ym = np.meshgrid(xs,ys) # makes some sort of mesh, 
    vm = 0.85 * np.ones_like(xm) # change Cu_content at each run to obtain different occasions.
# can also use loop to achieve, will be more complicated
    um = u * np.ones_like(xm)
    # This set the third value you need to send to the predictor
    r = np.c_[vm.flatten(), um.flatten(), xm.flatten(), ym.flatten()] 
    # flatten matrices into column vectors 
    c = reg_FoM.predict(r).reshape(x_len, y_len) 
    # feed flatteed mesh r to the predication algorithm, then reshape the predictions back to a matrix
  
    # Make a contour map
    cmap = ax.contour(xs, ys, c, vmin=0, vmax=220, cmap='gray_r')
    plt.clabel(cmap, inline=1, fontsize=13) # this specifies the contour labels
    # Make a value map inline=1 can also wrote as inline=True
    pmap = ax.pcolormesh(xs, ys, c, 
                    shading='gouraud',
                    vmin=0, vmax=220, 
                    cmap='viridis') # this makes the nice colors 
    df.Time = pd.to_numeric(df.Time)
    for label, data in df.query('Temperature == @u and Cu_content == 0.85').groupby('Set'): # finds the cases
        ax.plot('Time', 'EDTA', 'o', 
                color=data['color'].iloc[0], # used same color for the data points
                data=data.iloc[0], 
                mec='k', # with black outline
                mew=0.5, # line thickness
                label=label)
        ax.legend(loc='upper left', frameon=True) 
        font={'size': 18}
        ax.set_ylabel(f'{yi} (M)', font) # sets text for y axis label
        ax.set_xlabel(f'{xi} (min)', font) 
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.tight_layout()
plt.colorbar(pmap, ax=axs, fraction=0.04) # Finally, add color bar.
plt.show() # and display


