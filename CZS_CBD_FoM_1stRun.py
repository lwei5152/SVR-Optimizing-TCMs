# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:23:30 2018

@authors: LW and JWA
"""

# Basic packages
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler         # apparently not used, can change colors in the graphs?
import matplotlib as mpl          # also not used, only plt is used 
import pandas as pd               # this deals with dataframes

# stats
import statsmodels.api as sm
from statsmodels.formula.api import ols   # ordinary least squares model 
import statsmodels.stats.multicomp        # option for ANOVA, not used yet

# The machine learning modules
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm


# Get the data into a dateframe.  
df = pd.read_csv("CZS_CBD_1stRun.csv")
#
# this makes a box plot sorted by experiment

df.boxplot('FoM', by='Set', 
               vert=True, 
               boxprops = {'color':'blue'},    #填充颜色！！？？？？
               medianprops = {'linestyle':'-','color':'red'}) # 设置中位数线的属性，线的类型和颜色)
#for box in f['boxes']:
#    # change outline color
#    box.set(color='pink', linewidth=2)
#    # change fill color
#    box.set(facecolor = 'pink' )
plt.style.use('seaborn')
plt.xticks(fontsize=18) 
plt.yticks(fontsize=18)
font = {'size': 18}
plt.xlabel('Set', font)
plt.ylabel(r'FoM ($\mu$S)', font)
plt.title('1st_Run FoM')


#  I did the ANOVA in Matlab.  Should be able to do it here as well. 
#  use ols = ordinary least squares
#  not sure what C(factor) does, maybe groups things
#  format is similar to R

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

## Let's attempt the SVM regression and machine learning
#  I think we can throw out temperature as a variable in this dataset.  

# Make a list of variables for machine learning
#  This is for the first round with one of the factors omitted (additive) per the ANOVA above

names = ('Cu_content', 'Time', 'EDTA')  # temperature not used
variables = df.loc[:, names]            # this will select the columns with the levels of the 3 factors
                                        # I think the variables need to be numerical
# Define a Pipeline that scales the data and applies the model
#  pipeline somehow allows several commands to be strung together
#  scl = Standard scalar
#  clf = ??   A define for classification, i guess (Lwei)
#  svm = Support vector machine
#  SVR = Support Vector Regression
#  rbf = Radial basis function kernel 
#  apparently gamma was adjusted.  I got a better fit at 0.45
#  were default values of C and epsilon used?
#  Yes, C is 1 by default and I am guessing that epsilon is 0.1

reg_FoM = Pipeline([('scl', StandardScaler()),
                    ('clf', svm.SVR(kernel='rbf', gamma=0.3, 
                                    C=40, epsilon = 0.1,
                                    verbose=True))])

# Fit the variables to the PCE
reg_FoM.fit(variables, df.FoM)     # df.cfe gets the column with PCE 

SVM_score=reg_FoM.score(variables,df.FoM)
print('SVM score is', SVM_score)

# Get the predicted PCE from the model and save it to the DataFrame
#  Now there is a new column in there
df['FoM_pred_svm'] = reg_FoM.predict(variables)

# it runs but fit is unless I make C very large.  

# Make a plot of the real values vs the predicted
# Increase gamma in the pipeline until the data just starts on to lay
# on the line. If gamma is too high the data can be over fit
# (JWA, this seems arbitrary)
#  This makes something like Fig. S1 of ms.  

# [Optional] Add color labels to the data
#  Counts the number of unique labels, ""1-01", "1-02" etc.  There are 15
#  and it is added to the dataframe.  
colors = plt.cm.tab20(np.linspace(0, 1, 20)[0:len(df.Set.unique())])
color_dic = {label: color for label, color in zip(df.Set.unique(), colors)}
df['color'] = df.Set.map(color_dic)    # associates a colar with the labels in the dataframe.  

#  Now make the plot of predicted vs. measured PCE
fig, ax1 = plt.subplots(1, 1, 
                        clear=True,       # Clears figure if it already exists, does not work for bar charts?
                        num='Predicted vs measured FoM',   # sets the label at the top of the window
                        figsize=(8, 6))   # 
for label, data in df.groupby('Set'):   # loop through to catch each experimental condition
    plt.plot('FoM_pred_svm', 'FoM', 'o',  
             color=data['color'].iloc[0],
             data=data,                   # refers back to reference to dataframe
             label=label)
plt.legend(loc='upper left', frameon=True)     # make a legend, how to move it?
plt.plot([-1, 320], [0, 320], ls="--", c=".3")  # This draws a dotted line
plt.autoscale(enable=False)                    # turn off autoscale
plt.xlim(0,320)                             # setting plot range here
plt.ylim(0,320)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
font = {'size': 18}
ax1.set_xlabel(r'Predicted FoM ($\mu$S)', font)
ax1.set_ylabel(r'Measured FoM ($\mu$S)', font)

plt.style.use('seaborn')                 # Best style for this plot

plt.tight_layout()                       # finally, show the plot, 
plt.show()                               # tight means push plot to edges of the window
# plt.clf()

#  this is to make plot of Normalized measured-predicted vs. measured 
plt.figure(3)  
fig, ax1 = plt.subplots(1, 1, 
                        clear=True,       # Clears figure if it already exists, does not work for bar charts?
                        num='Predicted vs measured FoM',   # sets the label at the top of the window
                        figsize=(8, 6))   # 
df['min'] = (df.FoM-df.FoM_pred_svm)/df.FoM
for label, data in df.groupby('Set'):   # loop through to catch each experimental condition
    plt.plot('FoM_pred_svm', 'min', 'o',  
             color=data['color'].iloc[0],
             data=data,                   # refers back to reference to dataframe
             label=label)
plt.legend(loc='lower right', frameon=True)     # make a legend, how to move it

plt.xlim(0,320)
plt.ylim(-0.6,0.6)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
font = {'size': 18}
ax1.set_ylabel('Normalized meausured - predicted', font)
ax1.set_xlabel(r'Predicted FoM ($\mu$S)', font)

#plt.style.use('ggplot')
plt.style.use('seaborn')                 # Best style for this plot

plt.tight_layout()                       # finally, show the plot, 
plt.show()                               # tight means push plot to edges of the window


# plt.hist(df.FoM-df.FoM_pred_svm)
# When you figure out how to manage plots.  
# Try to make 2 contour plots, one at 65% Cu and one at 85% Cu?  
#  She varied EDTA between 0.02 and 0.1  
#  She varied time between 60 and 120 

#plt.style.use('ggplot')  
### Plot slices of the 3D fit with value maps
var_n = 2
v_len = 2
vs = np.array([0.65,0.85])               # for the 2 Cu contents
x_len, y_len = 100, 100                  # number of points to make the contour plot
xs = np.linspace(10, 130, x_len)         # vary time between 60 and 120
ys = np.linspace(0.01, 0.11, y_len)       # EDTA range, this will be y axis
vi, xi, yi = names                       # These are 3 strings defined above
# Make as many subplots as there are leves of donor concentration
#  The plots will have the same x and y axes
fig, axs = plt.subplots(nrows=1, ncols=v_len,      # 1 row, 4 columns
                        sharex=True, sharey=True,  # Same x and y axes
                        clear=True, 
                        num='Support Vector Machine Regression, FoM',        # text at top of window
                        figsize=(9.5, 4))

# Slice through the "Cu_content" direction
# Slice through the "Cu_content" direction
for ax, v in zip(axs, vs):                   # zip puts column vectors together
    xm, ym = np.meshgrid(xs, ys)             # makes some sort of mesh
    vm = v * np.ones_like(xm)                # array of ones same shape, then * the value of donor concentration.  
                                             # This set the third value you need to send to the predictor
    r = np.c_[vm.flatten(), xm.flatten(), ym.flatten()]  # flatten matrices into column vectors 
	                                                     # 3 column vectors are the argument to the predictor function
    # Compute the values from the fit
    c = reg_FoM.predict(r).reshape(x_len, y_len)  # feed flatteed mesh r to the predication algorithm.  
	                                              # then reshape the predictions back to a matrix
    # Make a contour map
    cmap = ax.contour(xs, ys, c, vmin=0, vmax=320, cmap='gray_r')   # contour plot, c represent the number of contours in the map
    plt.clabel(cmap, inline=1, fontsize=13)                       # this specifies the contour labels
    # Make a value map   inline=1 can also wrote as inline=True
    pmap = ax.pcolormesh(xs, ys, c, 
                         shading='gouraud',
                         vmin=0, vmax=320, 
                         cmap='viridis')  # this makes the nice colors somehow. 
    # Plot the experimental points - this was overwriting graph
    df.Time = pd.to_numeric(df.Time)
    for label, data in df.query('Cu_content == @v').groupby('Set'):     # finds the cases
        ax.plot('Time', 'EDTA', 'o', 
                color=data['color'].iloc[0],  # used same color for the data points
                data=data.iloc[0],            
                mec='k',                       # with black outline?
                mew=0.5,                       # this one is thickness of line
                label=label)
    ax.legend(loc='upper left', frameon=True) 
   
    font = {'size': 18}

    ax.set_ylabel(f'{yi} (M)', font)                  # sets text for y axis label
    ax.set_xlabel(f'{xi} (min) @ {vi} = {v:.2f}', font) # sets text for x axis label, note forma
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
#Plot the experimental points - this was overwriting graph
#    for label, data in df.query('Cu_content == @v').groupby('label'):     # finds the cases
#        ax.plot('time', 'EDTA', 'o', 
#                color=data['color'].iloc[0],  # used same color for the data points
#               data=data.iloc[0],            
#               mec='k',                       # with black outline?
#                mew=0.5,                       # this one is thickness of line
#                label=label)
#plt.xticks(fontsize=15)
#plt.yticks(fontsize=15)

plt.tight_layout()
plt.colorbar(pmap, ax=axs, fraction=0.04)   # Finally, add color bar.
plt.show()                                  # and display


#fig2, axs2 = plt.subplots(nrows=1, ncols=v_len,      # 1 row, 4 columns
#                        sharex=True, sharey=True,  # Same x and y axes
#                        clear=True, 
#                        num='Support Vector Machine Regression, FoM',        # text at top of window
#                        figsize=(13, 4))
#plt.show() 

###########This is to determine support vectors
# reg_FoM.named_steps.clf.dual_coef_
