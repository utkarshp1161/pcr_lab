#raw_plot_after_preprocessing
#raw_plot
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D


df=pd.read_csv('sample.csv')
print(df.head())

list_col_head=list(df)
print(list(df))


df['Label'].unique() 
names=df['Label'].unique()
print(names)

# Import label encoder 
from sklearn import preprocessing 
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
# Encode labels in column 'species'. 
df['Label']= label_encoder.fit_transform(df['Label']) 
df['Label'].unique() 
#df['Label']
names_numeric=df['Label'].unique() 
print(names_numeric)

df_labels=df[['Label']]
df=df.drop(['Label'],axis=1)
y=df_labels.values



from sklearn  import preprocessing
X=preprocessing.scale(df)
#print(df)

final=np.column_stack((X,y))
final_df=pd.DataFrame(final,columns=list_col_head)
print(final_df.head())

import plotly.express as px
import plotly
#iris = px.data.iris()
fig = px.scatter(final_df, x=list_col_head[1], y=list_col_head[0],
              color='Label')
plotly.offline.plot(fig, "raw_plot_post_preprocessing.html")



