#3d pca
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D


df=pd.read_csv('sample.csv')
print(df.head())

list_col_head=list(df)
list(df)

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
print(df)

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pc1','pc2','pc3'])
x=principalDf.values
print(pca.explained_variance_ratio_)

final=np.column_stack((x,y))
final_df=pd.DataFrame(final,columns=['pc1','pc2','pc3','labels'])
print(names)
print(names_numeric)

import plotly.express as px
import plotly
fig = px.scatter_3d(final_df, x='pc1', y='pc2', z='pc3',
              color='labels')
plotly.offline.plot(fig, "3d.html")



