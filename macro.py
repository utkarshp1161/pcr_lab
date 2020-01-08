import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D


#READ SHEET
import sys
file_to_open=sys.argv[1] # 0 is the index of name of python prog, 1 is the index of first argument in command line.
#df=pd.read_csv('macro_input.csv')
df=pd.read_csv(file_to_open,sep=',')



#print(df.head())#FINDING MAX OF EACH Column
Rmax1=df['R1'].max()
Rmax2=df['R2'].max()
Rmax3=df['R3'].max()
df['Rmax1']=Rmax1
df['Rmax2']=Rmax2
df['Rmax3']=Rmax3


# Rmax-R
delta_R1=abs(df['R1'].subtract(Rmax1))
delta_R2=abs(df['R2'].subtract(Rmax2))
delta_R3=abs(df['R3'].subtract(Rmax3))
#delta_R1.head()


# (Rmax-R)/Rmax
res1=delta_R1/Rmax1
res2=delta_R2/Rmax2
res3=delta_R3/Rmax3


# (Rmax-R)/Rmax * 100
res1_percent=res1*100
res2_percent=res2*100
res3_percent=res3*100





#Putting everything in one excel sheet
array1=df['R1'].values
array2=df['R2'].values
array3=df['R3'].values
array4=df['Rmax1'].values
array5=df['Rmax2'].values
array6=df['Rmax3'].values
array7=delta_R1.values
array8=delta_R2.values
array9=delta_R3.values
array10=res1.values
array11=res2.values
array12=res3.values
array13=res1_percent.values
array14=res2_percent.values
array15=res3_percent.values
Final=np.transpose(np.vstack((array1,array2,array3,array4,array5,array6,array7,array8,array9,array10,array11,array12,array13,array14,array15)))
df2=pd.DataFrame(Final,columns=['R1','R2','R3','Rmax1','Rmax2','Rmax3','Rmax1-R1','Rmax2-R2','Rmax3-R3','(Rmax1-R1)/Rmax1','(Rmax2-R2)/Rmax2','(Rmax3-R3)/Rmax3','(Rmax1-R1)/Rmax1*100','(Rmax2-R2)/Rmax2*100','(Rmax3-R3)/Rmax3*100'])
#df2=pd.DataFrame(Final)
df2.head()
df2.to_csv('macro_out.csv')
