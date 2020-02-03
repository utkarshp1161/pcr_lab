
from flask import Flask, make_response, request, send_file, render_template
import io
import csv
import pandas 
import os
import shutil
import numpy 
app = Flask(__name__)

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")


@app.route('/')
def form():
    return """
        <html>

       

            <body style="background-color:orchid;">
        

            
                <h1 style="color:blue" align="center">Welcome to ONE Lab Data Analysis Platform</h1>
                
                <h2>1. Run 2d pca (Input file format = .csv) </h2>


                <form  action="/pca2d" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" />
                    <input type="submit"  />   
                </form>
                
               <h2>2. Run 3d pca (Input file format = .csv) </h2>
                <form   action="/pca3d" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" />
                    <input type="submit" />
                </form>

                <h2>3. Run Macro (Nitrate-sensor) (Input file format = .csv) </h2>
                <form  action="/Nitrate" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" />
                    <input type="submit" />
                </form>
                <h2>4. Run excel to csv converter (Input file format = .xlsx) </h2>
                <form  action="/xlsx_to_csv" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" />
                    <input type="submit" />
                </form>
                <h2>5. Run csv to excel converter (Input file format = .csv) </h2>
                <form  action="/csv_to_xlsx" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" />
                    <input type="submit" />
                </form>

                <h2>6. Sutripto automated script (Input file format = .csv) </h2>
                <form  action="/sutripto" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" />
                    <input type="number" placeholder="t-value" step="0.0001" min="0" max="50"  name="quantity" >
                    <input type="submit" />
                </form>
        <h2 style="color:yellow" align="center">More stuff coming soon... </h1>
            
            </body>
        </html>
    """

@app.route('/pca2d', methods=["POST"])
def pca_2d():
   # df = request.files['data_file']
    df = pandas.read_csv(request.files.get('data_file'))
    f=df
   # if not f:
   #     return "No file"

   # stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
   # csv_input = csv.reader(stream)
    #print("file contents: ", file_contents)
    #print(type(file_contents))
    #print(csv_input)
    #for row in csv_input:
    #    print(row)

   # stream.seek(0)
   # result = transform(stream.read())

  #  response = make_response(result)
    if df.empty:
        print('No file')
    import pandas as pd
    import numpy as np
    import os
    #import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    #from mpl_toolkits.mplot3d import Axes3D


    #df=f
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
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1','principal component 2'])

    x=principalDf.values
    print(pca.explained_variance_ratio_)

    final=np.column_stack((x,y))
    final_df=pd.DataFrame(final,columns=['pc1','pc2','labels'])
    print(names)
    print(names_numeric)

    import plotly.express as px
    import plotly
    fig = px.scatter(final_df, x='pc1', y='pc2',
                  color='labels')
    plotly.offline.plot(fig, "2d.html")

    #path= "temp-plot.html"
    #os.remove("templates/plot.html")
    import shutil
    shutil.move("temp-plot.html", "templates/plot.html")

    #return send_file(path, as_attachment=True)
    return render_template('plot.html')
   

@app.route('/pca3d', methods=["POST"])
def pca_3d():
   # df = request.files['data_file']
    df = pandas.read_csv(request.files.get('data_file'))
    f=df
    if df.empty:
        print('No file')
    import pandas as pd
    import numpy as np
    #import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
   # from mpl_toolkits.mplot3d import Axes3D


    #df=pd.read_csv('sample.csv')
   # print(df.head())

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

    #path="temp-plot.html"
    #return send_file(path, as_attachment=True)
    import shutil
    shutil.move("temp-plot.html", "templates/plot.html")

    #return send_file(path, as_attachment=True)
    return render_template('plot.html')



#@app.route('/download')
#def download_file():
#	#path = "html2pdf.pdf"
#	#path = "info.xlsx"
#	path = "temp-plot.html"
#	#path = "sample.txt"
#	return send_file(path, as_attachment=True)

@app.route('/Nitrate', methods=["POST"])
def nitrate():
    df = pandas.read_csv(request.files.get('data_file'))
    f=df
    if df.empty:
        return ("No file")
    import pandas as pd 
    import numpy as np
    #import matplotlib.pyplot as plt

    #READ SHEET
    #import sys
    #file_to_open=sys.argv[1] # 0 is the index of name of python prog, 1 is the index of first argument in command line.
    #df=pd.read_csv('macro_input.csv')
    #df=pd.read_csv(file_to_open,sep=',')



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
    df2.to_csv('macro_out.csv',index=False)
    #df2.to_html('macro.html')
    #shutil.move("macro.html", "templates/macro.html")

    #return send_file(path, as_attachment=True)
    #return render_template('macro.html')

    
    path="macro_out.csv"
    return send_file(path, as_attachment=True)


@app.route('/xlsx_to_csv', methods=["POST"])
def xlsxtocsv():
    df = pandas.read_excel(request.files.get('data_file'))
    f=df
    if df.empty:
        return ("No file")
    df.to_csv('xlsx_to_csv.csv',index=False)
    path = "xlsx_to_csv.csv"
    return send_file(path, as_attachment=True)

@app.route('/csv_to_xlsx', methods=["POST"])
def csvtoxlsx():
    df = pandas.read_csv(request.files.get('data_file'))
    f=df
    if df.empty:
        return ("No file")
    df.to_excel('csv_to_xlsx.xlsx',index=False)
    path = "csv_to_xlsx.xlsx"
    return send_file(path, as_attachment=True)



@app.route('/sutripto', methods=["POST"])
def sut():
    import pandas as pd
    import numpy as np
    df = pandas.read_csv(request.files.get('data_file'))
    f=df
    t= request.form.get('quantity')
    t= float(t)
    if df.empty:
        return ("No file")
    freq=df['f']
    eps=df['e1']
    epsdp=df['e2']
    #eps = 15 ;
    #epsdp = 0.6;
    #epsr = eps-epsdp;
    epsr=eps.subtract(epsdp)


    #miu = xlsread(f,'D2:D202');
    miu0 = 1.26e-6
    miu = 1
    miudp = 0
    #t = 0.001
    




    #%paramater - Obtain x+iy form from r,theta form raw data
    #%miu   = xlsread(f,'D2:D439');
    #%miudp = xlsread(f,'E2:E439');
    #miur  = miu-1i.*miudp;
    miur = miu - miudp





    #% S-parameters
    #%S11 = xlsread(f,'K2:K202');
    pi=np.pi
    w = 2*pi*freq
    eps0 = 8.854187817e-12
    miu0 = 1.26e-6
    sigma = w*eps0*epsdp;
    Z0 = 377
    c = 3e8
    #delta = np.sqrt(2/abs((miu*miu0*w*sigma)))


    #%SEa = 8.68.*t./delta;
    #SEa= 8.68*t*np.sqrt(abs(pi*miu*miu0*freq*sigma))


    Zin = Z0*np.sqrt(miur/epsr)*np.tanh((2*pi*freq*t/c)*np.sqrt(miur*epsr))
    X = (Zin-Z0)/(Zin+Z0)
    #%modX = abs(X)
    RLoss = 20*np.log(abs(X))
    Z = abs(np.sqrt(miur/epsr)*np.sqrt(miu0/eps0))
    #%RLoss = 20.*log(abs(S11));


    D = (1.41421356237*pi*freq)/c
    alpha = D*np.sqrt((miudp*epsdp-miu*eps)+np.sqrt((miudp*epsdp-miu*eps)**2+(miu*epsdp+miudp*eps)**2))

    deltae = np.arctan(epsdp/eps)
    deltam = np.arctan(miudp/miu)


    K = 4*pi*(np.sqrt(miu*eps))*np.sin((deltae+deltam)/2)/(c*np.cos(deltae)*np.cos(deltam))
    M2 = (miu*np.cos(deltae)-eps*np.cos(deltam))**2+np.tan((deltam-deltae)/2)*np.tan((deltam-deltae)/2)*(miu*np.cos(deltae)+eps*np.cos(deltam))**2
    M2f = 1/M2
    M = (4*miu*np.cos(deltae)*eps*np.cos(deltam))*M2f
    P = (np.sinh(K*freq*t)*np.sinh(K*freq*t))-M
    EMD = abs(P)



    freq=freq.values
    EMD=EMD.values
    alpha=alpha.values
    RLoss=RLoss.values


    Final=np.transpose(np.vstack((freq,EMD,alpha,RLoss)))
    df=pd.DataFrame(Final,columns=['freq','EMD','alpha','Rloss'])
   # df.to_csv('sutripto_out.csv',index=False)

    df.to_csv('sutripto_out.csv',index=False)
    path = 'sutripto_out.csv'
    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    #app.run(host='0.0.0.0', debug=True, port=5000)
    app.run(host='0.0.0.0',debug='true')
