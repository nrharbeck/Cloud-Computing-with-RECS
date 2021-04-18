import pandas as pd
import numpy as np

#Data comes from the U.S. Energy Information Administration
df = pd.read_csv("https://www.eia.gov/consumption/residential/data/2015/csv/recs2015_public_v4.csv")

#KWH is probably the most useful target
Target = 'KWH'

#First, all imputation code columns are removed
df = df.loc[:, ~df.columns.str.startswith('Z')]

Census_Dict = {1:'New England', 2:'Middle Atlantic', 3:'East North Central', 4:'West North Central', 5:'South Atlantic', 6:'East South Central',7:'West South Central', 8:'Mountain North', 9:'Mountain South', 10:'Pacific'}

#Specify the list of housing features for model building. RECS Code information from https://www.eia.gov/consumption/residential/data/2015/xls/codebook_publicv4.xlsx
X_Features = ['REGIONC','TYPEHUQ','DISHWASH', 'CWASHER', 'DRYER', 'AIRCOND','NUMBERAC', 'NUMCFAN','NUMFLOORFAN','NUMWHOLEFAN','NUMATTICFAN','NOTMOIST','FUELH2O','USEEL','ELWARM','ELCOOL','ELWATER','ELFOOD','ELOTHER','TOTCSQFT','TOTHSQFT','TOTSQFT_EN','HEATHOME','TVCOLOR']

X = df[X_Features]

#Now replace an instance of negative numbers with 0 (i.e. no count of a feature)
X = X.clip(lower=0)
X = X.replace({"REGIONC": Census_Dict})

#Encode region features
X = pd.get_dummies(X)
y = df['KWH']

#Split data into train and test sets. Validation with the training set will be incorporated into the pipeline below
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Model building with Keras MLP
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
# determine the number of input features
n_features = X_train.shape[1]
# define model
model = Sequential()
model.add(Dense(70, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu'))
# compile the model
model.compile(optimizer='adam', loss='MeanSquaredError')
# fit the model
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)

#Guide to Save the Model as js from https://blog.tensorflow.org/2018/07/train-model-in-tfkeras-with-colab-and-run-in-browser-tensorflowjs.html
model.save('RECS_Model.h5')

!pip install tensorflowjs 
!mkdir model
!tensorflowjs_converter --input_format keras RECS_Model.h5 model/
!zip -r model.zip model 