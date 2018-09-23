import pandas as pd
from sklearn import preprocessing

#import the dataset
# make sure to change the path
# for windows user use \\
df=pd.read_csv("E:\\ML\\DATASET\\Bike-Sharing-Dataset\\day.csv")

#shows the number of rows and columns
df.shape

#show differnt rows
df.columns

#first first ten data
df.head(10)

#function for label encoder
def label_encoder(df,column):
    le=preprocessing.LabelEncoder()
    le.fit_transform(df[column])
    ohe = preprocessing.OneHotEncoder()
    
    temp_array=ohe.fit_transform( df[[column]]).toarray()
    columns_name=[column +"_" + str(m) for m in le.classes_]
    return( pd.DataFrame(temp_array,columns=columns_name))
    
    
categorical_variables=["season","yr","mnth","holiday","weekday","workingday","weathersit"]
numerical_variables =["hum" , "instant","temp","atemp","windspeed"]

new_df=df[numerical_variables]
new_df.head(10)

for columns in categorical_variables:
    new_df=pd.concat([new_df,label_encoder(df,columns)],axis=1)

#just a quick review for data
new_df.head(10)

new_df.columns

#linear regression 
from sklearn.model_selection import train_test_split

X,X_test,y,y_test =train_test_split(new_df,df["cnt"],test_size = 0.3)

from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(X,y)
y_pred = model.predict(X_test)

from sklearn.metrics import r2_score,mean_squared_error

r2_score(y_test,y_pred) # what good the prediction value is
model.score(X,y) # how good the model is fitted with training data

mean_squared_error(y_test,y_pred)#this should compare with different
                                                        # module the low the value the better the model

#mean absolute percentage error
sum(abs(y_test-y_pred)) / sum(y_test) #show how much the error is
