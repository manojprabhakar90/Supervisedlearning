import pandas as pd # type: ignore 
import numpy as np# type: ignore 
import os
from sklearn.linear_model import LogisticRegression # type: ignore 
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier# type: ignore 
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier# type: ignore 
from sklearn.model_selection import KFold# type: ignore 
from sklearn.metrics import accuracy_score,precision_score, recall_score,f1_score,confusion_matrix# type: ignore 
from sklearn.svm import SVC# type: ignore 
from sklearn.preprocessing import StandardScaler# type: ignore 
from sklearn.model_selection import train_test_split# type: ignore 
import joblib# type: ignore 
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("./data/dataset_phishing.csv")
data['target'] = np.where(data['status']=='legitimate',1,0)
data1 = data.copy()
data1.drop(['url','status'],axis=1,inplace=True)
correlation_matrix=pd.DataFrame(data1.corr()[['target']]).sort_values(by='target',ascending=False).fillna(0).reset_index().rename(columns={'index':'Features','target':'Correlation_Score'})
filtered=correlation_matrix[correlation_matrix['Correlation_Score']!=0]
positive=filtered[filtered['Correlation_Score']>0.1]['Features'].tolist()
negative=filtered[filtered['Correlation_Score']<-0.2]['Features'].tolist()
data2=pd.concat([data1[positive],data1[negative]],axis=1)
data2 = data2[['target', 'google_index','page_rank', 'nb_hyperlinks','nb_www', 'domain_age']]
X = data2.copy()
Y = X.pop('target')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train,y_train)

predictions = model.predict(X_test)

print("Precision of Random Forest Model ",precision_score(y_test,predictions))
print("Recall of Random Forest Model ",recall_score(y_test,predictions))
print("F1 Score of Random Forest Model ",f1_score(y_test,predictions))
print("Accuracy of Random Forest Model ",accuracy_score(y_test,predictions))
importances = model.feature_importances_
feature_importances = zip(X_train.columns, importances)
sorted_feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
df_feature_importances = pd.DataFrame(sorted_feature_importances, columns=['Feature', 'Importance'])

joblib.dump(model, 'RandomForestModel.pkl')
print("The Random Forest Model has been saved ")