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
missing_percentage = data.isna().mean() * 100
print("Percentage of missing values in each column:")
sorted_missing_percentage = missing_percentage.sort_values(ascending=False)
data['target'] = np.where(data['status']=='legitimate',1,0)
data1 = data.copy()
data1.drop(['url','status'],axis=1,inplace=True)
print(data1.columns)
correlation_matrix=pd.DataFrame(data1.corr()[['target']]).sort_values(by='target',ascending=False).fillna(0).reset_index().rename(columns={'index':'Features','target':'Correlation_Score'})
filtered=correlation_matrix[correlation_matrix['Correlation_Score']!=0]
positive=filtered[filtered['Correlation_Score']>0.1]['Features'].tolist()
negative=filtered[filtered['Correlation_Score']<-0.2]['Features'].tolist()
data2=pd.concat([data1[positive],data1[negative]],axis=1)
numeric_columns = data2.select_dtypes(include=['number']).columns.tolist()
X = data1.copy()
Y = X.pop('target')
def model_all_features(modelname):
    kf = KFold(n_splits=5)
    if modelname=='Logistic':
        print("Logistic Regression Model")
        model = LogisticRegression(max_iter=10000)
    elif modelname == 'RF':
        print("Random Forest Model")
        model = RandomForestClassifier()
    elif modelname == 'ExtraTrees':
        print("Extra Trees Model")
        model = ExtraTreeClassifier()
    elif modelname == 'AdaBoost':
        print("Adaboost Model")
        model = AdaBoostClassifier()
    elif modelname == 'SVM':
        print("SVM Model")
        model = SVC(kernel='rbf')
    elif modelname == 'Scaled SVM':
        print("Scaled SVM Model")
        model = SVC(kernel='rbf')

    accuracies,precision,recall,f1score = [],[],[],[]
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        if modelname == 'Scaled SVM':
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

    accuracies.append(accuracy_score(y_test, predictions))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))
    f1score.append(f1_score(y_test, predictions, average='macro'))
    average_accuracy = np.mean(accuracies)
    print("Average Accuracy of "+modelname +" Model", average_accuracy)
    average_precision = np.mean(precision)
    print("Average Precision of "+modelname +" Model", average_precision)
    average_recall = np.mean(recall)
    print("Average Recall of "+modelname +" Model", average_recall)
    average_f1score = np.mean(f1score)
    print("Average F1 Score of "+modelname +" Model", average_f1score)

model_all_features("Logistic")
model_all_features("RF")
model_all_features('ExtraTrees')
model_all_features('AdaBoost')
model_all_features('SVM')
model_all_features('Scaled SVM')
X = data1.copy()
Y = X.pop('target')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train,y_train)

predictions = model.predict(X_test)
print(confusion_matrix(y_test,predictions))

print("Precision of Random Forest Model ",precision_score(y_test,predictions))
print("Recall of Random Forest Model ",recall_score(y_test,predictions))
print("F1 Score of Random Forest Model ",f1_score(y_test,predictions))
print("Accuracy of Random Forest Model ",accuracy_score(y_test,predictions))
importances = model.feature_importances_
feature_importances = zip(X_train.columns, importances)
sorted_feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
df_feature_importances = pd.DataFrame(sorted_feature_importances, columns=['Feature', 'Importance'])
X = data2.copy()
Y = X.pop('target')
def model(modelname):
    kf = KFold(n_splits=5)
    if modelname=='Logistic':
        print("Logistic Regression Model")
        model = LogisticRegression(max_iter=10000)
    elif modelname == 'RF':
        print("Random Forest Model")
        model = RandomForestClassifier()
    elif modelname == 'ExtraTrees':
        print("Extra Trees Model")
        model = ExtraTreeClassifier()
    elif modelname == 'AdaBoost':
        print("Adaboost Model")
        model = AdaBoostClassifier()
    elif modelname == 'SVM':
        print("SVM Model")
        model = SVC(kernel='rbf')
    elif modelname == 'Scaled SVM':
        print("Scaled SVM Model")
        model = SVC(kernel='rbf')

    accuracies,precision,recall,f1score = [],[],[],[]
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        if modelname == 'Scaled SVM':
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

    accuracies.append(accuracy_score(y_test, predictions))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))
    f1score.append(f1_score(y_test, predictions, average='macro'))
    average_accuracy = np.mean(accuracies)
    print("Average Accuracy of "+modelname +" Model", average_accuracy)
    average_precision = np.mean(precision)
    print("Average Precision of "+modelname +" Model", average_precision)
    average_recall = np.mean(recall)
    print("Average Recall of "+modelname +" Model", average_recall)
    average_f1score = np.mean(f1score)
    print("Average F1 Score of "+modelname +" Model", average_f1score)

model("Logistic")
model("RF")
model('ExtraTrees')
model('AdaBoost')
model('SVM')
model('Scaled SVM')

X = data2.copy()
Y = X.pop('target')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train,y_train)

predictions = model.predict(X_test)
print(confusion_matrix(y_test,predictions))

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