
import numpy as np
import pandas as pd
import collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

category_json = pd.read_json("C:/Users/manud/Documents/Youtube-Video-Analysis-Classification-and-Prediction-master/Dataset/category_id.JSON")
category_json.head(5)
category_json.columns

category_dict = [{'id': item['id'], 'title': item['snippet']['title']} for item in category_json['items']]
category_dict

categories = pd.read_csv("C:/Users/manud/Documents/Youtube-Video-Analysis-Classification-and-Prediction-master/Dataset/categories.csv", header=0)
categories.head(20)

categories['Outcome'] = categories['Category'].str.contains("Non-Educational")

categories['Outcome'].replace({True:0,False:1}, inplace=True)

new_video = pd.read_csv("C:/Users/manud/Documents/Youtube-Video-Analysis-Classification-and-Prediction-master/Dataset/mydata.csv", header=0)
new_video.head(5)

new_video['Outcome'] = (new_video['Category_ID'] == 27) | (new_video['Category_ID'] == 28)

new_video['Outcome'].replace({True:1, False:0}, inplace=True)

new_video['Outcome'].value_counts()

new_video

new_video.loc[2316,'Title']

from sklearn.feature_extraction.text import TfidfVectorizer
vector = TfidfVectorizer(min_df=1)
counts = vector.fit_transform(new_video['Title'].values)

vector.get_feature_names_out()

NB_Model = MultinomialNB()
RFC_Model = RandomForestClassifier()
SVC_Model = SVC()
KNC_Model = KNeighborsClassifier()
DTC_Model = DecisionTreeClassifier()

output = new_video['Category_ID'].values

NB_Model.fit(counts,output)

RFC_Model.fit(counts,output)

SVC_Model.fit(counts,output)

KNC_Model.fit(counts,output)

DTC_Model.fit(counts,output)

X = counts
Y = new_video['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

NBtest = MultinomialNB().fit(X_train,Y_train)
nb_predictions = NBtest.predict(X_test)
acc_nb = NBtest.score(X_test, Y_test)
print('The Naive Bayes Algorithm has an accuracy of', acc_nb)

RFCtest = RandomForestClassifier().fit(X_train,Y_train)
rfc_predictions = RFCtest.predict(X_test)
acc_rfc = RFCtest.score(X_test, Y_test)
print('The Random Forest Algorithm has an accuracy of', acc_rfc)

SVCtest = SVC().fit(X_train,Y_train)
svc_predictions = SVCtest.predict(X_test)
acc_svc = SVCtest.score(X_test, Y_test)
print('The Support Vector Algorithm has an accuracy of', acc_svc)

KNCtest = KNeighborsClassifier().fit(X_train,Y_train)
knc_predictions = KNCtest.predict(X_test)
acc_knc = KNCtest.score(X_test, Y_test)
print('The K Neighbors Algorithm has an accuracy of', acc_knc)

DTCtest = DecisionTreeClassifier().fit(X_train,Y_train)
dtc_predictions = DTCtest.predict(X_test)
acc_dtc = DTCtest.score(X_test, Y_test)
print('The Decision Tree Algorithm has an accuracy of', acc_dtc)

xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, Y_train)
print('The XGBoost has an accuracy of', xgb_clf.score(X_test, Y_test))

Titles = ["The Riddle That Seems Impossible Even If You Know The Answer"]

Titles_counts = vector.transform(Titles)

PredictNB = NB_Model.predict(Titles_counts)
PredictNB

PredictRFC = RFC_Model.predict(Titles_counts)
PredictRFC

PredictSVC = SVC_Model.predict(Titles_counts)
PredictSVC

PredictKNC = KNC_Model.predict(Titles_counts)
PredictKNC

PredictDTC = DTC_Model.predict(Titles_counts)
PredictDTC

CategoryNamesListNB = []
for Category_ID in PredictNB:
    MatchingCategoriesNB = [x for x in category_dict if x["id"] == str(Category_ID)]
    if MatchingCategoriesNB:
        CategoryNamesListNB.append(MatchingCategoriesNB[0]["title"])

CategoryNamesListRFC = []
for Category_ID in PredictRFC:
    MatchingCategoriesRFC = [x for x in category_dict if x["id"] == str(Category_ID)]
    if MatchingCategoriesRFC:
        CategoryNamesListRFC.append(MatchingCategoriesRFC[0]["title"])

CategoryNamesListSVC = []
for Category_ID in PredictSVC:
    MatchingCategoriesSVC = [x for x in category_dict if x["id"] == str(Category_ID)]
    if MatchingCategoriesSVC:
        CategoryNamesListSVC.append(MatchingCategoriesSVC[0]["title"])

CategoryNamesListKNC = []
for Category_ID in PredictKNC:
    MatchingCategoriesKNC = [x for x in category_dict if x["id"] == str(Category_ID)]
    if MatchingCategoriesKNC:
        CategoryNamesListKNC.append(MatchingCategoriesKNC[0]["title"])

CategoryNamesListDTC = []
for Category_ID in PredictDTC:
    MatchingCategoriesDTC = [x for x in category_dict if x["id"] == str(Category_ID)]
    if MatchingCategoriesDTC:
        CategoryNamesListDTC.append(MatchingCategoriesDTC[0]["title"])

TitleDataFrameNB = []
for i in range(0, len(Titles)):
    TitleToCategoriesNB = {'Title': Titles[i],  'Category': CategoryNamesListNB[i]}
    TitleDataFrameNB.append(TitleToCategoriesNB)

TitleDataFrameRFC = []
for i in range(0, len(Titles)):
    TitleToCategoriesRFC = {'Title': Titles[i],  'Category': CategoryNamesListRFC[i]}
    TitleDataFrameRFC.append(TitleToCategoriesRFC)

TitleDataFrameSVC = []
for i in range(0, len(Titles)):
    TitleToCategoriesSVC = {'Title': Titles[i],  'Category': CategoryNamesListSVC[i]}
    TitleDataFrameSVC.append(TitleToCategoriesSVC)

TitleDataFrameKNC = []
for i in range(0, len(Titles)):
    TitleToCategoriesKNC = {'Title': Titles[i],  'Category': CategoryNamesListKNC[i]}
    TitleDataFrameKNC.append(TitleToCategoriesKNC)

TitleDataFrameDTC = []
for i in range(0, len(Titles)):
    TitleToCategoriesDTC = {'Title': Titles[i],  'Category': CategoryNamesListDTC[i]}
    TitleDataFrameDTC.append(TitleToCategoriesDTC)

PredictDFnb = pd.DataFrame(PredictNB)
TitleDFnb = pd.DataFrame(TitleDataFrameNB)
PreFinalDFnb = pd.concat([PredictDFnb, TitleDFnb], axis=1)
PreFinalDFnb.columns = (['Categ_ID', 'Predicted Category', 'Hypothetical Video Title'])
FinalDFnb = PreFinalDFnb.drop(['Categ_ID'],axis=1)
colsNB = FinalDFnb.columns.tolist()
colsNB = colsNB[-1:] + colsNB[:-1]
FinalDFnb= FinalDFnb[colsNB]

PredictDFrfc = pd.DataFrame(PredictRFC)
TitleDFrfc = pd.DataFrame(TitleDataFrameRFC)
PreFinalDFrfc = pd.concat([PredictDFrfc, TitleDFrfc], axis=1)
PreFinalDFrfc.columns = (['Categ_ID', 'Predicted Category', 'Hypothetical Video Title'])
FinalDFrfc = PreFinalDFrfc.drop(['Categ_ID'],axis=1)
colsRFC = FinalDFrfc.columns.tolist()
colsRFC = colsRFC[-1:] + colsRFC[:-1]
FinalDFrfc= FinalDFrfc[colsRFC]

PredictDFsvc = pd.DataFrame(PredictSVC)
TitleDFsvc = pd.DataFrame(TitleDataFrameSVC)
PreFinalDFsvc = pd.concat([PredictDFsvc, TitleDFsvc], axis=1)
PreFinalDFsvc.columns = (['Categ_ID', 'Predicted Category', 'Hypothetical Video Title'])
FinalDFsvc = PreFinalDFsvc.drop(['Categ_ID'],axis=1)
colsSVC = FinalDFsvc.columns.tolist()
colsSVC = colsSVC[-1:] + colsSVC[:-1]
FinalDFsvc= FinalDFsvc[colsSVC]

PredictDFknc = pd.DataFrame(PredictKNC)
TitleDFknc = pd.DataFrame(TitleDataFrameKNC)
PreFinalDFknc = pd.concat([PredictDFknc, TitleDFknc], axis=1)
PreFinalDFknc.columns = (['Categ_ID', 'Predicted Category', 'Hypothetical Video Title'])
FinalDFknc = PreFinalDFknc.drop(['Categ_ID'],axis=1)
colsKNC = FinalDFknc.columns.tolist()
colsKNC = colsKNC[-1:] + colsKNC[:-1]
FinalDFknc= FinalDFknc[colsKNC]

PredictDFdtc = pd.DataFrame(PredictDTC)
TitleDFdtc = pd.DataFrame(TitleDataFrameDTC)
PreFinalDFdtc = pd.concat([PredictDFdtc, TitleDFdtc], axis=1)
PreFinalDFdtc.columns = (['Categ_ID', 'Predicted Category', 'Hypothetical Video Title'])
FinalDFdtc = PreFinalDFdtc.drop(['Categ_ID'],axis=1)
colsDTC = FinalDFdtc.columns.tolist()
colsDTC = colsDTC[-1:] + colsDTC[:-1]
FinalDFdtc= FinalDFdtc[colsDTC]

import pickle

vec_file = 'vectorizer.pickle'
pickle.dump(vector, open(vec_file, 'wb'))

pickle.dump(xgb_clf, open('premodel.model', 'wb'))

vectorizer = pickle.load(open('C:/Users/manud/Documents/MyYtServer/models/vectorizer.pickle', 'rb'))
model = pickle.load(open('C:/Users/manud/Documents/MyYtServer/models/premodel.model', 'rb'))

Titles = ["Why this kolaveri di  ","LIFE IS TOUGH (Official Video) - Guru Mann Hindi Song | Cinematic Bollywood || Rubbal GTR"
         ,"The Riddle That Seems Impossible Even If You Know The Answer","How Electricity Actually Works",
         "Future Computers Will Be Radically Different (Analog Computing)","The Man Who Accidentally Killed The Most People In History",
         "Why can't you go faster than light?","One of the most counterintuitive facts of our universe is that you can’t go faster than the speed of light.  From this single observation arise all of the mind-bending behaviors of special relativity.  But why is this so?  In this in-depth video, Fermilab’s Dr. Don Lincoln explains the real reason that you can’t go faster than the speed of light.  It will blow your mind"]
print(model.predict(vectorizer.transform(Titles)))
