import pandas as pd
import numpy as np

X_full = pd.read_csv("NPK.csv", index_col= "SOILID")


conditions = [
    (X_full['Nitrogen'] <= 240),
    (X_full['Nitrogen'] > 240) & (X_full['Nitrogen'] <= 480),
    (X_full['Nitrogen'] > 480)]
choices = ['low', 'medium', 'high']

X_full['N_level'] = np.select(conditions, choices, default='null')


conditions = [
    (X_full['Phosphorus'] <= 11),
    (X_full['Phosphorus'] > 11) & (X_full['Phosphorus'] <= 22),
    (X_full['Phosphorus'] > 22)]
choices = ['low', 'medium', 'high']

X_full['P_level'] = np.select(conditions, choices, default='null')

conditions = [
    (X_full['Potassium'] <= 110),
    (X_full['Potassium'] > 110) & (X_full['Potassium'] <= 280),
    (X_full['Potassium'] > 280)]
choices = ['low', 'medium', 'high']

X_full['K_level'] = np.select(conditions, choices, default='null')


conditions = [
    (X_full['N_level'] == 'low'),
    (X_full['N_level'] == 'medium'),
    (X_full['N_level'] == 'high')]
choices = ['lots-of-nitrogen-fertilizer', 'small-amount-of-nitrogen-fertilizer', 'suitable-nitrogen']

X_full['suggested_N_fertilizer'] = np.select(conditions, choices, default='null')

conditions = [
    (X_full['P_level'] == 'low'),
    (X_full['P_level'] == 'medium'),
    (X_full['P_level'] == 'high')]
choices = ['lots-of-phosphorus-fertilizer', 'small-amount-of-phosphorus-fertilizer', 'suitable-phosphorus']
X_full['suggested_P_fertilizer'] = np.select(conditions, choices, default='null')

conditions = [
    (X_full['K_level'] == 'low'),
    (X_full['K_level'] == 'medium'),
    (X_full['K_level'] == 'high')]
choices = ['lots-of-potassium-fertilizer', 'small-amount-of-potassium-fertilizer', 'suitable-potassium']

X_full['suggested_K_fertilizer'] = np.select(conditions, choices, default='null')

X_full['suggestion'] = X_full[['suggested_N_fertilizer', 'suggested_P_fertilizer', 'suggested_K_fertilizer']].agg(', '.join, axis =1 )


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))

plt.title("Nitrogen levels for soil dataset")

sns.barplot(y=X_full['Nitrogen'], x=X_full['N_level'])

plt.ylabel("Nitrogen value(in kg/ha)")

plt.show()



plt.figure(figsize=(10,6))

plt.title("Phosphorus levels for soil dataset")

sns.barplot(y=X_full['Phosphorus'], x=X_full['P_level'])

plt.ylabel("Phosphorus value(in kg/ha)")

plt.show()



plt.figure(figsize=(10,6))

plt.title("Potassium levels for soil dataset")

sns.barplot(y=X_full['Potassium'], x=X_full['K_level'])

plt.ylabel("Potassium value(in kg/ha)")

plt.show()


'''X_full.dropna(axis = 0, subset=['N_level'], inplace=True)
X_full.drop(['N_level'], axis=1, inplace=True)

X_full.dropna(axis = 0, subset=['P_level'], inplace=True)
X_full.drop(['P_level'], axis=1, inplace=True)

X_full.dropna(axis = 0, subset=['K_level'], inplace=True)
X_full.drop(['K_level'], axis=1, inplace=True)'''

X_full.dropna(axis = 0, subset=['suggested_N_fertilizer'], inplace=True)
X_full.drop(['suggested_N_fertilizer'], axis=1, inplace=True)

X_full.dropna(axis = 0, subset=['suggested_P_fertilizer'], inplace=True)
X_full.drop(['suggested_P_fertilizer'], axis=1, inplace=True)

X_full.dropna(axis = 0, subset=['suggested_K_fertilizer'], inplace=True)
X_full.drop(['suggested_K_fertilizer'], axis=1, inplace=True)


X_full.dropna(axis=0, subset=['suggestion'], inplace=True)
y_new = X_full.suggestion
X_full.drop(['suggestion'], axis=1, inplace = True)

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

y = lb.fit_transform(y_new)

from sklearn.model_selection import train_test_split
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)


categorical_cols = list(X_train_full.columns[X_train_full.dtypes == 'object'])

numerical_cols = list(X_train_full.columns[X_train_full.dtypes == 'int64'])

continous_cols = list(X_train_full.columns[X_train_full.dtypes == 'float64'])

#categorical_cols2 = [cname for cname in y_train_full.columns if y_train_full[cname].dtype == "object"]


my_cols = categorical_cols + numerical_cols + continous_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()



from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

#preprocessing the numerical data
numerical_transformer = SimpleImputer(strategy='constant')

#preprocesssing the categorical data
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))])

continuous_transformer = Pipeline(steps= [('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore')) ])
#bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols), ('cont', continuous_transformer, continous_cols)])


from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.naive_bayes import GaussianNB


model = XGBRegressor(n_estimators=400)

#bundle preprocessing and model in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

from sklearn.model_selection import cross_val_score
cross_val_score(clf, X_full, y, cv = 5, scoring='neg_mean_absolute_error')
#print("Average MAE score:", scores.mean())

#preprocessing of training data, fit model
model = XGBRegressor(objective= 'reg:linear', colsample_bytree= 0.3, learning_rate=0.1, n_estimators=250)
clf.fit(X_train, y_train)

#preprocessing of validation data, get predictions

#save test predictions to file
#output = pd.DataFrame(preds, columns=['suggestions']).to_csv('suggestions.csv')


import pickle
filename = 'xgboost.sav'
pickle.dump(clf, open(filename, 'wb'))


loaded_model = pickle.load(open(filename, 'rb'))

from sklearn.metrics import confusion_matrix
preds = clf.predict(X_valid)


preds = np.around(preds)
preds = preds.astype(int)
print("MAE", mean_absolute_error(preds, y_valid))
preds = lb.inverse_transform(preds)