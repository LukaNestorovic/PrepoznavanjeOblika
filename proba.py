# %% podaci
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn import metrics

pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_columns', None)

df = pd.read_csv('ShanghaiPM20100101_20151231.csv')

print(df.head())
print(df.shape)
print(df.dtypes)
print(df.info())
print(df.isnull().sum() / df.shape[0] * 100)

# %% nesto

df.drop(['PM_Jingan'], axis=1, inplace=True)
df.drop(['PM_Xuhui'], axis=1, inplace=True)
print(df.isnull().sum() / df.shape[0] * 100)

df['DEWP'] = df['DEWP'].replace(-97, np.nan)

df['DEWP'].fillna(method='bfill', inplace=True)
df['HUMI'].fillna(method='bfill', inplace=True)
df['PRES'].fillna(np.nanmean(df['PRES']), inplace=True)
df['TEMP'].fillna(method='bfill', inplace=True)
df['cbwd'].fillna(method='bfill', inplace=True)
df['Iws'].fillna(np.nanmean(df['Iws']), inplace=True)
df.dropna(axis=0, inplace=True)
print(df.isnull().sum() / df.shape[0] * 100)
print(df.describe())

print(df['PM_US Post'].describe())

# %%

df_season = df.set_index('season')
zagadjenost = df_season.loc[1, 'PM_US Post']
sb.distplot(zagadjenost, fit=stats.norm)
plt.xlabel('Prolece cestice')
plt.ylabel('Verovatnoća')
plt.figure(5)
zagadjenost = df_season.loc[2, 'PM_US Post']
sb.distplot(zagadjenost, fit=stats.norm)
plt.xlabel('Leto cestice')
plt.ylabel('Verovatnoća')
plt.figure(6)
zagadjenost = df_season.loc[3, 'PM_US Post']
sb.distplot(zagadjenost, fit=stats.norm)
plt.xlabel('Jesen cestice')
plt.ylabel('Verovatnoća')
plt.figure(7)
zagadjenost = df_season.loc[4, 'PM_US Post']
sb.distplot(zagadjenost, fit=stats.norm)
plt.xlabel('Zima cestice')
plt.ylabel('Verovatnoća')
plt.figure(8)
df_year = df.set_index('year')
godine = df_year.loc[2012]['TEMP']
sb.distplot(godine, fit=stats.norm)
plt.figure(9)
df_year = df.set_index('year')
godine = df_year.loc[2013]['TEMP']
sb.distplot(godine, fit=stats.norm)
plt.figure(10)
df_year = df.set_index('year')
godine = df_year.loc[2014]['TEMP']
sb.distplot(godine, fit=stats.norm)
plt.figure(11)
df_year = df.set_index('year')
godine = df_year.loc[2015]['TEMP']
sb.distplot(godine, fit=stats.norm)

# %%

nesto = df['month'].unique()
nesto.sort()
gb = df.groupby(by=['month']).mean()
print(gb['PM_US Post'])
plt.figure(12)
plt.plot(nesto, gb['PM_US Post'])
#plt.plot(np.arange(1,13,1), gb['PM_US Post'])

gb_year = df.groupby(by=['year']).mean()
print(gb_year['PM_US Post'])

Iprec = df['Iprec'].unique()
Iprec.sort()
gb_Iprec = df.groupby(by=['Iprec']).mean()
print(gb_Iprec['PM_US Post'])
plt.figure(13)
#plt.plot(Iprec, gb_Iprec['PM_US Post'])
plt.plot(np.arange(1,483,1), gb_Iprec['PM_US Post'])

gb_humi = df.groupby(by=['HUMI']).mean()
print(gb_humi['PM_US Post'])
plt.figure(14)
plt.plot(np.arange(1,860,1), gb_humi['PM_US Post'])

gb_iws = df.groupby(by=['Iws']).mean()
print(gb_iws['PM_US Post'])
plt.figure(15)
plt.plot(np.arange(1,510,1), gb_iws['PM_US Post'])

# %%

corr_mat = df.corr()
sb.heatmap(corr_mat, annot=True)
plt.show()

print(df)

print(df)
print(df.isnull().sum() / df.shape[0] * 100)
print(df.describe())

#df['season'] = df['season'].replace(1,'prolece')
#df['season'] = df['season'].replace(2,'leto')
#df['season'] = df['season'].replace(3,'jesen')
#df['season'] = df['season'].replace(4,'zima')
#print(df)

print(df_season.head())
plt.figure(1)
plt.hist(df_season.loc[1, 'PM_US Post'], density=True, alpha=0.5, bins=50, label="prolece")
plt.hist(df_season.loc[2, 'PM_US Post'], density=True, alpha=0.5, bins=50, label="leto")
plt.hist(df_season.loc[3, 'PM_US Post'], density=True, alpha=0.5, bins=50, label="jesen")
plt.hist(df_season.loc[4, 'PM_US Post'], density=True, alpha=0.5, bins=50, label="zima")
plt.legend()

plt.figure(2)
gb = df.groupby(by=['month']).mean()
#plt.plot(np.arange(1, 13, 1), gb['PM_US Post'], 'b', label='Evropa')
plt.hist(gb.loc[1, 'PM_US Post'],  alpha=1, bins=10)
plt.hist(gb.loc[2, 'PM_US Post'],  alpha=0.5, bins=10)
plt.hist(gb.loc[3, 'PM_US Post'],  alpha=0.5, bins=10)
plt.hist(gb.loc[4, 'PM_US Post'],  alpha=0.5, bins=10)
plt.hist(gb.loc[5, 'PM_US Post'],  alpha=0.5, bins=10)
plt.hist(gb.loc[6, 'PM_US Post'],  alpha=0.5, bins=10)
plt.hist(gb.loc[7, 'PM_US Post'],  alpha=0.5, bins=10)
plt.hist(gb.loc[8, 'PM_US Post'],  alpha=0.5, bins=10)
plt.hist(gb.loc[9, 'PM_US Post'],  alpha=0.5, bins=10)
plt.hist(gb.loc[10, 'PM_US Post'],  alpha=0.5, bins=10)
plt.hist(gb.loc[11, 'PM_US Post'],  alpha=0.5, bins=10)
plt.hist(gb.loc[12, 'PM_US Post'],  alpha=0.5, bins=10)

df_month = df.set_index('month')
nesto = gb.loc[1, 'PM_US Post']

plt.figure(3)
plt.boxplot([df_month.loc[1,'PM_US Post'], df_month.loc[2,'PM_US Post'], df_month.loc[3,'PM_US Post'], df_month.loc[4,'PM_US Post'], df_month.loc[5,'PM_US Post'], df_month.loc[6,'PM_US Post'], df_month.loc[7,'PM_US Post'], df_month.loc[8,'PM_US Post'], df_month.loc[9,'PM_US Post'], df_month.loc[10,'PM_US Post'], df_month.loc[11,'PM_US Post'], df_month.loc[12,'PM_US Post']]) 
plt.ylabel('PM')
plt.xlabel('Mesec')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12], ["Jan", "Feb", "Mart", "Apr", "Maj", "Jun", "Jul", "Avg", "Sep", "Okt", "Nov", "Dec"])
plt.grid()
#df.drop(['cbwd'], axis=1, inplace=True)
df = df[df.cbwd != 'cv']
df_dummy = pd.get_dummies(df['cbwd'], prefix='wdir')
df = pd.concat([df, df_dummy], axis=1)
df.drop(['cbwd'], axis=1, inplace=True)

# %%

df1 = df
df2 = df

def model_evaluation(y, y_predicted, N, d):
    mse = mean_squared_error(y_test, y_predicted) # np.mean((y_test-y_predicted)**2)
    mae = mean_absolute_error(y_test, y_predicted) # np.mean(np.abs(y_test-y_predicted))
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_predicted)
    r2_adj = 1-(1-r2)*(N-1)/(N-d-1)

    # printing values
    print('Mean squared error: ', mse)
    print('Mean absolute error: ', mae)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)
    print('R2 adjusted score: ', r2_adj)
    
    # Uporedni prikaz nekoliko pravih i predvidjenih vrednosti
    res=pd.concat([pd.DataFrame(y.values), pd.DataFrame(y_predicted)], axis=1)
    res.columns = ['y', 'y_pred']
    print(res.head(20))

x = df.drop(['PM_US Post'], axis=1).copy()
y = df['PM_US Post'].copy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15,random_state=42)
x_train1, x_val, y_train1, y_val = train_test_split(x_train, y_train, test_size=0.15,random_state=42)

# Osnovni oblik linearne regresije sa hipotezom y=b0+b1x1+b2x2+...+bnxn
# Inicijalizacija
first_regression_model = LinearRegression(fit_intercept=True)

# Obuka
first_regression_model.fit(x_train, y_train)

# Testiranje
y_predicted = first_regression_model.predict(x_test)

# Evaluacija
print('Osnovni')
model_evaluation(y_test, y_predicted, x_train.shape[0], x_train.shape[1])

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(first_regression_model.coef_)),first_regression_model.coef_)
plt.show()
print("koeficijenti: ", first_regression_model.coef_)

X = sm.add_constant(x_train)

model = sm.OLS(y_train, X.astype('float')).fit()
print(model.summary())

X = sm.add_constant(x_train.drop(['Iws'], axis=1))

model = sm.OLS(y_train, X.astype('float')).fit()
print(model.summary())

numeric_feats = [item for item in x.columns if 'wdir' not in item]
print(numeric_feats)
dummy_feats = [item for item in x.columns if 'wdir' in item]
print(dummy_feats)

# Standardizacija obelezja (svodjenje na sr.vr. 0 i varijansu 1)
scaler = StandardScaler()
scaler.fit(x_train[numeric_feats])

x_train_std = pd.DataFrame(scaler.transform(x_train[numeric_feats]), columns = numeric_feats)
x_test_std = pd.DataFrame(scaler.transform(x_test[numeric_feats]), columns = numeric_feats)

x_train_std = pd.concat([x_train_std, x_train[dummy_feats].reset_index(drop=True)], axis=1)
x_test_std = pd.concat([x_test_std, x_test[dummy_feats].reset_index(drop=True)], axis=1)

x_train_std.head()

# Osnovni oblik linearne regresije sa hipotezom y=b0+b1x1+b2x2+...+bnxn
# Inicijalizacija
regression_model_std = LinearRegression()

# Obuka modela
regression_model_std.fit(x_train_std, y_train)

# Testiranje
y_predicted = regression_model_std.predict(x_test_std)

# Evaluacija
print('Standardizacija')
model_evaluation(y_test, y_predicted, x_train_std.shape[0], x_train_std.shape[1])

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_std.coef_)),regression_model_std.coef_)
plt.show()
print("koeficijenti: ", regression_model_std.coef_)

poly = PolynomialFeatures(interaction_only=True, include_bias=False)
x_inter_train = poly.fit_transform(x_train_std)
x_inter_test = poly.transform(x_test_std)

print(poly.get_feature_names())

# Linearna regresija sa hipotezom y=b0+b1x1+b2x2+...+bnxn+c1x1x2+c2x1x3+...

# Inicijalizacija
regression_model_inter = LinearRegression()

# Obuka modela
regression_model_inter.fit(x_inter_train, y_train)

# Testiranje
y_predicted = regression_model_inter.predict(x_inter_test)

# Evaluacija
print('Linearna')
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])


# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_inter.coef_)),regression_model_inter.coef_)
plt.show()
print("koeficijenti: ", regression_model_inter.coef_)

# RIDGE

# Inicijalizacija
ridge_model = Ridge(alpha=5)

# Obuka modela
ridge_model.fit(x_inter_train, y_train)

# Testiranje
y_predicted = ridge_model.predict(x_inter_test)

# Evaluacija
print('Ridge')
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])


# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(ridge_model.coef_)),ridge_model.coef_)
plt.show()
print("koeficijenti: ", ridge_model.coef_)

# LASSO

# Model initialization
lasso_model = Lasso(alpha=0.01)

# Fit the data(train the model)
lasso_model.fit(x_inter_train, y_train)

# Predict
y_predicted = lasso_model.predict(x_inter_test)

# Evaluation
print('Lasso')
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])


#ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(lasso_model.coef_)),lasso_model.coef_)
plt.show()
print("koeficijenti: ", lasso_model.coef_)

plt.figure(figsize=(10,5))
#plt.plot(regression_model_degree.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'linear',zorder=7) # zorder for ordering the markers
plt.plot(ridge_model.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge') # alpha here is for transparency
plt.plot(lasso_model.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Lasso')
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc='best')
plt.show()

# %%

#KNN

uslovi = [
    (df['PM_US Post'] <= 55.4),
    (df['PM_US Post'] >= 55.5) & (df['PM_US Post'] <= 150.4),
    (df['PM_US Post'] >= 150.5)
    ]
vrednost = [0, 1, 2]
df['Bezbednost'] = np.select(uslovi, vrednost)

X = df1.iloc[:, :-1].copy()
y = df.iloc[:, -1].copy()

print(y.unique())


def evaluation_classifier(conf_mat):
    
    TP = conf_mat[0,0]
    TN = conf_mat[1,1] + conf_mat[1,2] + conf_mat[2,1] + conf_mat[2,2]
    FP = conf_mat[1,0] + conf_mat[2,0]
    FN = conf_mat[0,1] + conf_mat[0,2]
    precision = TP/(TP+FP)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    F_score = 2*precision*sensitivity/(precision+sensitivity)
    print('Za bezbedan vazduh:')
    print('precision: ', precision)
    print('accuracy: ', accuracy)
    print('sensitivity/recall: ', sensitivity)
    print('specificity: ', specificity)
    print('F score: ', F_score)
    TP = conf_mat[1,1]
    TN = conf_mat[0,0] + conf_mat[0,2] + conf_mat[2,0] + conf_mat[2,2]
    FP = conf_mat[0,1] + conf_mat[2,1]
    FN = conf_mat[1,0] + conf_mat[1,2]
    precision = TP/(TP+FP)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    F_score = 2*precision*sensitivity/(precision+sensitivity)
    print('Za nebezbedan vazduh:')
    print('precision: ', precision)
    print('accuracy: ', accuracy)
    print('sensitivity/recall: ', sensitivity)
    print('specificity: ', specificity)
    print('F score: ', F_score)
    TP = conf_mat[2,2]
    TN = conf_mat[0,0] + conf_mat[0,1] + conf_mat[1,0] + conf_mat[1,1]
    FP = conf_mat[0,2] + conf_mat[1,2]
    FN = conf_mat[2,0] + conf_mat[2,1]
    precision = TP/(TP+FP)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    F_score = 2*precision*sensitivity/(precision+sensitivity)
    print('Za opasan vazduh:')
    print('precision: ', precision)
    print('accuracy: ', accuracy)
    print('sensitivity/recall: ', sensitivity)
    print('specificity: ', specificity)
    print('F score: ', F_score)
    
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15,random_state=42, stratify=y)
klasifikator = KNeighborsClassifier()
klasifikator.fit(x_train, y_train)
ypredicted = klasifikator.predict(x_test)
matrica_konfuzije = confusion_matrix(y_test, ypredicted)
print(matrica_konfuzije)
evaluation_classifier(matrica_konfuzije)

for m in ['hamming', 'euclidean']:
    acc = []
    for i in range(1,10):
        classifier = KNeighborsClassifier(n_neighbors=i, metric=m)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        c = confusion_matrix(y_test, y_pred)
        print(c)
        acc.append(accuracy_score(y_test, y_pred))
    plt.figure(figsize=(12,6))
    plt.plot(range(1,10), acc, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
    plt.title('Error rate for ' + m)
    plt.xlabel('K value')
    plt.ylabel('Acc')
    
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
indexes = kf.split(X,y)
fin_conf_mat = np.zeros((len(np.unique(y)), len(np.unique(y))))
for train_index, test_index in indexes:
    x_train = X.iloc[train_index,:]
    x_test = X.iloc[test_index,:]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]
    
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    classifier = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    
    conf_mat = confusion_matrix(y_test, y_pred, labels=[0,1,2])
    print(conf_mat)
    evaluation_classifier(conf_mat)
    
    fin_conf_mat += conf_mat

print('finalna matrica: ')
print(fin_conf_mat)    
evaluation_classifier(fin_conf_mat)

classifier = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
c = confusion_matrix(y_test, y_pred, labels=[0,1,2])
print('Klasifikator sa konačno odabranim parametrima obučiti na celokupnom trening skupu:')
print(c)
evaluation_classifier(c)