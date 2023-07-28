# %%
# ----------------------------------------- Install -----------------------------------------
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)


# ----------------------------------------- Check Documents -----------------------------------------

train_csv = "train.csv"
train = pd.read_csv(train_csv)

print("Table: \n", train.head(10))
print("Table properties: \n", train.describe())
print("Table shape (rows, columns): \n", train.shape)
## 58 columns, 56 cathegories with EJ being categorical and rest as integer
## 618 rows, 617 training
print("Empty values: \n", train.isnull().sum())


## BQ : 60, CB : 2, CC : 3, DU : 1, EL : 60, FC : 1, FL : 1, FS : 2, GL : 1  --> EMPTY VALUES

# ----------------------------------------- Fill empty values with KNN -----------------------------------------

df_train = train.copy(deep=True)
df_train["Class"] = df_train["Class"].replace(
    {0: "Healthy", 1: "Sick"}
)  # Turn Class to object dtype

categorical_data = df_train.select_dtypes(exclude="number")
numeric_data = df_train.select_dtypes(include="number")

# Initialize the KNN imputer
imputer = KNNImputer(weights="distance", add_indicator=False)

# Fit and transform the imputer on the numeric data
imputed_numeric_data = imputer.fit_transform(numeric_data)

# Get the feature names including the added indicator columns
imputed_columns = imputer.get_feature_names_out(numeric_data.columns)

# Convert the imputed numeric data back to a DataFrame
imputed_df = pd.DataFrame(imputed_numeric_data, columns=imputed_columns)

# Concatenate the imputed numeric data with the categorical data
df_train = pd.concat([categorical_data, imputed_df], axis=1)
print("Table: \n", df_train.head(10))
print(df_train.dtypes)


def Save_datafile(df):
    path = "/home/sudenurcure/ICR-Identifying-Age-Related-Conditions/imputed_df.csv"
    df.to_csv(path)


"""
# ----------------------------------------- Label Encode Categorical Data -----------------------------------------
LE = LabelEncoder()
imputed_df["EJ"] = LE.fit_transform(imputed_df["EJ"])

print("Empty values: \n", imputed_df.isnull().sum())
print("Table: \n", imputed_df.head(10))



# ----------------------------------------- Corrolation and Visualizations -----------------------------------------

"""


def Correlation_map(df):
    print("Corrolation Heatmap: \n")
    corrolation = df.corr(method="spearman", min_periods=0)
    dataplot_hm = sns.heatmap(df.corr(method="spearman", min_periods=0))
    plt.show()


def Box_plot(df):
    print("Box plot: ")
    df.boxplot(by="Class", layout=(8, 7), figsize=(15, 15))


# Correlation_map(df_train)


def PCA_dim_reduction(x):
    pca_df = PCA(n_components=2)
    principalComponents_df = pca_df.fit_transform(x)
    principal_table_Df = pd.DataFrame(
        data=principalComponents_df,
        columns=["principal component 1", "principal component 2"],
    )
    print(principal_table_Df.tail())
    print(
        "Explained variation per principal component: {}".format(
            pca_df.explained_variance_ratio_
        )
    )
    # 80% information is lost. PCA 1 caught only 12% and PCA 2 caught only 8% of information.

    return principal_table_Df


def PCA_plot(df, principal_table_Df):
    plt.figure()
    plt.figure(figsize=(10, 10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel("Principal Component - 1", fontsize=20)
    plt.ylabel("Principal Component - 2", fontsize=20)
    plt.title("Principal Component Analysis of ICR Dataset", fontsize=20)
    targets = ["Healthy", "Sick"]
    colors = ["r", "g"]
    for target, color in zip(targets, colors):
        indicesToKeep = df["Class"] == target
        plt.scatter(
            principal_table_Df.loc[indicesToKeep, "principal component 1"],
            principal_table_Df.loc[indicesToKeep, "principal component 2"],
            c=color,
            s=50,
        )

    plt.legend(targets, prop={"size": 15})


def PCA_analysis(df):
    features = df.select_dtypes(include="number").columns
    x = df.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    print("shape: ", x.shape, "\nmean: ", np.mean(x), "\nstd: ", np.std(x))

    feat_cols = ["feature" + str(i) for i in range(x.shape[1])]
    normalised_df = pd.DataFrame(x, columns=feat_cols)
    print(normalised_df.tail())
    principal_table_Df = PCA_dim_reduction(x)
    PCA_plot(df, principal_table_Df)
    return 0


# PCA_analysis(df_train)

# ------------------Classes-----------------


class ML_Model(object):
    def __init__(self, model, y_predict, y_test):
        self.model = model
        self.y_predict = y_predict
        self.y_test = y_test

        # Find scores
        self.accuracy = accuracy_score(y_predict, y_test) * 100
        self.precision = precision_score(y_predict, y_test) * 100
        self.recall = recall_score(y_predict, y_test) * 100
        self.f1 = f1_score(y_predict, y_test) * 100

    def ConfM(self):
        cm = confusion_matrix(self.y_test, self.y_predict)

        plt.rcParams["figure.figsize"] = (4, 4)
        sns.heatmap(cm, annot=True, cmap="Greens")
        plt.title("Confusion Matrix for " + self.model, fontweight=30, fontsize=20)
        plt.show()

    def score(self):  # print scores
        print("Model Accuracy Details (%):\n")
        print("Accuracy Score :", self.accuracy)  # Total true out of all
        print("Precision Score:", self.precision)  # True positives from all positives
        print(
            "Recall Score:", self.recall
        )  # True positives from real positive instances
        print(
            "f1 Score:", self.f1, "\n"
        )  # Rates the model based on precision and recall. 0-1 and higher means better score.
        self.ConfM()


# ----------------------------- ML Prep--------------------------
LE = LabelEncoder()
ml_df = df_train.copy(deep=True)

ml_df.drop(columns="Id", inplace=True)
ml_df["EJ"] = LE.fit_transform(ml_df["EJ"])
ml_df["Class"] = LE.fit_transform(ml_df["Class"])

print("Ml DF: \n", ml_df.head())

X = ml_df.drop(columns="Class")
y = ml_df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.25
)

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test: ", y_test.shape)


# --------------------- ML Algorithms--------------------------
# Decision Tree
DecisionTree = DecisionTreeClassifier()

model = DecisionTree.fit(X_train, y_train)
y_predict = model.predict(X_test)
DecTree = ML_Model("Decision Tree", y_predict, y_test)

# Random Forest
RandomForest = RandomForestClassifier(random_state=0)

model = RandomForest.fit(X_train, y_train)
y_predict = model.predict(X_test)
RandForr = ML_Model("Random Forest", y_predict, y_test)

# Logistic Regression
Logistic_Regression = LogisticRegression(max_iter=10000000)

model = Logistic_Regression.fit(X_train, y_train)
y_predict = model.predict(X_test)
LogReg = ML_Model("Logistic Regression", y_predict, y_test)

print(DecTree.model, "\n")
DecTree.score()
print(RandForr.model, "\n")
RandForr.score()
print(LogReg.model, "\n")
LogReg.score()

compare = pd.DataFrame(
    {
        "Model": [DecTree.model, RandForr.model, LogReg.model],
        "Accuracy Scores": [DecTree.accuracy, RandForr.accuracy, LogReg.accuracy],
    }
)

print(compare)

# Logistic Regression looks like a good candidate
# %%
