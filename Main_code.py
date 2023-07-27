# %%
# ----------------------------------------- Install -----------------------------------------
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


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


Correlation_map(df_train)


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


PCA_analysis(df_train)


# %%
