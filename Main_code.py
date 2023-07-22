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

# Separate the sample names column (first column) from the rest of the data
df_train_strings = df_train.iloc[:, :1]
df_train_ints = df_train.iloc[:, 1:]

# Separate categorical and numerical data
categorical_data = df_train_ints.select_dtypes(include="object")
numeric_data = df_train_ints.select_dtypes(include="number")

# Initialize the KNN imputer
imputer = KNNImputer(weights="distance", add_indicator=True)

# Fit and transform the imputer on the numeric data
imputed_numeric_data = imputer.fit_transform(numeric_data)

# Get the feature names including the added indicator columns
imputed_columns = imputer.get_feature_names_out(input_features=numeric_data.columns)

# Convert the imputed numeric data back to a DataFrame
imputed_df = pd.DataFrame(imputed_numeric_data, columns=imputed_columns)

# Concatenate the imputed numeric data with the categorical data
imputed_df = pd.concat([df_train_strings, imputed_df, categorical_data], axis=1)

# ----------------------------------------- Label Encode Categorical Data -----------------------------------------
LE = LabelEncoder()
imputed_df["EJ"] = LE.fit_transform(imputed_df["EJ"])

print("Empty values: \n", imputed_df.isnull().sum())
print("Table: \n", imputed_df.head(10))


# ----------------------------------------- Corrolation and Visualizations -----------------------------------------

vis_df = imputed_df.drop(imputed_df.filter(regex=r"missing|Id").columns, axis=1)


def Correlation_map(df):
    print("Corrolation Heatmap: \n")

    corrolation = df.corr(method="spearman", min_periods=0)
    dataplot_hm = sns.heatmap(df.corr(method="spearman", min_periods=0))
    plt.show()

    print("Box plot: ")
    df.boxplot(by="Class", layout=(8, 7), figsize=(15, 15))


def Save_datafile(df):
    path = "/home/sudenurcure/ICR-Identifying-Age-Related-Conditions/imputed_df.csv"
    df.to_csv(path)


# Correlation_map(vis_df)


def PCA_dim_reduction(df):
    X, sorted_eig_vectors = PCA_analysis(df)
    eig_scores = np.dot(X, sorted_eig_vectors[:, :2])
    return eig_scores, sorted_eig_vectors


def PCA_analysis(df):
    X = StandardScaler().fit_transform(df)
    cov = (X.T @ X) / (X.shape[0] - 1)
    eig_values, eig_vectors = np.linalg.eig(cov)
    idx = np.argsort(eig_values)[::-1]
    sorted_eig_vectors = eig_vectors[:, idx]
    return X, sorted_eig_vectors


def biplot(score, coeff, labels):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]

    fig, ax = plt.subplots()

    for i, u in enumerate(vis_df["Class"].unique()):
        xi = score[vis_df["Class"] == u, 0]
        yi = score[vis_df["Class"] == u, 1]
        ax.scatter(xi, yi, label=u)

    for i in range(n):
        ax.arrow(
            0, 0, coeff[i, 0], coeff[i, 1], color="r", head_width=0.05, head_length=0.1
        )
        ax.text(
            coeff[i, 0] * 1.35,
            coeff[i, 1] * 1.35,
            labels[i],
            color="g",
            ha="center",
            va="center",
        )

    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()
    plt.legend()


# Assuming vis_df is a global DataFrame variable containing the original data
X, sorted_eig_vectors = PCA_dim_reduction(vis_df)
biplot(X, sorted_eig_vectors, vis_df.columns)
plt.show()

# %%
