# %%
# ----------------------------------------- Install -----------------------------------------
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
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

print("Empty values: \n", imputed_df.isnull().sum())
print("Table: \n", imputed_df.head(10))


# ----------------------------------------- Label Encode Categorical Data -----------------------------------------
LE = LabelEncoder()
imputed_df["EJ"] = LE.fit_transform(imputed_df["EJ"])

# ----------------------------------------- Corrolation and Visualizations -----------------------------------------

vis_df = imputed_df.drop(
    imputed_df.filter(regex="missing").columns, axis=1, inplace=False
)

print("Corrolation Heatmap: \n")

corrolation = vis_df.corr(method="spearman", min_periods=0)
dataplot_hm = sns.heatmap(vis_df.corr(method="spearman", min_periods=0))
plt.show()

path = "/home/sudenurcure/ICR-Identifying-Age-Related-Conditions/imputed_df.csv"
vis_df.to_csv(path)

# %%
