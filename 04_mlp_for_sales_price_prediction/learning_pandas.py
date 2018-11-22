import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

# 1. read the data
print("\n1. Reading in training data...")
train = pd.read_csv('01_kaggle_dataset_house_prices/train.csv')
print("Type of train is ", type(train))

# 2. print data frame meta data
print("\n2. Shape of data frame is", train.shape)
nr_rows = train.shape[0]
nr_cols = train.shape[1]


# 3. show names of all columns and
#    each corresponding data type
print("\n3. Here are all column names:")
print("Type of train.columns.values is =",
      type(train.columns.values))
column_names = train.columns.values
i = 0
for column_name in column_names:
    print("column #", i, ":",
          "column name =", column_name,
          "\tdata type is", train.dtypes[i])
    i+=1


# 4. get a complete row and show it
print("\n4. Here is the first row of the table:")
first_row = train.iloc[0]
print("Type of first_row is", type(first_row))
print(first_row.values)


# 5. show for some rows year and SalesPrice
print("\n5. Now we show for the first 10 rows "
      "the year and the sales price:")
yearbuilt_column = train["YearBuilt"].values
salesprice_column = train["SalePrice"].values
print("Type of yearbuilt_column is", type(yearbuilt_column))
#for row_nr in range(0,nr_rows):
for row_nr in range(0,10):
    print ("row nr #", row_nr, " : "
           "YearBuilt: ", yearbuilt_column[row_nr],
           "--> SalePrice:" , salesprice_column[row_nr])


# 6. plot saleprice as a function of yearbuilt

print("\n6. Now we plot the sale price as a "
      "function of the year built")

# 6.1 set title of the plot
fig = plt.figure("Sale price")
fig.suptitle('Sale price as a function of year built',
             fontsize=20)

# 6.2 do a scatter plot
plt.scatter(yearbuilt_column, salesprice_column, marker='+')
plt.grid(True)
plt.xlabel('year built', fontsize=14)
plt.ylabel('sale price', fontsize=14)
plt.show(block=True)
plt.pause(1.0)


# 7. compute Pearson correlation of
#   (year_built, sale price) vectors
print("\n7. Pearson correlation coefficient "
      "of (year_built, sale price) vectors is : ",
      pearsonr(yearbuilt_column,salesprice_column)[0])


# 8. now get a new data frame only with numeric values
print("\n8. Preparing a new data frame with only "
      "numeric columns")
numerics = ['int16', 'int32', 'int64', 'float16',
            'float32', 'float64']
new_data_frame = train.select_dtypes(include=numerics)


# 9. show all numeric column names
print("\n9. Here are only the columns which "
      "contain numeric data")
numeric_column_names = list(new_data_frame.columns.values)
print("There are ", len(numeric_column_names),
      "numeric columns")
i = 0
for col_name in numeric_column_names:
    print("numeric column #", i, ":",
          "column name =", col_name)
    i+=1


# 10. try to find out what are columns
#     correlated highly to sale price
print("\n10. Here is the correlation of each "
      "numeric column with the 'SalesPrice' column:")
saleprice_col = new_data_frame["SalePrice"]
CORR_THRESHOLD = 0.6
highly_correlated_cols = []
for col_name in numeric_column_names:
    col = new_data_frame[col_name]
    pearsoncorr = pearsonr(col, saleprice_col)[0]
    print("Pearson correlation of ", col_name,
          "and 'SalePrice' is", pearsoncorr)
    if (pearsoncorr>CORR_THRESHOLD) and (col_name != "SalePrice"):
        highly_correlated_cols.append(col_name)


# 11. show names of columns that are highly correlated
#     with 'SalePrice' column
print("\n11. List of columns highly correlated with "
      "the sale price:")
print("Here highly correlated means, that the Pearson "
      "correlation coefficient is above", CORR_THRESHOLD)
print(highly_correlated_cols)


# 12. Now print some examples rows of the highly
#     correlated columns and the sale price
print("\n12. Here are some examples of the values in "
      "the columns that are highly correlated with the "
      "sale price")
col_saleprice = new_data_frame["SalePrice"].values
for row_nr in range(0,10):
    for column_name in highly_correlated_cols:
        col_data = new_data_frame[column_name].values
        print("(",
              column_name,
              ",",
              col_data[row_nr],
              ")",
              end=" ")
    print("--> salesprice was", col_saleprice[row_nr])





