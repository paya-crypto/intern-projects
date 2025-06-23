# World Bank Data – Data Cleaning & Preprocessing

## 📌 Objective
To clean and preprocess the World Bank dataset in preparation for Machine Learning tasks.

## 🛠️ Tools Used
- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib

## 🔍 Steps Performed

1. **Data Loading & Exploration**
   - Loaded the dataset and printed basic info (null counts, datatypes, sample rows).

2. **Missing Value Handling**
   - Filled numeric missing values using **mean** or **median**.
   - Filled categorical missing values using the **mode**.

3. **Encoding Categorical Features**
   - Converted categorical columns into numerical format using **one-hot encoding**.

4. **Normalization**
   - Scaled all numerical columns using **Z-score normalization**.

5. **Outlier Visualization & Removal**
   - Plotted boxplots to detect outliers.
   - Removed rows with extreme outliers using a 3-standard-deviation rule.

## 📊 Output
- Cleaned dataset with no missing values or extreme outliers.
- Boxplot showing distribution and outliers in numerical features.
