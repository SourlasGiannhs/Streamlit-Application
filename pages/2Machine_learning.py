import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


st.set_page_config(page_title="Machine Learning", layout="centered")

if st.session_state['df'] is not None:
    data = st.session_state['df']
    # Create tabs

    def check_table_structure(data):
        num_columns = len(data.columns) # Determine the no of columns
        if num_columns < 2:
            return 0, "Table must have at least one feature and one label column." # Condition for checking that file must have sufficient data
        if data.isnull().values.any():
            return 2, "Data contains missing values." # Checking for missing values in the data
        return 1, "Data is structured correctly."    # Data is properly structured




        # Validate table structure
    is_valid, message = check_table_structure(data)
    if is_valid==1 or is_valid==2 :
        if is_valid==2:                                 # Data contain missing values
            st,write(message)
            st.write("Now remvoing missing values")
        num_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
        for column in data.columns:
            if column in num_cols:
                data[column].fillna(data[column].mean(), inplace=True)    # Filling missing values with mean
            else:
                data[column].fillna(data[column].mode()[0], inplace=True)   # Filling missing values with mode 
        st.success("Data loaded successfully!")
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = data[col].astype('category').cat.codes
        features = data.iloc[:, :-1]  # Seperating the Features
        labels = data.iloc[:, -1]     # Seperating the Labels

        label_encoder = LabelEncoder()
        class_labels = label_encoder.fit_transform(labels)  # Encoding categorical labels into numerical values

        st.write("### Data Preview")
        st.write(data)                                              # Displaying the content of dataframe in the form of table
        st.write(f"Number of Samples (S): {data.shape[0]}")         # Displaying the number of samples
        st.write(f"Number of Features (F): {data.shape[1] - 1}")    # Displaying the number of features
        st.write(f"Output Label Column: {data.columns[-1]}")        # Displaying the name of label column
    
        st.subheader("Feature Selection using PCA")
        m,n = features.shape                            #initilizing m and n with number of rows and number of columns respectively
        # Slider for PCA components, show 1slider at all fetaures initially
        n_comp = st.slider("Select the number of PCA components", min_value=1, 
                                    max_value=n, value=n)      # Displaying a slider for adjusting the no of PCA components

        # PCA transformation
        pca = PCA(n_components=n_comp)
        reduced_features = pca.fit_transform(features)       # Returning reduced features based on result from the slider

        #st.write("Explained Variance Ratio: "+str(pca.explained_variance_ratio_))
        cols=['PCA'+str(i) for i in range(n_comp)]
        # Display reduced dataset
        df_new = pd.DataFrame(reduced_features, columns=cols)
        df_new['label'] = labels
        st.dataframe(df_new)          # Displaying the reduced dataset in the form of table
else:
    st.write("# Παρακαλώ ανεβάστε ενα αρχείο CSV ή Excel")
    st.page_link("Home.py", label="Upload File", icon="🏠")
