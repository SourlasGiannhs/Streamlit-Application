import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import umap.umap_ as umap
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

st.set_page_config(page_title="Visualisation", layout="centered")

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



        # Visualization Tab
        st.subheader("Visualization")

        # 2D PCA Visualization
        pca = PCA(n_components=2)                               # Reducing the dimensionality to two components  
        pca_result = pca.fit_transform(features)                # Returning the reduced data

        plt.figure()
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=class_labels, cmap='viridis')       # Creating the scatter plot
        plt.title("2D PCA Visualization")                   # Providing the title of plot
        plt.xlabel("PCA Component 1")                       # Providing the label for x axis of the plot 
        plt.ylabel("PCA Component 2")                       # Providing the label for y axis of the plot 
        plt.colorbar(scatter)                               # Adding colors to scatter plot
        st.pyplot(plt.gcf())                                # Getting current figure and displaying it

        # 3D PCA Visualization
        pca_3d = PCA(n_components=3)                         # Reducing the dimensionality to three components 
        pca_result_3d = pca_3d.fit_transform(features)       # Returning the reduced data
        fig = plt.figure()
        ax_3d = fig.add_subplot(111, projection='3d')
        scatter_3d = ax_3d.scatter(pca_result_3d[:, 0], pca_result_3d[:, 1], pca_result_3d[:, 2], 
                                    c=class_labels, cmap='viridis')               # Creating the scatter plot
        ax_3d.set_title("3D PCA Visualization")                                  # Providing the title of plot
        ax_3d.set_xlabel("PCA Component 1")                                      # Providing the label for x axis of the plot
        ax_3d.set_ylabel("PCA Component 2")                                      # Providing the label for y axis of the plot 
        ax_3d.set_zlabel("PCA Component 3")                                      # Providing the label for z axis of the plot 
        fig.colorbar(scatter_3d)                                                 # Adding colors to scatter plot
        st.pyplot(fig)                                                           # Displaying the plot

        # 2D UMAP Visualization
        reducer = umap.UMAP(n_components=2)                                       # Reducing the data to two dimensions
        umap_result = reducer.fit_transform(features)                              # Returning the reduced data
        plt.figure()
        scatter_umap = plt.scatter(umap_result[:, 0], umap_result[:, 1], c=class_labels, cmap='viridis')        # Creating the scatter plot
        plt.title("2D UMAP Visualization")             # Providing the title of plot
        plt.xlabel("UMAP Component 1")                 # Providing the label for x axis of the plot
        plt.ylabel("UMAP Component 2")                 # Providing the label for y axis of the plot
        plt.colorbar(scatter_umap)                     # Adding colors to scatter plot
        st.pyplot(plt.gcf())                           # Displaying the plot

        # 3D UMAP Visualization
        reducer_3d = umap.UMAP(n_components=3)                      # Reducing the data to three dimensions
        umap_result_3d = reducer_3d.fit_transform(features)         # Returning the reduced data
        fig = plt.figure()
        ax_3d_umap = fig.add_subplot(111, projection='3d')
        scatter_3d_umap = ax_3d_umap.scatter(umap_result_3d[:, 0], umap_result_3d[:, 1], umap_result_3d[:, 2], 
                                                c=class_labels, cmap='viridis')             # Creating the scatter plot
        ax_3d_umap.set_title("3D UMAP Visualization")                        # Providing the title of plot
        ax_3d_umap.set_xlabel("UMAP Component 1")                            # Providing the label for x axis of the plot
        ax_3d_umap.set_ylabel("UMAP Component 2")                            # Providing the label for y axis of the plot
        ax_3d_umap.set_zlabel("UMAP Component 3")                            # Providing the label for z axis of the plot 
        fig.colorbar(scatter_3d_umap)                                        # Adding colors to scatter plot
        st.pyplot(fig)                                                       # Displaying the plot

        # Exploratory Data Analysis (EDA)
        st.subheader("Exploratory Data Analysis (EDA)")

        # Correlation heatmap (using first 5 features)
        features_small = features  # Select the first 5 features
        fig = plt.figure()
        sns.heatmap(features_small.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")                    # Title of plot
        st.pyplot(fig)                                      # Displaying the plot

        # Pairplot for visualizing pairwise relationships (using first 5 features)
        subset_features = features  # Select the first 5 features
        subset_data = subset_features.copy()
        subset_data[data.columns[-1]] = labels  # Add label column for hue

        try:
            pairplot_fig = sns.pairplot(subset_data, hue=data.columns[-1], palette='viridis')
            plt.suptitle("Pairwise Relationships", y=1.02)
            st.pyplot(pairplot_fig)
        except Exception as e:
            st.error(f"Error generating pairplot: {e}")
else:
    st.write("# Παρακαλώ ανεβάστε ενα αρχείο CSV ή Excel")
    st.page_link("Home.py", label="Upload File", icon="🏠")
