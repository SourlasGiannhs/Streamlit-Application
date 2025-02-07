import streamlit as st
st.set_page_config(
    layout='wide'
)
st.image("streamlit.png")
st.write("## How to run the application using docker")
items = ["Run docker image using the following command", "docker run -d --name python-container -p 8501:8501 datamining-app", "The app will run on http://localhost:8501/"]
for i, item in enumerate(items, 1):
    st.write(f"{i}. {item}")   


st.write("### Steps performed" )
    
st.write("#### Data Upload" )
items = ["Upload tabular data files (csv,tsv and xlsx)", "Validate data and display error message if validation fails", "Check and impute missing values in data"]
for i, item in enumerate(items, 1):
    st.write(f"{i}. {item}")   
st.write("#### Visualization" )
items = ["Visualization of 2D PCA", "Visualization of 3D PCA", "Visualization of 2D UMAP", "Visualization of 3D UMAP", "Visualization of exploratory data analysis"]
for i, item in enumerate(items, 1):
    st.write(f"{i}. {item}")
st.write("#### Feature Selection" )
items = ["Provide slider for user to select no. of components", "Perform PCA on data", "Display transformed data"]
for i, item in enumerate(items, 1):
    st.write(f"{i}. {item}")
st.write("#### Classification" )
items = ["Provide slider for user to select k neighbors in KNN", "Provide slider for user to select n_estimators in Random Forest", "Split data into test and train data","Perform classification using both KNN and RF"]
for i, item in enumerate(items, 1):
    st.write(f"{i}. {item}")
st.write("#### Performance Evaluation" )
items = ["Perform predictions using KNN", "Perform predictions using Random Forest", 
"Evaluate KNN on train data using Accuracy, F1, and ROC-AUC ","Evaluate KNN on test data using Accuracy, F1, and ROC-AUC",
"Evaluate RF on train data using Accuracy, F1, and ROC-AUC ","Evaluate RF on test data using Accuracy, F1, and ROC-AUC",
"Plot performance of both models on train data ","Plot performance of both models on test data "]
for i, item in enumerate(items, 1):
    st.write(f"{i}. {item}")
st.write("#### Extensive testing" )
items = ["Perform Extensive testing for  all functionalities of the application"]
for i, item in enumerate(items, 1):
    st.write(f"{i}. {item}")
st.write("#### Deployment" )
items = ["Generate docker fom the code files"]
for i, item in enumerate(items, 1):
    st.write(f"{i}. {item}")


