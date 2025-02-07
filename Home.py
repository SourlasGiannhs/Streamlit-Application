import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
import streamlit as st
import csv

st.set_page_config(
    page_title="Home",
    page_icon="./ionio.png",
    layout='wide'
)



st.write("# Καλώς ήρθατε στην εφαρμογή οπτικοποίησης και ανάλυσης δεδομένων :bar_chart:")
st.title("Υλοποίηση 2D & 3D οπτικοποίησεων δεδομένων βασισμένες στους αλγορίθμους PCA & UMAP")
st.write('\n')
st.write('\n')
st.write('\n')
st.write("Σε αυτή την εφαρμογή χρησιμοποιώ τον αλγόριθμο **Random Forest** and **Extra Trees** "
         "ως 'Feature selection' αλγορίθμους για να με βοηθήσουν να μικρύνω το μέγεθος του dataset")
st.session_state['df'] = None
input_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "tsv"])
def check_table_structure(data):
    num_columns = len(data.columns) # Determine the no of columns
    if num_columns < 2:
        return 0, "Table must have at least one feature and one label column." # Condition for checking that file must have sufficient data
    if data.isnull().values.any():
        return 2, "Data contains missing values." # Checking for missing values in the data
    return 1, "Data is structured correctly."    # Data is properly structured

if input_file is not None:    # Condition to check the input file must have some data to process
    try:
        # Handle CSV, Excel, TSV files
        if input_file.name.endswith(".csv"):  # Comparing file extension to determine the type of file
            data = pd.read_csv(input_file)    # Function to read csv file
        elif input_file.name.endswith(".xlsx"):
            data = pd.read_excel(input_file)   # Function to read excel file
        elif input_file.name.endswith(".tsv"):
            data = pd.read_csv(input_file, delimiter='\t')  # Function to read tsv file
    except Exception as e:
        st.error(f"Error loading file: {e}")
    st.session_state['df'] = data
else:
    st.write("No file uploaded yet")
st.write("**Made By: Σούρλας Ιωάννης Π2016102**")
st.write("\n")
st.write("\n")
st.write("\n")



st.image("ionio.png", caption="Ionian University department of Informatics")

st.sidebar.success("Select a function above.")
















