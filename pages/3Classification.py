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

st.set_page_config(page_title="Classification", layout="centered")

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
        # Prompt user to choose 'k' for K-Nearest Neighbors
        k = st.slider("Choose 'k' for K-Nearest Neighbors", min_value=1, max_value=20, value=None)
        # Prompt user to choose 'n_estimators' for Random forest
        n = st.slider("Choose 'n_estimators' for Random Forest", min_value=1, max_value=100, value=None)
        if k is None :
            st.write("Please choose the value of 'k' for KNN before proceeding.")
        if n is None:
            st.write("Please choose the value of 'n_estimators' for Random Forest before proceeding.")
            # Optionally clear any previous plots or tables
            st.empty()
        else:
            st.write("### Results for K-Nearest Neighbors and Random Forest")
        
            # Split data into train-test sets
            rnd = 42
            test_size = 0.2
            pca = PCA(n_components=2)   
            reduced_features = pca.fit_transform(features) 
            X_train, X_test, y_train, y_test = train_test_split(features, class_labels, test_size=test_size, random_state=rnd,stratify=class_labels)
            X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(reduced_features, class_labels, test_size=test_size, random_state=rnd,stratify=class_labels)

            # Function to train and evaluate the model
            def train_and_evaluate(algo, X_train, X_test, y_train, y_test, p=None):
                if algo == "K-Nearest Neighbors":
                    model = KNeighborsClassifier(n_neighbors=p)
                elif algo == "Random Forest":
                    model = RandomForestClassifier(n_estimators=p)

                model.fit(X_train, y_train)
                ypred_tr = model.predict(X_train)
                ypred_tst = model.predict(X_test)

                tr_accuracy = accuracy_score(y_train, ypred_tr)
                tst_accuracy = accuracy_score(y_test, ypred_tst)
                tr_f1 = f1_score(y_train, ypred_tr, average='weighted')
                tst_f1 = f1_score(y_test, ypred_tst, average='weighted')

                # ROC AUC calculation
                try:
                    y_train_proba = model.predict_proba(X_train)
                    y_test_proba = model.predict_proba(X_test)
                    tr_roc_auc = roc_auc_score(y_train, y_train_proba, multi_class='ovr')
                    tst_roc_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr')
                except AttributeError:
                    tr_roc_auc = None
                    tst_roc_auc = None

                return tr_accuracy, tst_accuracy, tr_f1, tst_f1, tr_roc_auc, tst_roc_auc

            
            # Train and evaluate models on original and PCA features
            results=[]
            classifiers=["K-Nearest Neighbors","Random Forest"]
            params=[k,n]
            for ii,clf in enumerate(classifiers):
                results.append(train_and_evaluate(clf, X_train, X_test, y_train, y_test, p=params[ii]))
                results.append(train_and_evaluate("K-Nearest Neighbors", X_train_pca, X_test_pca, y_train_pca, y_test_pca, p=params[ii]))
                
            
            results =[np.array(res) for res in results]
            results = np.array(results)
            
            # Metrics Before PCA
            st.write("### Metrics Before Feature Reduction")
            perf_metrics = ["Train Accuracy", "Test Accuracy", "Train F1", "Test F1", "Train ROC AUC", "Test ROC AUC"]
            results_df = pd.DataFrame({
                "Metric": perf_metrics,
                "KNN": results[0],
                "Random Forest": results[2]
            })
            st.write(results_df)
            
            # Metrics After PCA
            st.write("### Metrics After Feature Reduction (PCA)")
            results_df_pca = pd.DataFrame({
                "Metric": perf_metrics,
                "KNN (PCA)": results[1],
                "Random Forest (PCA)": results[3]
            })
            st.write(results_df_pca)

            # Function to plot metrics for Train and Test in separate figures
            def plot_metrics(knn_train_before,knn_train_after,rf_train_before,rf_train_after,dist="train"):
                metrics =["Accuracy","F1", "ROC_AUC"]
                for j,met in enumerate(metrics):
                
                    bar_width = 0.1
                    index = range(len(labels))
                    # creating the dataset
                    data = {'KNN':knn_train_before[j],'KNN(PCA)':knn_train_after[j] ,'RF':rf_train_before[j],'RF(PCA)':rf_train_after[j]}
                    keys = list(data.keys())
                    vals = list(data.values())
                        
                    
                    colors = ['red', 'green', 'blue', 'orange']
                    fig = plt.figure()
                    # creating the bar plot
                    plt.bar(keys, vals, color =colors, width = 0.4)
                    plt.ylim(0, 1)
                    plt.ylabel(metrics[j])
                    plt.xlabel("Clasification Algorithms")
                    plt.title("Comaprison of "+ metrics[j]+" for "+ dist+" data")
                    st.pyplot(fig)

                
            # Prepare data for plotting (separate train and test)
            knn_train_before = results[0,[0,2,4]]  # Train Accuracy, Train F1, Train ROC AUC for KNN before PCA
            knn_train_after = results[1,[0,2,4]]   # Train Accuracy, Train F1, Train ROC AUC for KNN after PCA
            rf_train_before = results[2,[0,2,4]]   # Train Accuracy, Train F1, Train ROC AUC for RF before PCA
            rf_train_after = results[3,[0,2,4]]    # Train Accuracy, Train F1, Train ROC AUC for RF  After PCA 
            
            knn_test_before = results[0,[1,3,5]]   #  Test Accuracy, Train F1, Train ROC AUC for KNN before PCA
            knn_test_after = results[1,[1,3,5]]    # Test Accuracy, Train F1, Train ROC AUC for KNN after PCA
            rf_test_before = results[2,[1,3,5]]   # Test Accuracy, Train F1, Train ROC AUC for RF before PCA
            rf_test_after = results[3,[1,3,5]]  # Test Accuracy, Train F1, Train ROC AUC for RF after PCA
            
            # Plot separate Train and Test metrics in separate figures
            plot_metrics(knn_train_before,knn_train_after,rf_train_before,rf_train_after,"train")
            plot_metrics(knn_test_before,knn_test_after,rf_test_before,rf_test_after,"test")
else:
    st.write("# Παρακαλώ ανεβάστε ενα αρχείο CSV ή Excel")
    st.page_link("Home.py", label="Upload File", icon="🏠")
