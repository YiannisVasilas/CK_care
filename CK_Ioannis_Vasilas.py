#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
from lifelines import AalenAdditiveFitter
import hdbscan

def preprocess_data(file_path):
    # Load the data into a Pandas DataFrame
    df = pd.read_csv(file_path)

    # Remove the first column of the data
    df = df.drop(df.columns[0], axis=1)
    
    # Replace 'yes' with 1 and 'no' with 0 in the binary columns
    binary_columns = ['Food_allergy', 'Allergic_rhinitis', 'Psoriasis', 'Asthma', 'contact_animals_farm', 'contact_animals_domestic']
    for col in binary_columns:
        df[col] = df[col].replace({'yes': 1, 'no': 0})
    
    # Replace 'M' with 0 and 'F' with 1 in the 'gender' column
    df['gender'] = df['gender'].replace({'M': 0, 'F': 1})
    
    # Replace the values in the 'Outcome2' column with numerical values
    df['Outcome2_num'] = df['Outcome2'].replace({'healthy':1.0, 'mild':2.0, 'moderate':3.0, 'severe':4.0})
    
    # Convert columns to appropriate data types
    df[binary_columns + ['gender']] = df[binary_columns + ['gender']].astype(np.int64)
    df[['age', 'BMI', 'Outcome2_num']] = df[['age', 'BMI', 'Outcome2_num']].astype(np.float64)
    df[[f'biomarker_{i}' for i in range(1, 92)]] = df[[f'biomarker_{i}' for i in range(1, 92)]].astype(np.float64)
    
    return df


def plot_eda(file_path):
    df = preprocess_data(file_path)
    
    # Check the first few rows of the data
    print("\nFirst few rows of the data:")
    print(df.head())
    
    # Get summary statistics for the numerical variables
    print("\nSummary Statistics for Numerical Variables:")
    print(df.describe())
    
        # Filter the data to only keep the biomarker columns
    biomarker_cols = [col for col in df.columns if col.startswith('biomarker_')]
    df_b= df[biomarker_cols]
    # Plot histograms for the numerical variables
    print("\nHistograms of Numerical Variables:")
    df_b.hist(bins=50, figsize=(20,15), color='blue')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histograms of Numerical Variables')
    plt.show()
    
    # Plot box plots for the numerical variables
    print("\nBox Plots of Numerical Variables:")
    fig, ax = plt.subplots(figsize=(15,10))
    sns.boxplot(data=df_b, width=0.5, fliersize=5, ax=ax)
    plt.xlabel('Variables')
    plt.ylabel('Values')
    plt.title('Box Plots of Numerical Variables')
    plt.show()
    
    # Plot bar plots for the categorical variables
    print("\nBar Plot of Outcome1 Variable:")
    sns.countplot(x='Outcome1', data=df, palette='viridis')
    plt.xlabel('Outcome1')
    plt.ylabel('Count')
    plt.title('Bar Plot of Outcome1 Variable')
    plt.show()
    
    print("\nHeatmap of Correlation between Biomarkers:")
    corr = df_b.corr()
    
    plt.figure(figsize=(10,10))
    sns.heatmap(corr, annot=False, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.title('Heatmap of Correlation between Biomarkers')
    plt.show()
    
    # Plot a Pair Plot
    print("\nPair Plot of Numerical Variables:")
    sns.pairplot(df_b, diag_kind='hist', plot_kws={'alpha': 0.5})
    plt.title('Pair Plot of Numerical Variables')
    plt.show()
    
    # Plot a Violin Plot
    print("\nViolin Plot of Numerical Variables:")
    fig, ax = plt.subplots(figsize=(15,10))
    sns.violinplot(data=df_b, ax=ax)
    plt.xlabel('Variables')
    plt.ylabel('Values')
    plt.title('Violin Plot of Numerical Variables')
    plt.show()


def association_analysis(df, support=0.5, confidence=0.8, lift=1, plot=True):
    # Convert the data into a list of transactions
    te = TransactionEncoder()
    te_ary = te.fit(df).transform(df)

    # Convert the list of transactions into a Pandas DataFrame
    df_te = pd.DataFrame(te_ary, columns=te.columns_)

    # Perform association analysis using the Apriori algorithm
    frequent_itemsets = apriori(df_te, min_support=support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)
    rules = rules[rules['lift'] >= lift]
    
    if plot:
        # Generate a plot of support vs. number of itemsets
        supports = frequent_itemsets['support'].values
        plt.scatter(range(1, len(supports)+1), supports)
        plt.xlabel('Number of Itemsets')
        plt.ylabel('Support')
        plt.show()

        # Generate a plot of length vs. support
        lengths = frequent_itemsets['itemsets'].apply(len)
        plt.scatter(lengths, supports)
        plt.xlabel('Length of Itemsets')
        plt.ylabel('Support')
        plt.show()
    
    return rules


def perform_clustering(df):
    # Select the columns for clustering
    data = df[[f'biomarker_{i}' for i in range(1, 92)]]
    
    # Perform HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
    clusters = clusterer.fit_predict(data)
    
    # Add the cluster labels to the data frame
    df['cluster'] = clusters
    
    return df




def plot_clusters(df):
    sns.scatterplot(data=df, x='biomarker_1', y='biomarker_2', hue='cluster', palette='viridis')
    plt.title('HDBSCAN Clustering Results')
    plt.xlabel('Biomarker 1')
    plt.ylabel('Biomarker 2')
    plt.show()

def survival_analysis(file_path, columns, event, title):


    # Load the data into a pandas DataFrame
    data = pd.read_csv(file_path)

    # Select the columns of interest for the analysis
    data = data[columns]
    
    # Convert the data to numeric type
    data[columns[1]] = data[columns[1]].map({'AD': 1, 'HC': 0})
    data[columns[1]] = data[columns[1]].astype(float)
    data[columns[1]] = pd.to_numeric(data[columns[1]], errors='coerce')

    # Drop rows with NaN values
    data.dropna(inplace=True)
    
    # Fit the Kaplan-Meier estimator to the data
    kmf = KaplanMeierFitter()
    kmf.fit(data[columns[0]], event_observed=data[columns[1]])
    
    # Plot the survival function
    ax = kmf.plot()
    ax.set_title(title)
    
    # Show the plot
    plt.show()



def survival_analysis2(file_path, columns, event, title):


    # Load the data into a pandas DataFrame
    data = pd.read_csv(file_path)

    # Select the columns of interest for the analysis
    data = data[columns]
    
    # Convert the data to numeric type
    data[columns[1]] = data[columns[1]].map({'AD': 1, 'HC': 0})
    data[columns[1]] = data[columns[1]].astype(float)
    data[columns[1]] = pd.to_numeric(data[columns[1]], errors='coerce')

    # Drop rows with NaN values
    data.dropna(inplace=True)
    
    # Fit the Cox proportional hazards model to the data
    cph = CoxPHFitter()
    cph.fit(data, duration_col=columns[0], event_col=columns[1])
    
    # Plot the hazard ratio
    cph.plot()
    plt.title(title)
    
    # Show the plot
    plt.show()


def survival_analysis3(file_path, columns, event, title):

    # Load the data into a pandas DataFrame
    data = pd.read_csv(file_path)

    # Select the columns of interest for the analysis
    data = data[columns]
    
    # Convert the data to numeric type
    data[columns[1]] = data[columns[1]].map({'AD': 1, 'HC': 0})
    data[columns[1]] = data[columns[1]].astype(float)
    data[columns[1]] = pd.to_numeric(data[columns[1]], errors='coerce')

    # Drop rows with NaN values
    data.dropna(inplace=True)
    
    # Fit the Nelson-Aalen estimator to the data
    naf = NelsonAalenFitter()
    naf.fit(data[columns[0]], event_observed=data[columns[1]])
    
    # Plot the cumulative hazard rate
    ax = naf.plot()
    ax.set_title(title)
    
    # Show the plot
    plt.show()


def survival_analysis4(file_path, columns, event, title):

    # Load the data into a pandas DataFrame
    data = pd.read_csv(file_path)

    # Select the columns of interest for the analysis
    data = data[columns]
    
    # Convert the data to numeric type
    data[columns[1]] = data[columns[1]].map({'AD': 1, 'HC': 0})
    data[columns[1]] = data[columns[1]].astype(float)
    data[columns[1]] = pd.to_numeric(data[columns[1]], errors='coerce')

    # Drop rows with NaN values
    data.dropna(inplace=True)
    
    # Fit the Aalen's additive hazards model to the data
    aaf = AalenAdditiveFitter()
    aaf.fit(data, duration_col=columns[0], event_col=columns[1])
    
    # Plot the cumulative hazards
    ax = aaf.plot()
    ax.set_title(title)
    
    # Show the plot
    plt.show()




def survival_prediction_with_ML(file_path, columns, event, title):
  

    # Load the data into a pandas DataFrame
    data = pd.read_csv(file_path)

    # Select the columns of interest for the analysis
    data = data[columns]
    
    # Convert the data to numeric type
    data[columns[1]] = data[columns[1]].map({'AD': 1, 'HC': 0})
    data[columns[1]] = data[columns[1]].astype(float)
    data[columns[1]] = pd.to_numeric(data[columns[1]], errors='coerce')

    # Drop rows with NaN values
    data.dropna(inplace=True)
    
    # Split the data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(data[columns[0]], data[columns[1]], test_size=0.2)

    # Train a logistic regression model on the training data
    model = LogisticRegression()
    model.fit(train_data.values.reshape(-1, 1), train_labels)

    # Make predictions on the validation set
    val_predictions = model.predict(val_data.values.reshape(-1, 1))

    # Plot the predicted vs. actual labels
    plt.scatter(val_data, val_labels)
    plt.plot(val_data, val_predictions, color='red')
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.title(title)
    plt.show()
    
    # Plot the distribution of the predictions
    plt.hist(val_predictions, bins=2)
    plt.xlabel('Prediction')
    plt.ylabel('Count')
    plt.title(title)
    plt.show()
    
    # Calculate the mean squared error between the predictions and the actual labels
    mse = mean_squared_error(val_labels, val_predictions)
    print("Mean squared error:", mse)
    # Calculate the accuracy of the model
    accuracy = accuracy_score(val_labels, val_predictions)
    print("Accuracy:", accuracy)
    # Plot the confusion matrix
    cm = confusion_matrix(val_labels, val_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()




def voting_ensemble_classifier(file_path, columns, event, title):
    # Load the data into a pandas DataFrame
    data = pd.read_csv(file_path)
    
    # Select the columns of interest for the analysis
    data = data[columns]
    
    # Convert the data to numeric type
    data[columns[1]] = data[columns[1]].map({'AD': 1, 'HC': 0})
    data[columns[1]] = data[columns[1]].astype(float)
    data[columns[1]] = pd.to_numeric(data[columns[1]], errors='coerce')
    
    # Drop rows with NaN values
    data.dropna(inplace=True)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data[columns[0]], data[columns[1]], test_size=0.2, random_state=42)
    
    # Initialize the classifiers
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr_clf = LogisticRegression(random_state=42)
    knn_clf = KNeighborsClassifier()
    
    # Create the voting ensemble classifier
    vc_clf = VotingClassifier(estimators=[('rf', rf_clf), ('lr', lr_clf), ('knn', knn_clf)], voting='soft')
    
    # Fit the voting ensemble classifier to the training data
    vc_clf.fit(X_train.values.reshape(-1, 1), y_train)
    
    # Predict the labels of the test set
    X_test_array = X_test.to_numpy()
    y_pred = vc_clf.predict(X_test_array.reshape(-1, 1))
    
    # Compute the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    # Plot the results
    plt.plot(y_test, label='True')
    plt.plot(y_pred, label='Predicted')
    plt.title(title)
    plt.legend()
    plt.show()
    
    # Compute and plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, cmap='binary')
    plt.colorbar()
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()


file_path= '/home/ivasilas/Downloads/DF_for_CInterview.csv'
df = preprocess_data(file_path)
plot_eda(file_path)
n= perform_clustering(df)
association_analysis(df)

plot_clusters(n)

columns = ['age', 'Outcome1']
event = 'Outcome1'
title = 'Survival Analysis of Atopic Disease Outcome by Age'

survival_analysis(file_path, columns, event, title)
survival_analysis2(file_path, columns, event, title)
survival_analysis3(file_path, columns, event, title)
survival_analysis4(file_path, columns, event, title)
survival_prediction_with_ML(file_path, columns, event,title)
voting_ensemble_classifier(file_path, columns, event,title)





