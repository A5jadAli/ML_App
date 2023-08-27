import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
import plotly.express as px
import joblib

# Set title of the web app
st.title('Algorithmia')
st.subheader("Welcome to Alogrithmia! Unleash the Power of Algorithms and Choose the Best Model")
# Create a file upload button on the left side of the page
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv", "xlsx"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Load the data into a Pandas DataFrame
    data = pd.read_csv(uploaded_file)

    # Find the categorical and numerical columns of the uploaded data
    cat_cols = list(data.select_dtypes(include=['object']).columns)
    num_cols = list(data.select_dtypes(include=['float64', 'int64']).columns)

    # Print the categorical and numerical columns in the main web app
    st.write("""### Categorical Columns""")
    st.write(cat_cols)
    st.write("""### Numerical Columns""")
    st.write(num_cols)
    # st.write('Categorical Columns:', cat_cols)
    # st.write('Numerical Columns:', num_cols)

    # Check if missing values exist in the numeric variables/columns and if exist, use KNN imputer to impute missing values for numeric variables/columns in the dataset
    if data[num_cols].isnull().sum().sum() > 0:
        imputer = KNNImputer(n_neighbors=5)
        data[num_cols] = imputer.fit_transform(data[num_cols])

    # Check if missing values exist in the categorical variables/columns and if exist, use mode() to impute missing values in them
    if data[cat_cols].isnull().sum().sum() > 0:
        data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])

    # Use sklearn label encoder to encode the categorical variables/columns of the data
    le = LabelEncoder()
    data[cat_cols] = data[cat_cols].apply(lambda col: le.fit_transform(col))

    # Ask the user to select multiple columns as features for machine learning algorithms X, add this multiple select option on the sidebar
    feature_cols = st.sidebar.multiselect('Select feature columns', data.columns)

    # Ask the user to select y or label column as well on side bar adding multiple selection column also ask user to select the percentage of train test split
    label_col = st.sidebar.selectbox('Select label column', data.columns)
    test_size = st.sidebar.slider('Select test size percentage', 10, 90, 30, 5)

    # Ask the user if it should be a regression or a classification problem using select box
    problem_type = st.sidebar.selectbox('Select problem type', ['Classification', 'Regression'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data[feature_cols], data[label_col], test_size=test_size/100, random_state=42)

    # Use the selected regression models in the app
    if problem_type == 'Regression':
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'SVR': SVR(),
            'DT Regressor': DecisionTreeRegressor(),
            'KNN Regressor': KNeighborsRegressor(),
            'RF Regressor': RandomForestRegressor(),
            'GB Regressor': GradientBoostingRegressor(),
            'XGBoost Regressor': XGBRegressor()
        }
        metrics = {
            'Mean Squared Error': mean_squared_error,
            'Root Mean Squared Error': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'Mean Absolute Error': mean_absolute_error,
            'R-Squared': r2_score,
            'Mean Squared Percentage Error': lambda y_true, y_pred: np.mean(((y_true - y_pred) / y_true) ** 2) * 100
        }

    # Use the selected classification models in the app
    else:
        models = {
            'Logistic Regression': LogisticRegression(),
            'SVC': SVC(),
            'KNN Classifier': KNeighborsClassifier(),
            'DT Classifier': DecisionTreeClassifier(),
            'RF Classifier': RandomForestClassifier(),
            'NB Classifier': GaussianNB(),
            'GB Classifier': GradientBoostingClassifier(),
            'XGBoost Classifier': XGBClassifier()
        }
        metrics = {
            'Precision Score': precision_score,
            'Accuracy Score': accuracy_score,
            'Recall Score': recall_score,
            'F1 Score': f1_score,
            'ROC-AUC Score': roc_auc_score
        }

    # Train and evaluate the selected models
    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores[name] = {}
        for metric_name, metric_func in metrics.items():
            try:
                score = metric_func(y_test, y_pred)
            except:
                score = np.nan
            scores[name][metric_name] = score

    # Print the results in a table in descending order
    st.write(pd.DataFrame(scores).T.sort_values(by=list(metrics.keys()), ascending=False))

    # Plot the data using plotly, especially the model evaluation part
    fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'True Values', 'y': 'Predictions'})
    st.plotly_chart(fig)

    # Show the user the best model from all selected
    best_model_name = max(scores, key=lambda k: np.mean([v for v in scores[k].values() if not np.isnan(v)]))
    st.write('Best Model:', best_model_name)

    # Add an option to save the model using joblib library so that if user wants he can save the model for future use
    if st.button('Save Model'):
        joblib.dump(models[best_model_name], 'model.joblib')
        st.write('Model saved successfully!')