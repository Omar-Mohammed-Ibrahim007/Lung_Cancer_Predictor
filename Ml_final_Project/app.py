
# in terminal run --> streamlit run <path of .py app>


#---------------------------------------------------
#imports
#---------------------------------------------------
import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import RandomForestClassifier ,StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,ComplementNB,BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import time
import os
from sklearn.preprocessing import StandardScaler


#---------------------------------------------------
# Page Config
#---------------------------------------------------
st.set_page_config(page_title="Lung Cancer Predictor", layout="wide")

st.title("Lung Cancer Prediction App")
st.markdown(
    "Use the **Sidebar** to input patient data and select a **pretrained model** "
    "to predict lung cancer risk."
)

#---------------------------------------------------
# Load Artifacts
#---------------------------------------------------
@st.cache_resource
def load_artifacts():
    scaler = {}
    labelencoder = joblib.load(".\\workspace\\label_encoder.pkl")
    ordinalencoder = joblib.load(".\\workspace\\ordinal_encoder.pkl")
    dummies_cols = joblib.load(".\\workspace\\columns.pkl")
    

    models = {}
    for name in ['lr', 'gnb','mnb', 'cnb', 'bnb', 'knn', 'rf', 'dt', 'et', 'xgb', 'lgb']:
        models[name] = joblib.load(f".\\models\\{name}.pkl")
        scaler[name]=models[name].named_steps['scaler']

    with open(".\\workspace\\features.txt", "r") as f:
        features = f.read().splitlines() # file in csv formate
        features.remove("Final_Prediction")
    return scaler, ordinalencoder,labelencoder, dummies_cols,models, features

scaler,ordinalencoder, labelencoder,dummies_cols, models, features = load_artifacts()

#---------------------------------------------------
# Models Menu – Model Selection
#---------------------------------------------------
abbreviations={

"Logistic Regression":'lr',
"Gaussian Naive Bayes":'gnb',
"Multinomial Naive Bayes":'mnb',
"Complement Naive Bayes":'cnb',
"Bernoulli Naive Bayes":'bnb',
"K Nearest Neighbors":'knn',
"Random Forest":'rf',
"Decision Tree":'dt',
"Extra Trees":'et',
"XGBoost":'xgb',
"LightGMB":'lgb' 
}
st.sidebar.title("Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose ML Model",
    list(abbreviations.keys())
)

model = models[abbreviations[model_choice]]

#---------------------------------------------------
# Patient Menus – Patient Inputs
#---------------------------------------------------
st.sidebar.title("Patient Information")

Age = st.sidebar.slider("Age", 18, 90, 50)
Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
Smoking_Status = st.sidebar.selectbox(
    "Smoking Status",
    ["Non-Smoker", "Former Smoker", "Current Smoker"]
)
#---------------------------------------------------
# Select Menus
#---------------------------------------------------
Second_Hand_Smoke = st.sidebar.selectbox("Second Hand Smoke", ["Yes", "No"])
Air_Pollution = st.sidebar.selectbox("Air Pollution Exposure", ["Low", "Medium", "High"])
Occupational_Exposure = st.sidebar.selectbox("Occupational Exposure", ["Yes", "No"])
Rural_or_Urban = st.sidebar.selectbox("Rural_or_Urban ", ['Rural','Urban'])
Socioeconomic_Status = st.sidebar.selectbox("Socioeconomic Status", ["Low", "Middle", "High"])
Healthcare_Access = st.sidebar.selectbox("Healthcare Access", ["Poor", "Limited", "Good"])
Insurance = st.sidebar.selectbox("Health Insurance", ["Yes", "No"])
Screening = st.sidebar.selectbox("Cancer Screening", ["Yes", "No"])
Stage_at_Diagnosis = st.sidebar.selectbox("Stage at Diagnosis", ["I", "II", "III", "IV"])
Cancer_Type = st.sidebar.selectbox("Cancer Type", ["NSCLC", "SCLC","None"])
Mutation_Type = st.sidebar.selectbox("Mutation Type", ["EGFR", "ALK", "KRAS", "None"])
Treatment = st.sidebar.selectbox("Treatment", ["None", "Partial", "Full"])
Clinical_Trial = st.sidebar.selectbox("Clinical Trial", ["Yes", "No"])
Delay_in_Diagnosis = st.sidebar.selectbox("Delay in Diagnosis", ["Yes", "No"])
Family_History = st.sidebar.selectbox("Family History", ["Yes", "No"])
Indoor_Smoking = st.sidebar.selectbox("Indoor Smoking", ["Yes", "No"])
Tobacco_Use = st.sidebar.selectbox("Active Tobacco Use", ["Yes", "No"])

#---------------------------------------------------
#SLiders for continues values
#---------------------------------------------------
Mortality_Risk = st.sidebar.slider("Mortality Risk", 0.0, 1.0, 0.3)
Five_Year_Survival = st.sidebar.slider("5-Year Survival Probability", 0.0, 1.0, 0.7)

#---------------------------------------------------
# Prepare Input Data
#---------------------------------------------------
input_data = pd.DataFrame([{
    "Age": Age,
    "Gender": Gender,
    "Smoking_Status": Smoking_Status,
    "Second_Hand_Smoke": Second_Hand_Smoke,
    "Air_Pollution": Air_Pollution,
    "Occupational_Exposure": Occupational_Exposure,
    "Rural_or_Urban": Rural_or_Urban,
    "Socioeconomic_Status": Socioeconomic_Status,
    "Healthcare_Access": Healthcare_Access,
    "Insurance": Insurance,
    "Screening": Screening,
    "Stage_at_Diagnosis": Stage_at_Diagnosis,
    "Cancer_Type": Cancer_Type,
    "Mutation_Type": Mutation_Type,
    "Treatment": Treatment,
    "Clinical_Trial": Clinical_Trial,
    "Mortality_Risk": Mortality_Risk,
    "5_Year_Survival": Five_Year_Survival,
    "Delay_in_Diagnosis": Delay_in_Diagnosis,
    "Family_History": Family_History,
    "Indoor_Smoking": Indoor_Smoking,
    "Tobacco_Use": Tobacco_Use
}])

#---------------------------------------------------
# Encode Inputs
#---------------------------------------------------
ordinal_features = ['Air_Pollution','Socioeconomic_Status','Healthcare_Access','Stage_at_Diagnosis','Treatment']

label_features = [
    'Gender', 'Second_Hand_Smoke', 'Occupational_Exposure',
    'Rural_or_Urban', 'Insurance', 'Screening',
    'Clinical_Trial', 'Delay_in_Diagnosis', 'Family_History',
    'Indoor_Smoking', 'Tobacco_Use',
]
nominal_features=['Smoking_Status',"Mutation_Type",'Cancer_Type']
numerical_features=['Age','5_Year_Survival','Mortality_Risk']

btn=st.sidebar.button("Predict Lung Cancer Risk")

if btn:




    try:
# columns_order = same as your feature list

# put columns in right order and remove unimportant features
        input_data = input_data.reindex(columns=dummies_cols, fill_value=0)
        

# Transform ordinal features

        input_data[ordinal_features] = ordinalencoder.transform(input_data[ordinal_features])
       


# Transform label features

        for col in label_features:
            input_data[col] = labelencoder[col].transform(input_data[col])

# Transform   nominal data

        input_data=pd.get_dummies(input_data)
        

#---------------------------------------------------
# Scale Data
#---------------------------------------------------

        #input_data = scaler[model].transform(input_data)


#---------------------------------------------------
# Prediction
#---------------------------------------------------

   



        prediction = model.predict(input_data)
        
        # get probability of each class for  -> Final_Prediction label                   1st row(single prediction)for 'yes' value
        probability = model.predict_proba(input_data)[0][1]
       

        # Decode final label to return [Yes ,No]
        final_label = labelencoder['Final_Prediction'].inverse_transform(prediction)

        with st.spinner("Analyzing patient data..."):
            time.sleep(1)

        st.subheader("Prediction Result")
        st.success(f"Prediction: **{final_label}**")
        st.write(f"Risk Probability: **{probability:.2f}**")

        if final_label == "Yes":
            st.warning("High risk detected. Early medical consultation is strongly recommended.")
        else:
            st.info("Low risk detected. Continue preventive healthcare practices.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
