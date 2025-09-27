import streamlit as st
from streamlit_ketcher import st_ketcher
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from mordred import Calculator, descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Author : Dr. Sk. Abdul Amin
# [Details](https://www.scopus.com/authid/detail.uri?authorId=57190176332).
# Date : 28/09/2025
# logo_url = "https://raw.githubusercontent.com/Amincheminform/phKMOi_v1/main/phKMOi_v1_logo.jpg"
logo_url = "https://raw.githubusercontent.com/Amincheminform/Koc-Predictor_v1.0/main/Koc-Predictor.jpg"

calc = Calculator(descriptors, ignore_3D=True)

def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    descriptor_df = calc.pandas([mol])
    return mol, descriptor_df

st.set_page_config(
    page_title="Koc-Predictor",
    layout="wide",
    page_icon=logo_url
)

st.sidebar.image(logo_url)
st.sidebar.success("Thank you for using Koc-Predictor")
model_choice = st.sidebar.selectbox("Select Model", ["Classification Model", "Regression Model"])

# Descriptor columns
classification_descriptor_columns = [
    'BalabanJ', 'AMID', 'ATSC2s', 'nAromAtom', 'ATSC0pe',
    'ETA_shape_x', 'fMF', 'FilterItLogS', 'ATS8se', 'ATSC0dv',
    'nBondsD', 'NddsN', 'SMR', 'TIC1', 'NdO', 'nP', 'TpiPC10',
    'SLogP', 'BCUTs-1h', 'ATSC1are', 'EState_VSA7', 'NdS',
    'ATS5s', 'nG12FRing', 'piPC8', 'ZMIC1'
]

regression_descriptor_columns = [
    'NddsN', 'nP', 'TpiPC10', 'SLogP', 'BCUTs-1h',
    'NdS', 'ATS5s', 'nG12FRing', 'ZMIC1'
]

target_column = 'Experimental'

# Load dataset and train models
classification_model, regression_model, regression_scaler = None, None, None

if model_choice == "Classification Model":
    try:
        #train_data = pd.read_csv(r'C:\Users\Amin\Downloads\Soil work ML\Classification Train.csv')
        #test_data = pd.read_csv(r'C:\Users\Amin\Downloads\Soil work ML\Classification Test.csv')

        train_url = "https://github.com/Amincheminform/Koc-Predictor_v1.0/raw/main/Classification%20Train.csv"
        test_url = "https://github.com/Amincheminform/Koc-Predictor_v1.0/raw/main/Classification%20Test.csv"

        train_data = pd.read_csv(train_url, sep=',')
        test_data = pd.read_csv(test_url, sep=',')

        X_train = train_data[classification_descriptor_columns]
        y_train = train_data['Experimental (Koc)']

        classification_model = RandomForestClassifier(
            n_estimators=20, max_features=0.5, criterion='gini', random_state=42
        )
        classification_model.fit(X_train, y_train)

        y_pred = classification_model.predict(test_data[classification_descriptor_columns])
        acc = accuracy_score(test_data['Experimental (Koc)'], y_pred)
        st.sidebar.success(f"Random Forest Accuracy: {acc:.2f}")
    except Exception as e:
        st.sidebar.error(f"Classification training failed: {e}")

elif model_choice == "Regression Model":
    try:
        #train_data = pd.read_csv(r'C:\Users\Amin\Downloads\Soil work ML\Regression_Train.csv')
        #test_data = pd.read_csv(r'C:\Users\Amin\Downloads\Soil work ML\Regression_Test.csv')

        train_url = "https://github.com/Amincheminform/Koc-Predictor_v1.0/raw/main/Regression_Train.csv"
        test_url = "https://github.com/Amincheminform/Koc-Predictor_v1.0/raw/main/Regression_Test.csv"

        train_data = pd.read_csv(train_url, sep=',')
        test_data = pd.read_csv(test_url, sep=',')

        train_data = train_data.dropna(subset=regression_descriptor_columns + [target_column])
        test_data = test_data.dropna(subset=regression_descriptor_columns + [target_column])

        X_train = train_data[regression_descriptor_columns]
        y_train = train_data[target_column]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        regression_model = LinearRegression()
        regression_model.fit(X_train_scaled, y_train)
        regression_scaler = scaler

        st.sidebar.success(f"Regression model trained with {len(regression_descriptor_columns)} features.")
    except Exception as e:
        st.sidebar.error(f"Regression training failed: {e}")

st.markdown("""
    <h1 style='text-align: center; font-size: 3.2em; color: #2C3E50;'>
        K<sub>OC</sub>-Predictor
    </h1>
""", unsafe_allow_html=True)

st.subheader("Draw or Enter Your Molecule")
drawn_smiles = st_ketcher()
manual_smiles = st.text_input("Or enter SMILES manually:", value=drawn_smiles if drawn_smiles else "")

if manual_smiles:
    st.markdown(f"**Detected SMILES:** `{manual_smiles}`")
    mol, descriptor_df = smiles_to_descriptors(manual_smiles)

    if mol:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(Draw.MolToImage(mol, size=(300, 300)), caption="Compound Structure")

        with col2:
            if descriptor_df is not None:
                if model_choice == "Classification Model" and classification_model is not None:
                    try:
                        desc_ready = descriptor_df[classification_descriptor_columns].dropna(axis=1)
                        if desc_ready.shape[1] != len(classification_descriptor_columns):
                            st.warning("Some descriptors missing; prediction may be less accurate.")
                        y_pred = classification_model.predict(desc_ready)[0]
                        sorption_class = "High sorption to soil" if y_pred == 1 else "Low sorption to soil"
                        color = "red" if y_pred == 1 else "green"
                        st.markdown(
                            f"<h2 style='color:{color};'>Predicted Class: {sorption_class}</h2>",
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.error(f"Classification prediction failed: {e}")

                # Author : Dr. Sk. Abdul Amin
                # [Details](https://www.scopus.com/authid/detail.uri?authorId=57190176332).

                elif model_choice == "Regression Model" and regression_model is not None:
                    try:
                        missing = [f for f in regression_descriptor_columns if f not in descriptor_df.columns]
                        if missing:
                            st.error(f"Missing descriptors for regression: {missing}")
                        else:
                            desc_ready = descriptor_df[regression_descriptor_columns]
                            desc_scaled = regression_scaler.transform(desc_ready)
                            y_pred = regression_model.predict(desc_scaled)[0]

                            import math
                            koc1 = 10 ** y_pred if not math.isnan(y_pred) else float('nan')

                            result_df = pd.DataFrame({
                                'Property': ['Predicted Log10(Koc)', 'Predicted Koc (Lit/Kg)'],
                                'Value': [round(y_pred, 2), round(koc1, 2)]
                            })

                            st.subheader("Prediction Summary")
                            st.table(result_df)
                    except Exception as e:
                        st.error(f"Regression prediction failed: {e}")
            else:
                st.warning("Descriptor calculation returned None.")
    else:
        st.error("Invalid SMILES. Please check your input.")
else:
    st.info("Please draw or input a SMILES to begin prediction.")

# Author : Dr. Sk. Abdul Amin
# [Details](https://www.scopus.com/authid/detail.uri?authorId=57190176332).
# Contact section
with st.expander("Contact", expanded=False):
    st.write('''
        #### Report an Issue

        Report a bug or contribute here: [GitHub](https://github.com/Amincheminform)

        #### Contact Us
        - [Dr. Supratik Kar](mailto:skar@kean.edu)

        - [Dr. Sk. Abdul Amin](mailto:pharmacist.amin@gmail.com)
    ''')
