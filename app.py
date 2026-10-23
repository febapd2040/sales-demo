import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Sales Predictor", layout="wide")
st.title("Business Sales Predictor")
st.caption("Demo using 90 SIMULATED records created for a school project.")

@st.cache_data
def load_data():
    return pd.read_csv("business_dataset_simulated_90_rows.csv")

data = load_data()
features = ["Ad_Spend", "Price", "Economy", "Competition"]
X, y = data[features], data["Sales"]

correlation = data.corr(numeric_only=True)

def insight(corr):
    if corr > 0.7:
        return "Strong positive relationship"
    elif corr > 0.3:
        return "Moderate positive relationship"
    elif corr < -0.7:
        return "Strong negative relationship"
    elif corr < -0.3:
        return "Moderate negative relationship"
    else:
        return "Weak or no linear relationship"

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
coefficients = model.named_steps["linearregression"].coef_

tab1, tab2, tab3 = st.tabs(["Predict", "Model", "Data"])

# ---- Tab 1: prediction with sliders ----
with tab1:
    st.subheader("Choose business conditions")
    values = []
    cols = st.columns(len(features))
    for c, f in zip(cols, features):
        low, high = float(data[f].min()), float(data[f].max())
        values.append(c.slider(f, low, high, float(data[f].median())))

    future_input = pd.DataFrame([values], columns=features)
    prediction = model.predict(future_input)[0]
    st.metric("Predicted Sales", f"{prediction:,.2f}")
    st.caption("Estimate from simulated data, not a real-world forecast. "
               "Sliders are limited to the training data range to avoid extrapolation.")

# ---- Tab 2: model quality ----
with tab2:
    c1, c2 = st.columns(2)
    c1.metric("MAE (sales units)", f"{mae:.3f}")
    c2.metric("R²", f"{r2:.3f}")

    fig, ax = plt.subplots()
    ax.bar(features, coefficients)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Coefficient (per 1 std dev)")
    ax.set_title("Regression Coefficients")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm",
                fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

    st.write("Coefficients show associations, not proof of cause and effect. "
             f"Only {len(data)} records, test set of {len(y_test)}, so metrics are noisy.")
    
    st.subheader("Correlation-based business insights")
    for col in features:
        corr_value = correlation.loc[col, "Sales"]
        st.write(f"**{col}** → {insight(corr_value)} (corr = {corr_value:.2f})")

    st.subheader("Model coefficients")
    for feature, coefficient in zip(features, coefficients):
        st.write(f"**{feature}:** {coefficient:.3f}")
    st.caption("A coefficient estimates the change in predicted Sales for a one "
               "standard deviation increase in that feature, while the other "
               "features are held constant. These are associations, not proof "
               "of cause and effect.")
# ---- Tab 3: raw data and scatter plots ----
with tab3:
    st.dataframe(data)
    for f in features:
        fig, ax = plt.subplots()
        ax.scatter(data[f], data["Sales"])
        ax.set_xlabel(f)
        ax.set_ylabel("Sales")
        ax.set_title(f"{f} vs Sales")
        st.pyplot(fig)
