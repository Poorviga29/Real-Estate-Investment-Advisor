
# Imported the Libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings("ignore")


#  1. Database Connection

engine = create_engine("mysql+pymysql://root:Mrbean%40123456789d@localhost:3306/realestate_db")

# Load cleaned (for display) and scaled (for model) data
df_cleaned = pd.read_sql("SELECT * FROM realestate_cleaneddata", engine)
df_scaled = pd.read_sql("SELECT * FROM realestate_investment", engine)

#  2. Streamlit Page Setup

st.set_page_config(page_title="🏠 Real Estate Investment Advisor", layout="wide")

st.title("🏠 Real Estate Investment Advisor")
# st.markdown("#### Empowering smarter property investment decisions with AI-driven insights.")
st.markdown("#### Predicting Property Profitability & Future Value")
st.divider()

#  3. Sidebar Filters (Rebranded)

st.sidebar.header("🏡 Smart Investment Finder")
st.sidebar.caption("Refine your search to discover the most profitable opportunities.")

st.sidebar.subheader("📍 Location")
selected_state = st.sidebar.selectbox("Select State", sorted(df_cleaned["State"].unique()))
filtered_city = df_cleaned[df_cleaned["State"] == selected_state]["City"].unique()
selected_city = st.sidebar.selectbox("Select City", sorted(filtered_city))

st.sidebar.subheader("💰 Price & Property Type")
bhk_options = sorted(df_cleaned["BHK"].unique())
bhk_filter = st.sidebar.multiselect("BHK Options", bhk_options, default=[bhk_options[0]])

price_range = st.sidebar.slider(
    "Price Range (Lakhs)",
    int(df_cleaned["Price_in_Lakhs"].min()),
    int(df_cleaned["Price_in_Lakhs"].max()),
    (50, 500)
)

filtered_df = df_cleaned[
    (df_cleaned["State"] == selected_state)
    & (df_cleaned["City"] == selected_city)
    & (df_cleaned["BHK"].isin(bhk_filter))
    & (df_cleaned["Price_in_Lakhs"].between(price_range[0], price_range[1]))
]


#  4. Summary KPI Cards

st.markdown("### 📊 Market Overview")

col1, col2, col3 = st.columns(3)
col1.metric("🏘️ Total Listings", len(filtered_df))
col2.metric("💰 Avg. Price (Lakhs)", f"{filtered_df['Price_in_Lakhs'].mean():.2f}")
col3.metric("📐 Avg. Size (SqFt)", f"{filtered_df['Size_in_SqFt'].mean():.0f}")


#  5. Filtered Property Table + Download

st.write(f"### 🏘️ Property Insights in {selected_city} ({len(filtered_df)})")

columns_to_show = [
    "State",
    "City",
    "Locality",
    "Property_Type",
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Year_Built",
    "Total_Floors",
    "Parking_Space",
    "Furnished_Status",
    "Owner_Type",
    "Availability_Status",
]

filtered_display = filtered_df[columns_to_show].rename(
    columns={
        "Size_in_SqFt": "Size (SqFt)",
        "Price_in_Lakhs": "Price (₹ Lakhs)",
        "Year_Built": "Built Year",
        "Furnished_Status": "Furnishing",
    }
)

st.data_editor(
    filtered_display,
    use_container_width=True,
    hide_index=True,
    num_rows="dynamic",
)

# Download button
csv_data = filtered_display.to_csv(index=False).encode("utf-8")
st.download_button(
    label=" Download ",
    data=csv_data,
    file_name=f"Property_List_{selected_city}.csv",
    mime="text/csv",
)


#  6. Visualization Section

st.markdown("### 📈 Investment Trends & Insights")

col1, col2 = st.columns(2)

with col1:
    
    st.markdown("####  Correlation Heatmap")

    if len(filtered_df) >= 50:
        corr = filtered_df.select_dtypes(include=np.number).corr()
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(corr, cmap="YlGnBu", annot=False, ax=ax)
        st.pyplot(fig)
    elif len(filtered_df) > 1:
        st.info("⚠️ Not enough data to generate a reliable heatmap. Showing overall dataset correlation instead.")
        corr = df_cleaned.select_dtypes(include=np.number).corr()
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(corr, cmap="YlOrBr", annot=False, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No sufficient numeric data available for correlation analysis.")


with col2:
    st.markdown("####   Average Price per BHK ")
    if len(filtered_df) > 0:
        avg_price = (
            filtered_df.groupby("BHK")["Price_in_Lakhs"]
            .mean()
            .reset_index()
            .sort_values("BHK")
        )
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.barplot(data=avg_price, x="BHK", y="Price_in_Lakhs", palette="viridis", ax=ax2)
        ax2.set_xlabel("BHK")
        ax2.set_ylabel("Average Price (₹ Lakhs)")
        st.pyplot(fig2)
    else:
        st.warning("No data available for the selected filters.")



# 📊 State-wise & Property Type Price Comparison

st.markdown("### 💹 Price Comparison Insights")

col1, col2 = st.columns(2)

#  Chart 1: State-wise Price Comparison 
with col1:
    st.markdown("####  State-wise Price Comparison")
    avg_state_price = (
        df_cleaned.groupby("State")["Price_in_Lakhs"]
        .mean()
        .reset_index()
        .sort_values(by="Price_in_Lakhs", ascending=False)
    )

    fig_state, ax_state = plt.subplots(figsize=(5, 3))
    sns.barplot(
        data=avg_state_price,
        x="Price_in_Lakhs",
        y="State",
        palette="crest",
        ax=ax_state
    )
    ax_state.set_xlabel("Average Price (₹ Lakhs)", fontsize=9)
    ax_state.set_ylabel("State", fontsize=9)
    ax_state.tick_params(axis='x', labelsize=8)
    ax_state.tick_params(axis='y', labelsize=8)
    st.pyplot(fig_state, use_container_width=True)

#  Chart 2: Property Type vs Price 
with col2:
    st.markdown("#### Property Type vs Price Comparison")
    avg_property_price = (
        df_cleaned.groupby("Property_Type")["Price_in_Lakhs"]
        .mean()
        .reset_index()
        .sort_values(by="Price_in_Lakhs", ascending=False)
    )

    fig_prop, ax_prop = plt.subplots(figsize=(5, 3))
    sns.barplot(
        data=avg_property_price,
        x="Price_in_Lakhs",
        y="Property_Type",
        palette="viridis",
        ax=ax_prop
    )
    ax_prop.set_xlabel("Average Price (₹ Lakhs)", fontsize=9)
    ax_prop.set_ylabel("Property Type", fontsize=9)
    ax_prop.tick_params(axis='x', labelsize=8)
    ax_prop.tick_params(axis='y', labelsize=8)
    st.pyplot(fig_prop, use_container_width=True)


st.divider()


#  7. Appreciation Calculator

st.markdown("### 🏠💹 Property Appreciation Estimator")

col1, col2, col3 = st.columns(3)
base_price = col1.number_input("Current Price (₹ Lakhs)", min_value=10.0, value=75.0)
rate = col2.slider("Expected Annual Growth (%)", 2.0, 15.0, 8.0)
years = col3.slider("Years", 1, 20, 5)

future_value = base_price * ((1 + rate / 100) ** years)
st.success(f" Estimated Price after {years} years: ₹ {future_value:.2f} Lakhs")

st.divider()


#  8. MLflow Model Loading

st.markdown("### 🤖 AI-Powered Investment Prediction")

try:
    model_reg = mlflow.sklearn.load_model("models:/Best_Regression_Model/Production")
    model_cls = mlflow.sklearn.load_model("models:/Best_Classification_Model/Production")
    st.success(" Models loaded successfully from MLflow Registry!")
except Exception as e:
    model_reg = None
    model_cls = None
    st.error(f" Model loading failed: {e}")


#  9. Prediction Input Form

st.markdown("### 🧾 Investment Prediction")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    input_bhk = col1.selectbox("BHK", sorted(df_cleaned["BHK"].unique()))
    input_size = col2.number_input("Size (SqFt)", min_value=300, max_value=10000, value=1200)
    input_price = col3.number_input("Current Price (Lakhs)", min_value=10.0, max_value=2000.0, value=75.0)

    col4, col5, col6 = st.columns(3)
    input_age = col4.slider("Age of Property (Years)", 0, 50, 5)
    input_amenities = col5.slider("Amenities Score", 0.0, 10.0, 5.0)
    input_infra = col6.slider("Infrastructure Score", 0.0, 10.0, 6.0)

    submitted = st.form_submit_button("🔮 Predict Investment Potential")

if submitted:
    if model_reg and model_cls:
        model_features = df_scaled.drop(columns=["Good_Investment", "Future_Price_5Y"], errors="ignore").columns
        input_full = pd.DataFrame([df_scaled[model_features].mean()])

        replacements = {
            "BHK": input_bhk,
            "Size_in_SqFt": input_size,
            "Price_in_Lakhs": input_price,
            "Age_of_Property": input_age,
            "Amenities": input_amenities,
            "Infrastructure_Score": input_infra,
        }
        for k, v in replacements.items():
            if k in input_full.columns:
                input_full.at[0, k] = v

        # Predictions
        price_future = model_reg.predict(input_full)[0]
        invest_pred = model_cls.predict(input_full)[0]
        invest_prob = model_cls.predict_proba(input_full)[0][1] if hasattr(model_cls, "predict_proba") else 0

        st.success(f"💸 **Estimated Price after 5 Years:** ₹ {price_future:.2f} Lakhs")
        st.info(f"🏆 **Good Investment?** {' Yes' if invest_pred == 1 else ' No'}")
        st.progress(float(invest_prob))

        # AI Insight
        st.markdown("### AI Investment Insight")
        insight = f"""
        For a {input_bhk}-BHK property in **{selected_city}**, priced at ₹{input_price} Lakhs,  
        the estimated 5-year future price is **₹{price_future:.2f} Lakhs**.  
        Given the amenities score of {input_amenities} and infrastructure score of {input_infra},  
        this property shows **{'strong' if invest_pred == 1 else 'moderate'} potential** for investment growth.
        """
        st.info(insight)

        # Feature Importance
        st.markdown("### 🔍 Feature Importance ")
        try:
            if hasattr(model_cls, "feature_importances_"):
                importance = pd.Series(model_cls.feature_importances_, index=model_features).sort_values(ascending=False)
                fig3, ax3 = plt.subplots(figsize=(5, 3))
                sns.barplot(x=importance.values[:10], y=importance.index[:10], palette="viridis", ax=ax3)
                ax3.set_title("Top 10 Influential Features")
                st.pyplot(fig3, use_container_width=False)
        except Exception:
            st.warning(" Could not plot feature importance.")
    else:
        st.warning(" Models not loaded. Please train and register models in MLflow first.")


#  10. Sidebar Feature Information

st.sidebar.markdown("---")
st.sidebar.markdown("### 📘 Feature Descriptions")
feature_info = {
    "BHK": "Number of Bedrooms, Hall, Kitchen",
    "Size_in_SqFt": "Total built-up area of the property",
    "Price_in_Lakhs": "Current market price in Lakhs",
    "Age_of_Property": "Years since construction",
    "Amenities": "Quality and number of amenities (pool, gym, parks, etc.)",
    "Infrastructure_Score": "Surrounding infrastructure quality (roads, schools, transport)",
}
for feature, desc in feature_info.items():
    st.sidebar.markdown(f"**{feature}** – {desc}")

st.sidebar.info("💡 Tip: Adjust filters and sliders to explore market trends dynamically!")

