import pandas as pd
import streamlit as st
from rapidfuzz import process  # replaced fuzzywuzzy with rapidfuzz
import google.generativeai as genai
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ---------- Gemini API Setup ----------
GEMINI_API_KEY = "AIzaSyCht9ImA8fwan8aw99gLTnyT9k86ewmdH8"
genai.configure(api_key=GEMINI_API_KEY)

def get_gemini_analysis(car_name):
    prompt = f"""
    I'm considering buying a used car named '{car_name}'.

    Please provide a detailed analysis including:
    - Is this car considered fast?
    - What modifications or upgrades can make it faster?
    - Are there better or faster options available in the same price range?
    - Is it safe to drive fast with this car considering its build and engine?

    Be clear, helpful, and easy to understand.
    """
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini API Error: {e}"

# ---------- Load and Prepare CSV ----------
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("usedCarsFinal.csv")
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    df["engine_capacity"] = df["engine_capacity"].astype(str).str.extract(r"(\d{3,5})")
    df["engine_capacity"] = pd.to_numeric(df["engine_capacity"], errors="coerce")
    df = df.dropna(subset=["name", "engine_capacity"])

    df["fast"] = df["engine_capacity"].apply(lambda x: 1 if x > 1600 else 0)
    return df

# ---------- ML Model Training ----------
@st.cache_resource
def train_logistic_model(df):
    X = df[["engine_capacity"]]
    y = df["fast"]
    model = LogisticRegression()
    model.fit(X, y)
    return model

# ---------- Match Car Name ----------
def get_similar_car_name(df, user_input):
    user_input_lower = user_input.lower()
    names = df["name"].dropna().tolist()
    names_lower = [name.lower() for name in names]
    match = process.extractOne(user_input_lower, names_lower)
    if match and match[1] >= 60:
        matched_index = names_lower.index(match[0])
        return names[matched_index]
    return None

# ---------- Analyze from Database ----------
def analyze_from_database(df, car_name):
    matched_name = get_similar_car_name(df, car_name)
    if not matched_name:
        return None, "❌ No matching car found in database."

    df_filtered = df[df["name"] == matched_name]

    def get_recommendation(cc):
        if cc > 2200:
            return "Already fast"
        elif cc > 1600:
            return "Mods can make it faster"
        else:
            return "Normal performance"

    df_filtered["Recommendation"] = df_filtered["engine_capacity"].apply(get_recommendation)
    df_filtered["Fast"] = df_filtered["fast"].map({1: "Yes", 0: "No"})
    return df_filtered[["name", "engine_capacity", "Fast", "Recommendation"]], None

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Car Performance Analyzer", layout="centered")
st.title("Car Performance Analyzer")

df = load_and_prepare_data()
model = train_logistic_model(df)

# --- Input Field ---
car_name_input = st.text_input("Enter Car Name", "")

# --- Buttons Layout ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    db_btn = st.button("Analyze with Database")
with col2:
    ai_btn = st.button("Analyze with Gemini AI")
with col3:
    both_btn = st.button("Analyze with Both")
with col4:
    ml_btn = st.button("Predict with AI Model")

# --- Logic Handling ---
if car_name_input.strip():
    car_name_input = car_name_input.strip()

    # Database Only
    if db_btn:
        result_df, err = analyze_from_database(df, car_name_input)
        if err:
            st.warning(err)
        else:
            st.success("Database Result:")
            st.dataframe(result_df)

    # AI Only
    elif ai_btn:
        with st.spinner("Getting AI Response..."):
            ai_output = get_gemini_analysis(car_name_input)
        st.subheader("Gemini AI Response")
        st.markdown(ai_output)

    # Both
    elif both_btn:
        result_df, err = analyze_from_database(df, car_name_input)
        if err:
            st.warning("Car not found in dataset. Proceeding with Gemini AI only.")
        else:
            st.success("Database Result:")
            st.dataframe(result_df)

        with st.spinner("Getting AI Response..."):
            ai_output = get_gemini_analysis(car_name_input)
        st.subheader("Gemini AI Response")
        st.markdown(ai_output)

    # ML Model
    elif ml_btn:
        matched_name = get_similar_car_name(df, car_name_input)
        if not matched_name:
            st.warning("Car not found in dataset. ML prediction skipped.")
        else:
            engine_cc = df[df["name"] == matched_name]["engine_capacity"].mean()
            prediction = model.predict([[engine_cc]])[0]
            st.success(f"ML Prediction based on engine capacity ({engine_cc:.0f} cc):")
            st.markdown(f"**Fast:** {'Yes' if prediction else 'No'}")

else:
    st.info("Please enter a car name to begin.")
