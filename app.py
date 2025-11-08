import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

st.set_page_config(
    page_title="Steel Industry Load Type Predictor",
    page_icon="⚙️",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load('best_model_lightgbm.pkl')
    scaler = joblib.load('standard_scaler.pkl')
    metadata = joblib.load('model_metadata.pkl')
    return model, scaler, metadata

def prepare_features(usage_kwh, lagging_reactive, leading_reactive, co2, 
                     lagging_pf, leading_pf, date_time, week_status, day_of_week, scaler, metadata):
    feature_names = metadata.get('feature_names', [])
    
    total_reactive = lagging_reactive + leading_reactive
    pf_diff = leading_pf - lagging_pf
    power_efficiency = usage_kwh / (total_reactive + 1) if total_reactive > 0 else usage_kwh
    
    nsm = date_time.hour * 3600 + date_time.minute * 60
    day_of_year = date_time.timetuple().tm_yday
    week_of_year = date_time.isocalendar()[1]
    quarter = (date_time.month - 1) // 3 + 1
    
    hour_sin = np.sin(2 * np.pi * date_time.hour / 24)
    hour_cos = np.cos(2 * np.pi * date_time.hour / 24)
    month_sin = np.sin(2 * np.pi * date_time.month / 12)
    month_cos = np.cos(2 * np.pi * date_time.month / 12)
    
    weekstatus_encoded = 1 if week_status == 'Weekend' else 0
    
    day_features = {
        'Monday': {'Day_of_week_Monday': 1},
        'Tuesday': {'Day_of_week_Tuesday': 1},
        'Wednesday': {'Day_of_week_Wednesday': 1},
        'Thursday': {'Day_of_week_Thursday': 1},
        'Friday': {},
        'Saturday': {'Day_of_week_Saturday': 1},
        'Sunday': {'Day_of_week_Sunday': 1}
    }
    
    if usage_kwh < 3.20:
        intensity_features = {'Usage_Intensity_Low': 0, 'Usage_Intensity_Medium': 0, 'Usage_Intensity_High': 0}
    elif usage_kwh < 4.57:
        intensity_features = {'Usage_Intensity_Low': 1, 'Usage_Intensity_Medium': 0, 'Usage_Intensity_High': 0}
    elif usage_kwh < 51.24:
        intensity_features = {'Usage_Intensity_Low': 0, 'Usage_Intensity_Medium': 1, 'Usage_Intensity_High': 0}
    else:
        intensity_features = {'Usage_Intensity_Low': 0, 'Usage_Intensity_Medium': 0, 'Usage_Intensity_High': 1}
    
    features = {
        'Usage_kWh': usage_kwh,
        'Lagging_Current_Reactive.Power_kVarh': lagging_reactive,
        'Leading_Current_Reactive_Power_kVarh': leading_reactive,
        'CO2(tCO2)': co2,
        'Lagging_Current_Power_Factor': lagging_pf,
        'Leading_Current_Power_Factor': leading_pf,
        'NSM': nsm,
        'year': date_time.year,
        'month': date_time.month,
        'day': date_time.day,
        'hour': date_time.hour,
        'minute': date_time.minute,
        'day_of_year': day_of_year,
        'week_of_year': week_of_year,
        'quarter': quarter,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'Total_Reactive_Power': total_reactive,
        'Power_Factor_Difference': pf_diff,
        'Power_Efficiency': power_efficiency,
        'Usage_kWh_lag1': usage_kwh,
        'Usage_kWh_lag2': usage_kwh,
        'Usage_kWh_rolling_mean_3': usage_kwh,
        'Usage_kWh_rolling_std_3': 0.0,
        'WeekStatus_encoded': weekstatus_encoded,
        'Day_of_week_Monday': 0,
        'Day_of_week_Tuesday': 0,
        'Day_of_week_Wednesday': 0,
        'Day_of_week_Thursday': 0,
        'Day_of_week_Saturday': 0,
        'Day_of_week_Sunday': 0,
        **day_features.get(day_of_week, {}),
        **intensity_features
    }
    
    df = pd.DataFrame({name: [features.get(name, 0.0)] for name in feature_names})
    
    if hasattr(scaler, 'feature_names_in_'):
        scaler_features = list(scaler.feature_names_in_)
    else:
        scaler_features = []
    
    df_scaled = df.copy()
    if scaler_features and len(scaler_features) > 0:
        df_scaled[scaler_features] = scaler.transform(df[scaler_features])
    
    return df_scaled

def main():
    st.title("⚙️ Steel Industry Load Type Predictor")
    st.markdown("Predict load type using the trained LightGBM model")
    
    try:
        model, scaler, metadata = load_model()
    except ModuleNotFoundError as e:
        st.error(f"❌ Missing required package: {e}")
        st.info("""
        **Please install the required dependencies:**
        
        ```bash
        pip install lightgbm streamlit
        ```
        
        Or install all requirements:
        ```bash
        pip install -r requirements.txt
        ```
        """)
        st.stop()
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.info("""
        **Required files:**
        - `best_model_lightgbm.pkl`
        - `standard_scaler.pkl`
        - `model_metadata.pkl`
        
        Make sure these files are in the same directory as `app.py`
        """)
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.exception(e)
        st.stop()
    
    with st.sidebar:
        st.header("📊 Model Info")
        st.metric("Accuracy", "99.97%")
        st.metric("Model", "LightGBM")
        
    st.header("Input Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        usage_kwh = st.number_input("Usage (kWh)", min_value=0.0, max_value=200.0, value=4.57, step=0.1)
        lagging_reactive = st.number_input("Lagging Reactive Power (kVarh)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
        leading_reactive = st.number_input("Leading Reactive Power (kVarh)", min_value=0.0, max_value=30.0, value=0.0, step=0.1)
        co2 = st.number_input("CO2 (tCO2)", min_value=0.0, max_value=0.1, value=0.0, step=0.001)
        
    with col2:
        lagging_pf = st.number_input("Lagging Power Factor", min_value=0.0, max_value=100.0, value=87.96, step=0.01)
        leading_pf = st.number_input("Leading Power Factor", min_value=0.0, max_value=100.0, value=100.0, step=0.01)
        
        if 'selected_date' not in st.session_state:
            st.session_state.selected_date = datetime.now().date()
        if 'selected_time' not in st.session_state:
            st.session_state.selected_time = datetime.now().time()
        
        date_input = st.date_input("Date", value=st.session_state.selected_date, key='date_input')
        time_input = st.time_input("Time", value=st.session_state.selected_time, key='time_input')
        
        st.session_state.selected_date = date_input
        st.session_state.selected_time = time_input
        
        date_time = datetime.combine(date_input, time_input)
        
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        current_day_index = date_time.weekday()
        
        week_status = st.selectbox("Week Status", ["Weekday", "Weekend"], 
                                   index=1 if date_time.weekday() >= 5 else 0)
        day_of_week = st.selectbox("Day of Week", day_names, index=current_day_index)
    
    if st.button("🚀 Predict Load Type", type="primary"):
        try:
            features_df = prepare_features(
                usage_kwh, lagging_reactive, leading_reactive, co2,
                lagging_pf, leading_pf, date_time, week_status, day_of_week, scaler, metadata
            )
            
            prediction = model.predict(features_df)[0]
            probabilities = model.predict_proba(features_df)[0]
            
            class_names = ['Light_Load', 'Maximum_Load', 'Medium_Load']
            predicted_class = class_names[prediction]
            confidence = probabilities[prediction] * 100
            
            st.success("Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 0:
                    st.metric("Predicted Load Type", "🔵 Light Load", f"{confidence:.2f}%")
                elif prediction == 1:
                    st.metric("Predicted Load Type", "🔴 Maximum Load", f"{confidence:.2f}%")
                else:
                    st.metric("Predicted Load Type", "🟡 Medium Load", f"{confidence:.2f}%")
            
            with col2:
                st.metric("Confidence", f"{confidence:.2f}%")
            
            with col3:
                st.metric("Model Accuracy", "99.97%")
            
            st.subheader("Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Load Type': class_names,
                'Probability (%)': [f"{p*100:.2f}" for p in probabilities]
            })
            st.bar_chart({name: [prob] for name, prob in zip(class_names, probabilities)})
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()
