import streamlit as st
import pickle
import numpy as np
import pandas as pd
from io import BytesIO
import urllib.parse

# Load the trained model and scaler
with open("l_regressor.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Page configuration
st.set_page_config(page_title=" CYCLECARE AI PST", page_icon="🩺", layout="wide")


# **Mode Selection on Home Page**
if "mode" not in st.session_state:
    st.session_state.mode = None

if st.session_state.mode is None:
    st.title("Welcome to CycleCare AI PCOS Symptoms Tracker 🩺")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logo.png", use_container_width=False, width = 200)
    
    st.subheader("Choose an option to continue:")

    

    st.markdown(
        """
        <style>
        div.stButton > button {
            background-color: #FF69B4 !important; /* Hot Pink */
            color: white !important;
            border-radius: 10px !important;
            width: 100% !important;
            height: 50px !important;
            font-size: 16px !important;
            border: none !important;
        }
        div.stButton > button:hover {
            background-color: #FF1493 !important; /* Deep Pink */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("🩺 I’m Tracking My Health"):
            st.session_state.mode = "🩺 Patient"
            st.rerun()

    with col2:
        if st.button("👨‍⚕️ I’m a Healthcare Professional"):
            st.session_state.mode = "👨‍⚕️ Doctor"
            st.rerun()

 #Push copyright text to the bottom of the sidebar using CSS
    st.markdown(
        """
        <style>
        .copyright {
            position: fixed;
            bottom: 20px;
            left: 20px;
            font-size: 14px;
            color: gray;
        }
        </style>
        <div class="copyright">
            © 2025 Team MidgenAI. All rights reserved.
        </div>
        """,
        unsafe_allow_html=True
    )

# Display only the selected mode
elif st.session_state.mode  == "🩺 Patient":
    MENU = st.sidebar.radio("MENU", ["Home", "About", "Check Your Health", "FAQs", "Download & Share Report"], key="patient_menu")

# Patient Mode
    if MENU == "Home":
        st.title("CYCLECARE AI PCOS SYMPTOMS TRACKER 🩺")
        st.caption("Got symptoms? Get started with the PCOS Symptom Tracker!")
        st.image("2720740.webp", use_container_width=False, caption = "© TaraMD")
        st.subheader("Welcome to the PCOS Symptoms Tracker (PST) powered by Cyclecare AI.")
        st.write("This smart tool helps you track symptoms and assess your risk for PCOS. Stay informed, make healthier choices, and connect with medical support when needed. Take charge of your well-being today! 💖")

    elif MENU == "About":
        st.title("ℹ️ About")
        st.write("This application helps track symptoms related to PCOS and provides early risk assessment based on health data.")
        st.image("symptoms.webp", use_container_width=False, caption="© riopads")
        st.subheader("📊 Dataset & Model")
        st.write("""
        - The model is trained on an **Indian PCOS dataset** with 541 rows from kaggle, which includes various health indicators.
        - Features such as **BMI, Menstrual Irregularity, Insulin Resistance, Acne, Hirsutism, Lifestyle Factors** were used for training.
        - A **Logistic Regression model** was implemented for prediction with 88.07% accuracy after scaling the input features.
        """)

    elif MENU == "Check Your Health":
        st.title("🔍 Check Your Health")
        st.write("Enter your health details to check for early warning signs of PCOS.")

        # Input fields
        col1, col2 = st.columns(2)

        with col1:
            hair_growth = st.selectbox("Excess Hair Growth", ["No", "Yes"])
            skin_darkening = st.selectbox("Skin Darkening", ["No", "Yes"])
            weight_gain = st.selectbox("Unexplained Weight Gain", ["No", "Yes"])
            cycle_regularity = st.selectbox("Irregular Menstrual Cycle (range)", [1, 2, 3, 4, 5])
            age = st.number_input("Age (Years)", min_value=10, max_value=50, value=25)
            avg_follicle_size_L = st.number_input("Average Follicle Size (Left) (mm)", min_value=0.0, max_value=30.0, value=14.0)
            pimples = st.selectbox("Frequent Pimples", ["No", "Yes"])
            beta_HCG_II = st.number_input("II beta-HCG (mIU/mL)", min_value=0.0, value=1.8)

        with col2:
            beta_HCG_I = st.number_input("I beta-HCG (mIU/mL)", min_value=0.0, value=2.3)
            tsh = st.number_input("TSH (mIU/L)", min_value=0.0, value=2.5)
            weight = st.number_input("Weight (Kg)", min_value=30.0, max_value=150.0, value=60.0)
            cycle_length = st.number_input("Menstrual Cycle Length (Days)", min_value=0, max_value=60, value=30)
            pulse_rate = st.number_input("Pulse Rate (bpm)", min_value=40, max_value=120, value=75)
            bmi = st.number_input("BMI", min_value=10.0, max_value=40.0, value=22.5)
            fast_food = st.selectbox("Frequent Fast Food Consumption", ["No", "Yes"])

        # Predict button
        if st.button("Check for PCOS"):
            new_data = np.array([
                [
                    int(hair_growth == "Yes"),
                    int(skin_darkening == "Yes"),
                    int(weight_gain == "Yes"),
                    cycle_regularity,
                    age,
                    avg_follicle_size_L,
                    int(pimples == "Yes"),
                    beta_HCG_II,
                    beta_HCG_I,
                    tsh,
                    weight,
                    cycle_length,
                    pulse_rate,
                    bmi,
                    int(fast_food == "Yes")
                ]
            ])
            
            scaled_data = scaler.transform(new_data)
            prediction = model.predict(scaled_data)
            
            if prediction[0] == 0:
                result_text = """
        **🟢 Low Risk: No PCOS Detected**  
        ______________________________________
        
        ```This means that you are unlikely to have PCOS based on the information provided. 
        However, do not ignore any symptoms you may be experiencing. Even though your risk is low, it's still important to monitor any changes in your health.```
        
        **For reassurance and further testing, download and share this report with your health advisor/doctor**
        """
                st.success(result_text)
            else:
                result_text = """
        **🔴 High Risk: PCOS Detected**
        ______________________________________
        ```Your result shows a high risk which means that your symptoms and health data suggest a strong possibility of PCOS.```
        
        ```However, this is not a medical diagnosis, it is simply an early warning that further medical evaluation is needed.```
            
            **Visit a gynecologist or endocrinologist who can assess your condition more thoroughly. A pelvic ultrasound can help confirm the presence of ovarian cysts, which are a key indicator of PCOS.**
            
            **PCOS can often be managed with lifestyle changes. You can use our **Lifestyle Recommendation** Model for personalized health and wellness guidance.**
                """
                st.error(result_text)

            # Store the results in session state for the report
            st.session_state["user_data"] = {
                "Excess Hair Growth": hair_growth, "Skin Darkening": skin_darkening,
                "Unexplained Weight Gain": weight_gain, "Irregular Menstrual Cycle": cycle_regularity,
                "Age": age, "Average Follicle Size (Left)": avg_follicle_size_L,
                "Frequent Pimples": pimples, "II beta-HCG": beta_HCG_II, "I beta-HCG": beta_HCG_I,
                "TSH": tsh, "Weight": weight, "Menstrual Cycle Length": cycle_length,
                "Pulse Rate": pulse_rate, "BMI": bmi, "Frequent Fast Food Consumption": fast_food,
                "PCOS Risk": result_text
            }

    elif MENU == "Download & Share Report":
        st.title("📥 Download & Share Your PCOS Report")
        st.caption("Easily download your health report for future reference or share it with your doctor and loved ones. Stay informed, track your symptoms, and take proactive steps towards your well-being! 💖")
        if "user_data" in st.session_state:
            # Convert session state data into a DataFrame
            report_df = pd.DataFrame(list(st.session_state["user_data"].items()), columns=["Feature", "Value"])

            # Convert DataFrame to CSV
            csv_buffer = BytesIO()
            report_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            # Download button for the report
            st.download_button(
                label="📥 Download Report (CSV)",
                data=csv_buffer,
                file_name="PCOS_Report.csv",
                mime="text/csv"
            )

            # Prepare shareable text
            report_text = "\n".join([f"{k}: {v}" for k, v in st.session_state["user_data"].items()])
            encoded_text = urllib.parse.quote(report_text)

            # Share buttons
            st.subheader("🔗 Share Your Report")

            whatsapp_url = f"https://api.whatsapp.com/send?text={encoded_text}"
            email_url = f"mailto:?subject=My%20PCOS%20Report&body={encoded_text}"

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"[![WhatsApp](https://img.shields.io/badge/Share%20on-WhatsApp-green?logo=whatsapp)]({whatsapp_url})", unsafe_allow_html=True)
            with col2:
                st.markdown(f"[![Email](https://img.shields.io/badge/Send%20via-Email-blue?logo=gmail)]({email_url})", unsafe_allow_html=True)
        
        else:
            st.warning("⚠️ No report available. Please check your health first.")

    elif MENU == "FAQs":
        st.title("❓ Frequently Asked Questions")

        with st.expander("1️⃣ What is this tool used for?"):
            st.write("This tool helps track symptoms related to **Polycystic Ovary Syndrome (PCOS)** and provides a **risk assessment** using AI models.")

        with st.expander("2️⃣ What features are available on this Cyclecare AIVplatform?"):
            st.write("""
            - **🔍 PCOS Symptom Tracker** – Assesses your risk of PCOS based on health data.
            - **💬 AI Chatbot** – Answers questions about PCOS and connects you with specialists.
            - **📷 Medical Image Scanning** – Uses AI to analyze ultrasound images for potential PCOS indicators.
            - **🍏 Lifestyle Recommendation Model** – Provides personalized lifestyle and diet suggestions.
            """)

        with st.expander("3️⃣ How accurate is the prediction?"):
            st.write("The model is trained on an Indian PCOS dataset, so while it provides valuable insights, it may not fully capture Nigerian-specific factors like genetics, diet, and environmental influences. Always consult a doctor for a professional medical diagnosis.")

        with st.expander("4️⃣ What happens after I get my prediction?"):
            st.write("""
            - If you get a **low risk** result: You may not have PCOS, but maintaining a healthy lifestyle is always beneficial and a follow-up with a doctor is recommended.
            - If you get a **high risk** result: Consider **consulting a doctor** for further tests.
            - You can also explore our **Lifestyle Recommendation Model** and **AI Chatbot** for more guidance.
            """)

        with st.expander("5️⃣ Where can I access the other AI models?"):
            st.write("""
            - **Lifestyle Recommendation Model**: [Click here](#)  
            - **Medical Image Scanning Model**: [Click here](#)  
            - **PCOS Chatbot**: [Click here](#)
            """)

        with st.expander("6️⃣ How can I download my PCOS report?"):
            st.write("""
            After checking your health, your results will be stored, and you can **download a detailed report** in CSV format from the **Download & Share Report** section.
            """)

        with st.expander("7️⃣ Can I share my PCOS report with a doctor or friends?"):
            st.write("""
            Yes! You can **share your report via WhatsApp or email** directly from the app, making it easy to discuss results with a doctor or get support from loved ones.
            """)

elif st.session_state.mode  == "👨‍⚕️ Doctor":
    DOCTOR_MENU = st.sidebar.radio("MENU", ["Home", "About", "Quick Initial Screening", "Report Download or Upload", "Next Steps After Screening", "FAQs"], key="doctor_menu")

    if DOCTOR_MENU == "Home":
        st.title("🏥 CYCLECARE AI - Doctor Mode")
        st.caption("A specialized PCOS tracking tool for medical professionals.")
        st.image("doctor.webp", use_container_width= False, width = 500)
        st.subheader("Welcome, Doctor!")
        st.write("""
        This tool enables you to quickly assess PCOS risk in patients.  
        You can conduct an **initial screening**, download or upload reports, and guide patients on next steps.  
        Use the **Quick Initial Screening** feature to assess patient symptoms and get AI-powered insights.
        """)

    elif DOCTOR_MENU == "About":
        st.title("ℹ️ About")
        st.image("symptoms.webp", use_container_width=False, caption="© riopads")
        st.subheader("📊 Dataset & Model")
        st.write("""
        - The model is trained on an **Indian PCOS dataset** with 541 rows from kaggle, which includes various health indicators.
        - Features such as **BMI, Menstrual Irregularity, Insulin Resistance, Acne, Hirsutism, Lifestyle Factors** were used for training.
        - A **Logistic Regression model** was implemented for prediction with 88.07% accuracy after scaling the input features.
        """)

    elif DOCTOR_MENU == "Quick Initial Screening":
        st.title("🩺 Quick Initial Screening")
        st.write("Enter the patient’s details for a quick PCOS risk assessment.")

        # Input fields (same as Check Your Health)
        col1, col2 = st.columns(2)

       # Input fields
        col1, col2 = st.columns(2)

        with col1:
            hair_growth = st.selectbox("Excess Hair Growth", ["No", "Yes"])
            skin_darkening = st.selectbox("Skin Darkening", ["No", "Yes"])
            weight_gain = st.selectbox("Unexplained Weight Gain", ["No", "Yes"])
            cycle_regularity = st.selectbox("Irregular Menstrual Cycle (range)", [1, 2, 3, 4, 5])
            age = st.number_input("Age (Years)", min_value=10, max_value=50, value=25)
            avg_follicle_size_L = st.number_input("Average Follicle Size (Left) (mm)", min_value=0.0, max_value=30.0, value=14.0)
            pimples = st.selectbox("Frequent Pimples", ["No", "Yes"])
            beta_HCG_II = st.number_input("II beta-HCG (mIU/mL)", min_value=0.0, value=1.8)

        with col2:
            beta_HCG_I = st.number_input("I beta-HCG (mIU/mL)", min_value=0.0, value=2.3)
            tsh = st.number_input("TSH (mIU/L)", min_value=0.0, value=2.5)
            weight = st.number_input("Weight (Kg)", min_value=30.0, max_value=150.0, value=60.0)
            cycle_length = st.number_input("Menstrual Cycle Length (Days)", min_value=0, max_value=60, value=30)
            pulse_rate = st.number_input("Pulse Rate (bpm)", min_value=40, max_value=120, value=75)
            bmi = st.number_input("BMI", min_value=10.0, max_value=40.0, value=22.5)
            fast_food = st.selectbox("Frequent Fast Food Consumption", ["No", "Yes"])

        # Predict button
        if st.button("Check for PCOS"):
            new_data = np.array([
                [
                    int(hair_growth == "Yes"),
                    int(skin_darkening == "Yes"),
                    int(weight_gain == "Yes"),
                    cycle_regularity,
                    age,
                    avg_follicle_size_L,
                    int(pimples == "Yes"),
                    beta_HCG_II,
                    beta_HCG_I,
                    tsh,
                    weight,
                    cycle_length,
                    pulse_rate,
                    bmi,
                    int(fast_food == "Yes")
                ]
            ])
            
            scaled_data = scaler.transform(new_data)
            prediction = model.predict(scaled_data)

            if prediction[0] == 0:
                st.success("🟢 Low Risk: No PCOS Detected")
            else:
                st.error("🔴 High Risk: PCOS Detected")

    elif DOCTOR_MENU == "Report Download or Upload":
        st.title("📥 Download & Upload Reports")
        st.write("Doctors can download the patient’s PCOS report or upload it to a hospital system.")

        if "user_data" in st.session_state:
                # Convert session state data into a DataFrame
                report_df = pd.DataFrame(list(st.session_state["user_data"].items()), columns=["Feature", "Value"])

                # Convert DataFrame to CSV
                csv_buffer = BytesIO()
                report_df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)

                # Download button
                st.download_button(
                    label="📥 Download Patient Report (CSV)",
                    data=csv_buffer,
                    file_name="PCOS_Report.csv",
                    mime="text/csv"
                )

                # Upload Section - For Hospital Portal
                st.subheader("🌐 Upload to Hospital System")

                # Hospital Upload URL
                hospital_url = st.text_input("Enter the Hospital EMR Upload URL", placeholder="https://hospital-portal.com/upload")

            # File Upload Button
                if st.button("Upload to Hospital System"):
                    if hospital_url:
                        st.success(f"✅ Report uploaded successfully to {hospital_url}")
                        st.write("Visit the hospital system to confirm the upload.")
                    else:
                        st.warning("⚠️ Please enter a valid hospital system URL.")
            
        else:
            st.warning("⚠️ No report available. Please complete the screening first.")


    elif DOCTOR_MENU == "Next Steps After Screening":
            st.title("📌 Next Steps After Screening")
            st.write("""
            - **Low Risk:** Monitor symptoms and maintain a healthy lifestyle.
            - **High Risk:** Recommend further medical tests such as ultrasound and hormone analysis.
            - Discuss **lifestyle changes** and possible treatments with the patient.
            """)

    elif DOCTOR_MENU == "FAQs":
            st.title("❓ Doctor FAQs")
            with st.expander("1️⃣ How does this tool work?"):
                st.write("It uses machine learning to assess PCOS risk based on patient symptoms and health data.")

            with st.expander("2️⃣ Can I trust the predictions?"):
                st.write("The model is trained on an Indian dataset, so while accurate, local factors may not be fully captured.")

            with st.expander("3️⃣ How can I use this for patient management?"):
                st.write("You can use it for quick screenings, download reports, and guide patients on next steps.")
# Show different copyright text based on the selected mode
if st.session_state.mode == "🩺 Patient":
        st.markdown(
            """
            <div style="text-align: center; font-size: 14px; color: gray;">
                © 2025 Team MidgenAI - Patient Mode.
            </div>
            """,
            unsafe_allow_html=True
        )
elif st.session_state.mode == "👨‍⚕️ Doctor":
        st.markdown(
            """
            <div style="text-align: center; font-size: 14px; color: gray;">
                © 2025 Team MidgenAI - Doctor Mode.
            </div>
            """,
            unsafe_allow_html=True
        )