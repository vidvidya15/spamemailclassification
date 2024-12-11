import streamlit as st
import pickle
from PIL import Image

# Load model and vectorizer
model = pickle.load(open('spam123.pkl', 'rb'))
cv = pickle.load(open('vec123.pkl', 'rb'))

# Set page configuration
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="wide",
)

# Add custom background and styling
def add_custom_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #283048 0%, #859398 100%);
            font-family: 'Arial', sans-serif;
            color: #ffffff;
        }
        .main-title {
            font-size: 36px;
            color: #ffffff;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
        }
        .sub-text {
            color: #dfe6e9;
            font-size: 18px;
            text-align: center;
            margin-bottom: 30px;
        }
        .email-classification {
            background: #2d3436;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.3);
        }
        .result-box {
            border: 3px solid #00cec9;
            background-color: #636e72;
            padding: 20px;
            border-radius: 12px;
            font-size: 20px;
            font-weight: bold;
            color: #ffffff;
            text-align: center;
        }
        .spam {
            border-color: #d63031;
            color: #d63031;
        }
        .not-spam {
            border-color: #00b894;
            color: #00b894;
        }
        .sidebar-header {
            font-size: 20px;
            font-weight: bold;
            color: #ffffff;
            margin-bottom: 10px;
        }
        .footer {
            text-align: center;
            color: #ffffff;
            font-size: 14px;
            margin-top: 50px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def main():
    add_custom_styles()

    # Header Section
    st.markdown("<h1 class='main-title'>📧 Email Spam Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-text'>A simple tool to classify emails as Spam or Not Spam using Machine Learning.</p>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("<div class='sidebar-header'>About the App</div>", unsafe_allow_html=True)
        st.write("🚀 This app uses a pre-trained Machine Learning model to detect spam emails.")
        st.write("💡 Just paste your email content and hit the **Classify** button to get started!")

    # Main Layout
    st.markdown("<div class='email-classification'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #dfe6e9;'>🔍 Email Classification</h3>", unsafe_allow_html=True)

    user_input = st.text_area(
        "Enter your email content below:",
        height=200,
        placeholder="Type your email here...",
    )

    if st.button("Classify Email 🚀"):
        if user_input.strip():
            data = [user_input]
            vec = cv.transform(data).toarray()
            result = model.predict(vec)

            if result[0] == 0:
                st.markdown(
                    "<div class='result-box not-spam'>✅ This is <strong>Not a Spam Email</strong> 😊</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='result-box spam'>🚨 This is <strong>A Spam Email</strong> ⚠️</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.warning("⚠️ Please enter some text to classify.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown(
        "<div class='footer'>Developed by Srujana H B • Powered by Streamlit</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
