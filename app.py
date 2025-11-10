import streamlit as st
import requests

st.set_page_config(page_title="B737 RAG Service", layout="wide")

st.title("Boeing 737 RAG Service")
st.markdown("Retrieve and analyze Boeing 737 manual content using multimodal RAG.")

query = st.text_input("Enter your question:", placeholder="e.g. What does the FUEL PUMP switch do?")

API_URL = "http://127.0.0.1:8000/query"  # FastAPI endpoint

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Contacting RAG backend..."):
            payload = {"question": query, "top_k": 5}

            try:
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    st.subheader("Answer")
                    st.write(result.get("answer", "No answer found."))

                    if "pages" in result:
                        st.markdown("### Referenced Pages")
                        st.write(result["pages"])

                else:
                    st.error(f"Error {response.status_code}: {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"API connection failed: {e}")
