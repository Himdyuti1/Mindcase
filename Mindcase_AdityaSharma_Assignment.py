import streamlit as st
from mindcase_adityasharma_assignmnent import query_engine
st.title("Blade Runner 2049 Expert")
user_input = st.text_input("Enter your questions about Blade Runner 2049 :", "")
if st.button("Submit"):
    try:
        response = query_engine(user_input)
        st.write(response)
    except Exception as e:
        st.write(f"An error occurred: {e}")