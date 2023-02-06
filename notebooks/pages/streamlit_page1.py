import streamlit as st
import pandas as pd

st.markdown('Page 2')

if('data' in st.session_state):
    data = st.session_state['data']
    st.write(data)
else:
    st.text('No data uploaded')