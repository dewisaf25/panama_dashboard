import streamlit as st

st.set_page_config(page_title="Panama Load Forecast", page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")

from utils import inject_css  # noqa: E402

inject_css()

pages = [
    st.Page("views/home.py", title="Ringkasan", icon="🏠", default=True),
    st.Page("views/eksplorasi.py", title="Eksplorasi Data", icon="📊"),
    st.Page("views/performa.py", title="Performa Model", icon="🎯"),
    st.Page("views/variabel.py", title="Variabel Berpengaruh", icon="🔍"),
    st.Page("views/forecast.py", title="Forecast Explorer", icon="🔮"),
    st.Page("views/metodologi.py", title="Metodologi & Catatan", icon="📘"),
]
nav = st.navigation(pages)

with st.sidebar:
    st.markdown("---")
    st.markdown("**⚡ Panama Load Forecast**  \nRamalan beban listrik nasional per jam, 24 jam ke depan, dengan **LightGBM**.")
    st.caption("Data: beban listrik per jam Panama 2015–2020 · Target: `nat_demand` (MWh)")

nav.run()
