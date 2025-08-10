import psycopg2
from dotenv import load_dotenv
import os
import streamlit as st
import datasource

load_dotenv()

def main():
    results = datasource.get_stations_names()
    st.title("台鐵車站名稱列表")
    if results:
        st.table([[station] for station in results])
    else:
        st.error("無法取得車站資料")

if __name__ == "__main__":
    main()