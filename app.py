import os
import json
import streamlit as st
from dotenv import load_dotenv
import requests

load_dotenv()

# FastAPI base URL 
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="DrawMind",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 DrawMind")
st.caption("Engineering Drawing Intelligence System")

# SIDEBAR

with st.sidebar:
    st.header("Database Stats")
    try:
        response = requests.get(f"{API_URL}/stats")
        if response.status_code == 200:
            data = response.json()
            total = data.get("total", 0)
            st.metric("Total Drawings", total)
        else:
            st.error("Could not load stats")
            total = 0
    except Exception as e:
        st.error(f"Error fetching stats: {e}")
        total = 0
    st.divider()

    st.subheader("Add New Drawings")
    st.caption("Upload PNG, JPG, or WEBP files")

    uploaded_files = st.file_uploader(
        "Upload engineering drawings",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        if st.button("Ingest Uploaded Drawings", type="primary", use_container_width=True):
            for uploaded_file in uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/ingest",
                            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                            timeout=60
                        )
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"Successfully ingested {uploaded_file.name}")
                            st.write(f"Component Type: {result.get('component_type', 'unknown').title()}")
                            material = result.get('material', 'unknown')
                            st.write(f"Material: {material if material != 'unknown' else 'Not specified'}")
                        else:
                            st.error(f"Failed to ingest {uploaded_file.name}: {response.text}")
                    except Exception as e:
                        st.error(f"Error ingesting {uploaded_file.name}: {e}")
            st.rerun()

# MAIN AREA

st.subheader("Ask a Question")
st.write("Ask anything about the engineering drawings in natural language.")


question = st.text_input(
    "Your question",
    placeholder="e.g. which drawing has alloy steel? show me all gear drawings",
    label_visibility="collapsed"
)

col1, col2 = st.columns([1, 5])
with col1:
    ask_button = st.button("Ask", type="primary", use_container_width=True)

if ask_button and question:
    with st.spinner("Analysing drawings and generating answer..."):
        try:
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                referenced_drawings = data["referenced_drawings"]

                st.markdown("### Answer")
                st.write(answer)

                if referenced_drawings:
                    st.markdown("### Referenced Drawings")
                    cols = st.columns(len(referenced_drawings))
                    for idx, drawing in enumerate(referenced_drawings):
                        with cols[idx]:
                            image_url = drawing.get("image_url")
                            if image_url:
                                st.image(image_url, use_container_width=True)
                            st.caption(f"**{drawing['filename']}**")
                            st.caption(f"Component: {drawing['component_type'].title()}")
                            material = drawing.get('material', 'unknown')
                            st.caption(f"Material: {material if material != 'unknown' else 'Not specified'}")
                            if drawing.get("dimensions"):
                                st.caption(f"Key dimension: {drawing['dimensions'][0]}")
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure FastAPI is running on http://localhost:8000")
        except Exception as e:
            st.error(f"Error: {e}")

elif ask_button and not question:
    st.warning("Please enter a question first.")