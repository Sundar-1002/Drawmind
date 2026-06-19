import os
import json
import streamlit as st
from PIL import Image
from core.ingest import ingest_single, DRAWINGS_FOLDER
from core.search import ask_question, get_database_stats

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
    total, component_types = get_database_stats()
    st.metric("Total Drawings", total)

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
                save_path = os.path.join(DRAWINGS_FOLDER, uploaded_file.name)

                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner(f"Processing {uploaded_file.name}..."):
                    success, result = ingest_single(save_path, uploaded_file.name)

                if success:
                    st.success(f"{uploaded_file.name}")
                    st.write(f"Component: {result.get('component_type', 'unknown').title()}")
                    st.write(f"Material: {result.get('material', 'Not specified')}")
                else:
                    st.error(f"Failed: {result}")

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
        answer, referenced_metadatas = ask_question(question)

    st.markdown("### Answer")
    st.write(answer)

    if referenced_metadatas:
        st.markdown("### Referenced Drawings")
        cols = st.columns(len(referenced_metadatas))
        for idx, meta in enumerate(referenced_metadatas):
            with cols[idx]:
                image_path = meta.get("image_path")
                if image_path and os.path.exists(image_path):
                    img = Image.open(image_path)
                    st.image(img, use_container_width=True)
                raw = json.loads(meta.get("raw_json", "{}"))
                st.caption(f"**{meta['filename']}**")
                st.caption(f"Component: {meta['component_type'].title()}")
                material = meta.get('material', 'unknown')
                st.caption(f"Material: {material if material != 'unknown' else 'Not specified'}")
                if raw.get("dimensions"):
                    st.caption(f"Key dimension: {raw['dimensions'][0]}")

elif ask_button and not question:
    st.warning("Please enter a question first.")