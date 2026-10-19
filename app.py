import os
import tempfile

import streamlit as st
import streamlit.components.v1 as components
from pypdf import PdfReader
from pyvis.network import Network
from knowledge_graph import build_knowledge_graph

st.set_page_config(page_title="BrainMesh AI", page_icon="🧠", layout="wide")
st.title("🧠 BrainMesh AI")
st.write("Turn your PDF notes into an interactive knowledge graph.")

uploaded_files = st.file_uploader(
    "Upload your PDF notes", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    documents = {}
    for file in uploaded_files:
        try:
            reader = PdfReader(file)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if text.strip():
                documents[file.name] = text
            else:
                st.warning(f"No extractable text found in {file.name}. It may be scanned.")
        except Exception as error:
            st.error(f"Could not read {file.name}: {error}")

    if documents:
        st.success(f"Read {len(documents)} PDF document(s).")
        with st.spinner("Finding related concepts..."):
            graph = build_knowledge_graph(documents)

        col1, col2 = st.columns(2)
        col1.metric("Topic nodes", graph.number_of_nodes())
        col2.metric("Connections", graph.number_of_edges())

        if graph.number_of_nodes():
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".html", delete=False, encoding="utf-8"
            ) as temp:
                graph_file = temp.name
            try:
                network = Network(height="600px", width="100%", bgcolor="#101827", font_color="white")
                network.barnes_hut()
                network.from_nx(graph)
                network.save_graph(graph_file)
                with open(graph_file, "r", encoding="utf-8") as html:
                    components.html(html.read(), height=620, scrolling=True)
            finally:
                if os.path.exists(graph_file):
                    os.remove(graph_file)

            with st.expander("Explore extracted topics"):
                for _, data in graph.nodes(data=True):
                    st.markdown(f"**{data['source']}**")
                    st.write(data["details"])
                    st.divider()
        else:
            st.info("No usable text found. Try another PDF.")
else:
    st.info("Upload PDF notes to discover connections between topics.")
