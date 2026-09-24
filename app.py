# ============================================================
# app.py
#
# Persistent Multi-Modal RAG Streamlit Application
#
# Features:
#
#   ✅ PDF persistence
#   ✅ FAISS persistence
#   ✅ Chat persistence
#   ✅ Separate chat per PDF
#   ✅ Continue conversation after refresh
#   ✅ Retrieved images
#   ✅ Sources button
#   ✅ Sources persisted with chat
#
# IMPORTANT:
#
# The RAG flow is NOT changed.
#
# Sources are extracted from the documents that the RAG
# already retrieved.
# ============================================================


import streamlit as st

import base64
import json
import shutil
import hashlib

from pathlib import Path

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from multimodel_rag import (
    build_multimodal_rag,
    load_multimodal_rag,
    rag_exists,
    get_pdf_storage_paths,
)


# ============================================================
# Streamlit configuration
# ============================================================

st.set_page_config(

    page_title=(
        "Multimodal RAG Chat"
    ),

    page_icon="📄",

    layout="wide",
)


st.title(
    "📄 Multi-Modal RAG by Virtual Techbox"
)


# ============================================================
# Persistent storage
# ============================================================

BASE_STORAGE_DIR = Path(
    "./rag_storage"
)

CHAT_HISTORY_FILE = (
    BASE_STORAGE_DIR
    / "chat_history.json"
)

PDF_STORAGE_DIR = (
    BASE_STORAGE_DIR
    / "pdfs"
)


BASE_STORAGE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PDF_STORAGE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load all PDF chat histories
# ============================================================

def load_all_chat_history():

    if not CHAT_HISTORY_FILE.exists():

        return {}

    try:

        with open(
            CHAT_HISTORY_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(
                f
            )

        if isinstance(
            data,
            dict,
        ):

            return data

    except Exception:

        pass

    return {}


# ============================================================
# Save all PDF chat histories
# ============================================================

def save_all_chat_history(
    all_history,
):

    BASE_STORAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        CHAT_HISTORY_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(

            all_history,

            f,

            indent=2,

            ensure_ascii=False,
        )


# ============================================================
# Serialize LangChain history
# ============================================================

def serialize_chat_history(
    chat_history,
):

    serialized = []

    for msg in chat_history:

        if isinstance(
            msg,
            HumanMessage,
        ):

            serialized.append(

                {
                    "type": "human",

                    "content": msg.content,
                }
            )

        elif isinstance(
            msg,
            AIMessage,
        ):

            serialized.append(

                {
                    "type": "ai",

                    "content": msg.content,
                }
            )

    return serialized


# ============================================================
# Restore LangChain history
# ============================================================

def restore_langchain_history(
    serialized_history,
):

    history = []

    for msg in serialized_history:

        if msg.get(
            "type"
        ) == "human":

            history.append(

                HumanMessage(
                    content=msg.get(
                        "content",
                        "",
                    )
                )
            )

        elif msg.get(
            "type"
        ) == "ai":

            history.append(

                AIMessage(
                    content=msg.get(
                        "content",
                        "",
                    )
                )
            )

    return history


# ============================================================
# Get PDF display name
# ============================================================

def get_pdf_display_name(
    pdf_id,
):

    metadata_file = (
        PDF_STORAGE_DIR
        / f"{pdf_id}.json"
    )

    if metadata_file.exists():

        try:

            with open(
                metadata_file,
                "r",
                encoding="utf-8",
            ) as f:

                metadata = json.load(
                    f
                )

            return metadata.get(

                "filename",

                f"{pdf_id[:12]}.pdf",
            )

        except Exception:

            pass

    return (
        f"{pdf_id[:12]}.pdf"
    )


# ============================================================
# Save PDF metadata
# ============================================================

def save_pdf_metadata(
    pdf_id,
    filename,
):

    metadata_file = (
        PDF_STORAGE_DIR
        / f"{pdf_id}.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(

            {
                "pdf_id": pdf_id,

                "filename": filename,
            },

            f,

            indent=2,
        )


# ============================================================
# Get persisted PDFs
# ============================================================

def get_available_pdfs():

    pdfs = []

    for pdf_file in (
        PDF_STORAGE_DIR.glob(
            "*.pdf"
        )
    ):

        pdf_id = (
            pdf_file.stem
        )

        if rag_exists(
            pdf_id
        ):

            pdfs.append(

                {
                    "id": pdf_id,

                    "filename": (
                        get_pdf_display_name(
                            pdf_id
                        )
                    ),
                }
            )

    return sorted(

        pdfs,

        key=lambda x:
        x["filename"].lower(),
    )


# ============================================================
# Process uploaded PDF
# ============================================================

def process_uploaded_pdf(
    uploaded_file,
):

    # --------------------------------------------------------
    # Read PDF
    # --------------------------------------------------------

    pdf_bytes = (
        uploaded_file.getvalue()
    )

    # --------------------------------------------------------
    # Generate deterministic PDF ID
    # --------------------------------------------------------

    pdf_id = hashlib.sha256(
        pdf_bytes
    ).hexdigest()

    # --------------------------------------------------------
    # Persistent paths
    # --------------------------------------------------------

    (
        persistent_pdf_path,

        faiss_path,

        docstore_path,

    ) = get_pdf_storage_paths(
        pdf_id
    )

    # --------------------------------------------------------
    # Save PDF
    # --------------------------------------------------------

    if not persistent_pdf_path.exists():

        with open(
            persistent_pdf_path,
            "wb",
        ) as f:

            f.write(
                pdf_bytes
            )

    # --------------------------------------------------------
    # Save filename
    # --------------------------------------------------------

    save_pdf_metadata(

        pdf_id,

        uploaded_file.name,
    )

    # --------------------------------------------------------
    # Build or load RAG
    # --------------------------------------------------------

    if not rag_exists(
        pdf_id
    ):

        with st.spinner(
            "Processing PDF and building RAG index..."
        ):

            rag_chain = (
                build_multimodal_rag(

                    str(
                        persistent_pdf_path
                    ),

                    pdf_id=pdf_id,
                )
            )

        st.success(
            "PDF processed and RAG index saved."
        )

    else:

        rag_chain = (
            load_multimodal_rag(
                pdf_id
            )
        )

    return (
        pdf_id,
        rag_chain,
    )


# ============================================================
# Render Sources button
# ============================================================

def render_sources(
    sources,
):
    """
    Display a compact Sources popover.

    Example:

        [ Sources ]

    Clicking it opens:

        Sources

        📄 Text — Page 2
        📊 Table — Page 4
        🖼️ Image — Page 5
    """

    # --------------------------------------------------------
    # No source metadata
    # --------------------------------------------------------

    if not sources:

        with st.popover(
            "Sources"
        ):

            st.caption(
                "No source metadata available."
            )

        return

    # --------------------------------------------------------
    # Sources button
    # --------------------------------------------------------

    with st.popover(
        "Sources"
    ):

        st.markdown(
            "### Sources"
        )

        for index, source in enumerate(
            sources,
            start=1,
        ):

            source_type = source.get(
                "type",
                "Document",
            )

            page = source.get(
                "page"
            )

            # ------------------------------------------------
            # Select icon
            # ------------------------------------------------

            if source_type == "Text":

                icon = "📄"

            elif source_type == "Table":

                icon = "📊"

            elif source_type == "Image":

                icon = "🖼️"

            else:

                icon = "📌"

            # ------------------------------------------------
            # Page information
            # ------------------------------------------------

            if page:

                location = (
                    f"Page {page}"
                )

            else:

                location = (
                    "Page information unavailable"
                )

            st.markdown(

                f"**{index}. {icon} "
                f"{source_type}**  \n"
                f"{location}"
            )

            if index < len(
                sources
            ):

                st.divider()


# ============================================================
# Initialize session state
# ============================================================

if (
    "all_chat_history"
    not in st.session_state
):

    st.session_state.all_chat_history = (
        load_all_chat_history()
    )


if (
    "selected_pdf_id"
    not in st.session_state
):

    st.session_state.selected_pdf_id = None


if (
    "rag_chain"
    not in st.session_state
):

    st.session_state.rag_chain = None


if (
    "messages"
    not in st.session_state
):

    st.session_state.messages = []


if (
    "chat_history"
    not in st.session_state
):

    st.session_state.chat_history = []


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header(
        "📚 Your PDFs"
    )

    available_pdfs = (
        get_available_pdfs()
    )

    pdf_options = [

        pdf["id"]

        for pdf in available_pdfs
    ]

    pdf_labels = {

        pdf["id"]: pdf["filename"]

        for pdf in available_pdfs
    }

    # --------------------------------------------------------
    # PDF selector
    # --------------------------------------------------------

    if pdf_options:

        current_index = 0

        if (
            st.session_state.selected_pdf_id
            in pdf_options
        ):

            current_index = (
                pdf_options.index(
                    st.session_state.selected_pdf_id
                )
            )

        selected_pdf_id = (
            st.selectbox(

                "Select PDF",

                options=pdf_options,

                index=current_index,

                format_func=lambda x:
                    pdf_labels.get(
                        x,
                        x[:12],
                    ),
            )
        )

        # ----------------------------------------------------
        # Switch PDF
        # ----------------------------------------------------

        if (
            selected_pdf_id
            != st.session_state.selected_pdf_id
        ):

            st.session_state.selected_pdf_id = (
                selected_pdf_id
            )

            # Load RAG
            if rag_exists(
                selected_pdf_id
            ):

                with st.spinner(
                    "Loading saved RAG..."
                ):

                    st.session_state.rag_chain = (
                        load_multimodal_rag(
                            selected_pdf_id
                        )
                    )

            # Load PDF-specific chat
            pdf_history = (
                st.session_state
                .all_chat_history
                .get(
                    selected_pdf_id,
                    {},
                )
            )

            st.session_state.messages = (
                pdf_history.get(
                    "messages",
                    [],
                )
            )

            st.session_state.chat_history = (
                restore_langchain_history(

                    pdf_history.get(
                        "chat_history",
                        [],
                    )
                )
            )

            st.rerun()

    else:

        st.info(
            "No previously uploaded PDFs."
        )

    st.divider()

    # ========================================================
    # Clear current PDF chat
    # ========================================================

    if st.session_state.selected_pdf_id:

        if st.button(
            "🧹 Clear Current PDF Chat",
            use_container_width=True,
        ):

            current_pdf = (
                st.session_state.selected_pdf_id
            )

            st.session_state.all_chat_history.pop(

                current_pdf,

                None,
            )

            save_all_chat_history(
                st.session_state.all_chat_history
            )

            st.session_state.messages = []

            st.session_state.chat_history = []

            st.rerun()

    # ========================================================
    # Delete current PDF
    # ========================================================

    if st.session_state.selected_pdf_id:

        if st.button(
            "🗑️ Delete Current PDF",
            use_container_width=True,
        ):

            current_pdf = (
                st.session_state.selected_pdf_id
            )

            (
                pdf_path,

                faiss_path,

                docstore_path,

            ) = get_pdf_storage_paths(
                current_pdf
            )

            # Delete PDF
            if pdf_path.exists():

                pdf_path.unlink()

            # Delete PDF metadata
            metadata_file = (
                PDF_STORAGE_DIR
                / f"{current_pdf}.json"
            )

            if metadata_file.exists():

                metadata_file.unlink()

            # Delete FAISS
            if faiss_path.exists():

                shutil.rmtree(
                    faiss_path
                )

            # Delete docstore
            if docstore_path.exists():

                docstore_path.unlink()

            # Delete chat
            st.session_state.all_chat_history.pop(

                current_pdf,

                None,
            )

            save_all_chat_history(
                st.session_state.all_chat_history
            )

            # Reset state
            st.session_state.selected_pdf_id = None

            st.session_state.rag_chain = None

            st.session_state.messages = []

            st.session_state.chat_history = []

            st.rerun()


# ============================================================
# Upload PDF
# ============================================================

uploaded_file = st.file_uploader(

    "Upload a new PDF",

    type="pdf",
)


# ============================================================
# Process uploaded PDF
# ============================================================

if uploaded_file:

    pdf_id, rag_chain = (
        process_uploaded_pdf(
            uploaded_file
        )
    )

    # --------------------------------------------------------
    # Select uploaded PDF
    # --------------------------------------------------------

    if (
        st.session_state.selected_pdf_id
        != pdf_id
    ):

        st.session_state.selected_pdf_id = (
            pdf_id
        )

        st.session_state.rag_chain = (
            rag_chain
        )

        # ----------------------------------------------------
        # Restore existing chat
        # ----------------------------------------------------

        pdf_history = (
            st.session_state
            .all_chat_history
            .get(
                pdf_id,
                {},
            )
        )

        st.session_state.messages = (
            pdf_history.get(
                "messages",
                [],
            )
        )

        st.session_state.chat_history = (
            restore_langchain_history(

                pdf_history.get(
                    "chat_history",
                    [],
                )
            )
        )

    else:

        st.session_state.rag_chain = (
            rag_chain
        )


# ============================================================
# Restore RAG after refresh
# ============================================================

if (

    st.session_state.selected_pdf_id

    and

    st.session_state.rag_chain is None

):

    selected_pdf_id = (
        st.session_state.selected_pdf_id
    )

    if rag_exists(
        selected_pdf_id
    ):

        with st.spinner(
            "Restoring saved RAG..."
        ):

            st.session_state.rag_chain = (
                load_multimodal_rag(
                    selected_pdf_id
                )
            )

        # ----------------------------------------------------
        # Restore chat for selected PDF
        # ----------------------------------------------------

        pdf_history = (
            st.session_state
            .all_chat_history
            .get(
                selected_pdf_id,
                {},
            )
        )

        st.session_state.messages = (
            pdf_history.get(
                "messages",
                [],
            )
        )

        st.session_state.chat_history = (
            restore_langchain_history(

                pdf_history.get(
                    "chat_history",
                    [],
                )
            )
        )


# ============================================================
# Current PDF
# ============================================================

if st.session_state.selected_pdf_id:

    current_filename = (
        get_pdf_display_name(
            st.session_state.selected_pdf_id
        )
    )

    st.caption(

        f"📄 Current PDF: "
        f"**{current_filename}**"
    )


# ============================================================
# Display previous messages
# ============================================================

for msg in (
    st.session_state.messages
):

    with st.chat_message(
        msg["role"]
    ):

        # ----------------------------------------------------
        # Message content
        # ----------------------------------------------------

        st.write(
            msg["content"]
        )

        # ----------------------------------------------------
        # Retrieved images
        # ----------------------------------------------------

        if (
            isinstance(
                msg,
                dict,
            )

            and

            msg.get(
                "images"
            )
        ):

            st.write(
                "### Retrieved Images"
            )

            for img_b64 in (
                msg["images"]
            ):

                try:

                    image_bytes = (
                        base64.b64decode(
                            img_b64
                        )
                    )

                    st.image(
                        image_bytes
                    )

                except Exception:

                    st.warning(
                        "Could not render image"
                    )

        # ----------------------------------------------------
        # Sources button
        #
        # This is also shown for old persisted messages
        # if source metadata exists.
        # ----------------------------------------------------

        if (
            isinstance(
                msg,
                dict,
            )

            and

            "sources" in msg
        ):

            render_sources(
                msg.get(
                    "sources",
                    [],
                )
            )


# ============================================================
# Chat input
# ============================================================

user_question = st.chat_input(

    "Ask a question about the PDF..."
)


# ============================================================
# Process question
# ============================================================

if user_question:

    # --------------------------------------------------------
    # Ensure RAG is available
    # --------------------------------------------------------

    if (

        not st.session_state.selected_pdf_id

        or

        st.session_state.rag_chain is None

    ):

        st.warning(
            "Please upload or select a PDF first."
        )

        st.stop()

    # --------------------------------------------------------
    # Current PDF
    # --------------------------------------------------------

    pdf_id = (
        st.session_state.selected_pdf_id
    )

    # ========================================================
    # User message
    # ========================================================

    st.session_state.messages.append(

        {
            "role": "user",

            "content": user_question,
        }
    )

    with st.chat_message(
        "user"
    ):

        st.write(
            user_question
        )

    # ========================================================
    # Assistant
    # ========================================================

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Thinking..."
        ):

            # ------------------------------------------------
            # EXISTING RAG INVOCATION
            #
            # DO NOT CHANGE THIS FLOW.
            # ------------------------------------------------

            response = (
                st.session_state
                .rag_chain
                .invoke(

                    {

                        "input":
                            user_question,

                        "chat_history":
                            st.session_state
                            .chat_history,
                    }
                )
            )

        # ====================================================
        # Extract response
        # ====================================================

        if isinstance(
            response,
            dict,
        ):

            answer = response.get(
                "answer",
                "",
            )

            context = response.get(
                "context",
                {},
            )

            images = context.get(
                "images",
                [],
            )

            # ------------------------------------------------
            # NEW:
            # Sources are returned from the same retrieved docs.
            # ------------------------------------------------

            sources = context.get(
                "sources",
                [],
            )

        elif isinstance(
            response,
            list,
        ):

            answer = str(
                response
            )

            images = []

            sources = []

        else:

            answer = str(
                response
            )

            images = []

            sources = []

        # ====================================================
        # Display answer
        # ====================================================

        st.write(
            answer
        )

        # ====================================================
        # Display retrieved images
        # ====================================================

        if images:

            st.write(
                "### Retrieved Images"
            )

            for img_b64 in images:

                try:

                    image_bytes = (
                        base64.b64decode(
                            img_b64
                        )
                    )

                    st.image(
                        image_bytes
                    )

                except Exception:

                    st.warning(
                        "Could not render image"
                    )

        # ====================================================
        # SOURCES BUTTON
        #
        # This appears at the END of the assistant response.
        # ====================================================

        render_sources(
            sources
        )

    # ========================================================
    # Update LangChain chat history
    # ========================================================

    st.session_state.chat_history.append(

        HumanMessage(
            content=user_question
        )
    )

    st.session_state.chat_history.append(

        AIMessage(
            content=answer
        )
    )

    # ========================================================
    # Store assistant message
    # ========================================================

    st.session_state.messages.append(

        {

            "role": "assistant",

            "content": answer,

            "images": images,

            # ------------------------------------------------
            # Persist sources with the answer.
            # ------------------------------------------------

            "sources": sources,
        }
    )

    # ========================================================
    # Persist chat PER PDF
    # ========================================================

    st.session_state.all_chat_history[
        pdf_id
    ] = {

        "filename":
            get_pdf_display_name(
                pdf_id
            ),

        "messages":
            st.session_state.messages,

        "chat_history":
            serialize_chat_history(

                st.session_state
                .chat_history
            ),
    }

    save_all_chat_history(

        st.session_state
        .all_chat_history
    )