# ============================================================
# multimodel_rag.py
#
# Persistent Multi-Modal RAG
#
# Flow:
#
# PDF
#   ↓
# Unstructured PDF extraction
#   ↓
# Text / Tables / Images
#   ↓
# Text + Table summaries
# Image summaries
#   ↓
# FAISS
#   ↓
# MultiVectorRetriever
#   ↓
# History-aware Retriever
#   ↓
# Multimodal Prompt
#   ↓
# LLM
#   ↓
# Answer + Images + Sources
#
# IMPORTANT:
# The existing RAG flow has NOT been changed.
#
# The Sources feature only adds metadata about documents that
# were ALREADY retrieved by the existing retriever.
#
# Persistence:
#   - PDF
#   - FAISS index
#   - Parent document store
# ============================================================


import os
import uuid
import base64
import io
import pickle
import hashlib

from pathlib import Path

from dotenv import load_dotenv

from PIL import Image

from unstructured.partition.pdf import partition_pdf

from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.output_parsers import (
    StrOutputParser,
)

from langchain_core.runnables import (
    RunnableLambda,
)

from langchain_core.documents import Document

from langchain_classic.retrievers.multi_vector import (
    MultiVectorRetriever,
)

from langchain_core.stores import (
    InMemoryStore,
)

from langchain_community.vectorstores import (
    FAISS,
)

from langchain_classic.chains import (
    create_history_aware_retriever,
)


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# Persistent storage
# ============================================================

BASE_STORAGE_DIR = Path(
    "./rag_storage"
)

PDF_STORAGE_DIR = (
    BASE_STORAGE_DIR / "pdfs"
)

FAISS_STORAGE_DIR = (
    BASE_STORAGE_DIR / "faiss"
)

DOCSTORE_STORAGE_DIR = (
    BASE_STORAGE_DIR / "docstores"
)


PDF_STORAGE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FAISS_STORAGE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DOCSTORE_STORAGE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SourceString
# ============================================================
#
# Tables and images are stored as strings in the existing
# MultiVectorRetriever docstore.
#
# SourceString behaves exactly like a normal Python string,
# but additionally carries source metadata.
#
# Therefore:
#
#     str(table)
#
# still behaves exactly as before.
#
# The RAG prompt flow is not changed.
# ============================================================

class SourceString(str):
    """
    String subclass that carries source metadata.

    It behaves like a normal string, so existing RAG logic
    continues to work.
    """

    def __new__(
        cls,
        value,
        source_metadata=None,
    ):

        obj = str.__new__(
            cls,
            value,
        )

        obj.source_metadata = (
            source_metadata or {}
        )

        return obj


# ============================================================
# Calculate PDF ID
# ============================================================

def calculate_pdf_id(
    pdf_path: str,
) -> str:
    """
    Calculate SHA256 hash of the PDF.

    The hash is used as the unique PDF ID.
    """

    sha256 = hashlib.sha256()

    with open(
        pdf_path,
        "rb",
    ) as f:

        while True:

            data = f.read(
                1024 * 1024
            )

            if not data:
                break

            sha256.update(
                data
            )

    return sha256.hexdigest()


# ============================================================
# Normalize image
# ============================================================

def normalize_image_b64(
    image_b64: str,
):
    """
    Convert extracted image to PNG base64.

    This ensures that images sent to the vision model
    use a supported image format.
    """

    if not image_b64:
        return None

    try:

        image_bytes = (
            base64.b64decode(
                image_b64
            )
        )

        img = Image.open(
            io.BytesIO(
                image_bytes
            )
        ).convert("RGB")

        buffer = io.BytesIO()

        img.save(
            buffer,
            format="PNG",
        )

        return base64.b64encode(
            buffer.getvalue()
        ).decode(
            "utf-8"
        )

    except Exception:

        return None


# ============================================================
# Extract page number from Unstructured metadata
# ============================================================

def get_page_number(
    metadata,
):
    """
    Safely extract page_number from Unstructured metadata.

    Different Unstructured versions may expose metadata slightly
    differently, so both attribute access and to_dict() are tried.
    """

    if metadata is None:
        return None

    # Direct attribute
    page_number = getattr(
        metadata,
        "page_number",
        None,
    )

    if page_number is not None:
        return page_number

    # Metadata object -> dictionary
    try:

        metadata_dict = metadata.to_dict()

        return metadata_dict.get(
            "page_number"
        )

    except Exception:

        return None


# ============================================================
# Format page number
# ============================================================

def format_page_number(
    page_number,
):
    """
    Convert page number/list/range into a readable string.
    """

    if page_number is None:
        return None

    if isinstance(
        page_number,
        (list, tuple, set),
    ):

        return ", ".join(
            str(x)
            for x in page_number
        )

    return str(
        page_number
    )


# ============================================================
# Paths associated with PDF
# ============================================================

def get_pdf_storage_paths(
    pdf_id: str,
):

    pdf_path = (
        PDF_STORAGE_DIR
        / f"{pdf_id}.pdf"
    )

    faiss_path = (
        FAISS_STORAGE_DIR
        / pdf_id
    )

    docstore_path = (
        DOCSTORE_STORAGE_DIR
        / f"{pdf_id}.pkl"
    )

    return (
        pdf_path,
        faiss_path,
        docstore_path,
    )


# ============================================================
# Check whether RAG exists
# ============================================================

def rag_exists(
    pdf_id: str,
) -> bool:

    (
        pdf_path,
        faiss_path,
        docstore_path,
    ) = get_pdf_storage_paths(
        pdf_id
    )

    faiss_index = (
        faiss_path
        / "index.faiss"
    )

    faiss_metadata = (
        faiss_path
        / "index.pkl"
    )

    return (
        pdf_path.exists()
        and faiss_index.exists()
        and faiss_metadata.exists()
        and docstore_path.exists()
    )


# ============================================================
# Save parent document store
# ============================================================

def save_docstore(
    store,
    docstore_path: Path,
):
    """
    Persist InMemoryStore to disk.

    This is required because FAISS only stores the vectors;
    MultiVectorRetriever needs the original parent documents.
    """

    with open(
        docstore_path,
        "wb",
    ) as f:

        pickle.dump(
            store.store,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


# ============================================================
# Load parent document store
# ============================================================

def load_docstore(
    docstore_path: Path,
):

    with open(
        docstore_path,
        "rb",
    ) as f:

        stored_data = pickle.load(
            f
        )

    store = InMemoryStore()

    store.store.update(
        stored_data
    )

    return store


# ============================================================
# Build NEW multimodal RAG
# ============================================================

def build_multimodal_rag(
    pdf_path: str,
    pdf_id: str | None = None,
):

    # --------------------------------------------------------
    # Generate PDF ID
    # --------------------------------------------------------

    if pdf_id is None:

        pdf_id = calculate_pdf_id(
            pdf_path
        )

    (
        persistent_pdf_path,
        faiss_path,
        docstore_path,
    ) = get_pdf_storage_paths(
        pdf_id
    )

    # --------------------------------------------------------
    # Persist original PDF
    # --------------------------------------------------------

    if not persistent_pdf_path.exists():

        with open(
            pdf_path,
            "rb",
        ) as source:

            with open(
                persistent_pdf_path,
                "wb",
            ) as destination:

                destination.write(
                    source.read()
                )

    # ========================================================
    # 1. Extract text, tables and images
    # ========================================================

    chunks = partition_pdf(

        filename=str(
            persistent_pdf_path
        ),

        strategy="hi_res",

        infer_table_structure=True,

        extract_image_block_types=[
            "Image"
        ],

        extract_image_block_to_payload=True,

        chunking_strategy="by_title",

        max_characters=10000,

        combine_text_under_n_chars=2000,

        new_after_n_chars=6000,
    )

    texts = []

    tables = []

    images = []

    # --------------------------------------------------------
    # Separate extracted elements
    # --------------------------------------------------------

    for chunk in chunks:

        # ----------------------------------------------------
        # CompositeElement
        # ----------------------------------------------------

        if (
            "CompositeElement"
            in str(type(chunk))
        ):

            texts.append(
                chunk
            )

            for el in (
                chunk.metadata.orig_elements
            ):

                if (
                    "Image"
                    in str(type(el))
                ):

                    images.append(
                        {
                            "base64": (
                                el.metadata.image_base64
                            ),

                            "page_number": (
                                get_page_number(
                                    el.metadata
                                )
                            ),
                        }
                    )

        # ----------------------------------------------------
        # Table
        # ----------------------------------------------------

        if (
            "Table"
            in str(type(chunk))
        ):

            tables.append(
                chunk
            )

    # ========================================================
    # 2. Text + table summaries
    # ========================================================

    summarizer_llm = ChatOpenAI(
        model="gpt-4.1-mini"
    )

    summary_prompt = (
        ChatPromptTemplate.from_template(
            """
Summarize the following content concisely.

The content can be normal text or a table.

For tables:
- Identify what the table represents.
- Preserve important column names.
- Preserve important row values.
- Mention important relationships or comparisons.

Content:
{element}
"""
        )
    )

    summarize_chain = (
        {
            "element": lambda x: x
        }

        | summary_prompt

        | summarizer_llm

        | StrOutputParser()
    )

    # --------------------------------------------------------
    # Text summaries
    # --------------------------------------------------------

    text_summaries = (
        summarize_chain.batch(
            texts
        )
    )

    # --------------------------------------------------------
    # Table HTML
    #
    # IMPORTANT:
    # The table remains a string for the existing RAG flow.
    #
    # We only attach source metadata to it.
    # --------------------------------------------------------

    tables_html = []

    for table in tables:

        html = getattr(
            table.metadata,
            "text_as_html",
            None,
        )

        if not html:
            continue

        table_source = SourceString(

            html,

            {
                "type": "Table",

                "page_number": (
                    get_page_number(
                        table.metadata
                    )
                ),
            },
        )

        tables_html.append(
            table_source
        )

    table_summaries = (
        summarize_chain.batch(
            tables_html
        )
    )

    # ========================================================
    # 3. Image processing
    # ========================================================

    normalized_images = []

    for image_info in images:

        raw_image = image_info.get(
            "base64"
        )

        normalized = (
            normalize_image_b64(
                raw_image
            )
        )

        if normalized:

            normalized_images.append(
                SourceString(

                    normalized,

                    {
                        "type": "Image",

                        "page_number": (
                            image_info.get(
                                "page_number"
                            )
                        ),
                    },
                )
            )

    # --------------------------------------------------------
    # Image summarization
    # --------------------------------------------------------

    image_prompt = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "user",
                    [
                        {
                            "type": "text",

                            "text": (
                                "Describe this image "
                                "in detail for retrieval. "
                                "Include important objects, "
                                "diagrams, labels, charts, "
                                "relationships and visual "
                                "information."
                            ),
                        },

                        {
                            "type": "image_url",

                            "image_url": {
                                "url": (
                                    "data:image/png;base64,"
                                    "{image}"
                                )
                            },
                        },
                    ],
                )
            ]
        )
    )

    image_chain = (
        image_prompt

        | ChatOpenAI(
            model="gpt-4.1-mini"
        )

        | StrOutputParser()
    )

    image_summaries = (
        image_chain.batch(
            normalized_images
        )
    )

    # ========================================================
    # 4. Build FAISS vector store
    # ========================================================

    embedding = OpenAIEmbeddings()

    summary_docs = []

    store = InMemoryStore()

    # --------------------------------------------------------
    # Existing RAG structure
    # --------------------------------------------------------

    all_summaries = (
        text_summaries
        + table_summaries
        + image_summaries
    )

    all_originals = (
        texts
        + tables_html
        + normalized_images
    )

    # --------------------------------------------------------
    # Generate parent IDs
    # --------------------------------------------------------

    ids = [
        str(uuid.uuid4())
        for _ in all_summaries
    ]

    # --------------------------------------------------------
    # Create child/summary documents
    # --------------------------------------------------------

    for i, summary in enumerate(
        all_summaries
    ):

        summary_docs.append(
            Document(

                page_content=summary,

                metadata={
                    "doc_id": ids[i]
                },
            )
        )

    # --------------------------------------------------------
    # Create FAISS index
    # --------------------------------------------------------

    vectorstore = FAISS.from_documents(
        summary_docs,
        embedding,
    )

    # --------------------------------------------------------
    # Store original parent documents
    # --------------------------------------------------------

    store.mset(
        list(
            zip(
                ids,
                all_originals,
            )
        )
    )

    # --------------------------------------------------------
    # Persist FAISS
    # --------------------------------------------------------

    vectorstore.save_local(
        str(
            faiss_path
        )
    )

    # --------------------------------------------------------
    # Persist parent store
    # --------------------------------------------------------

    save_docstore(
        store,
        docstore_path,
    )

    # ========================================================
    # 5. MultiVectorRetriever
    # ========================================================

    retriever = MultiVectorRetriever(

        vectorstore=vectorstore,

        docstore=store,

        id_key="doc_id",
    )

    # ========================================================
    # 6. Existing conversational RAG
    # ========================================================

    return build_conversational_chain(
        retriever
    )


# ============================================================
# Load existing RAG
# ============================================================

def load_multimodal_rag(
    pdf_id: str,
):

    if not rag_exists(
        pdf_id
    ):

        raise FileNotFoundError(
            f"No persisted RAG found for PDF: {pdf_id}"
        )

    (
        pdf_path,
        faiss_path,
        docstore_path,
    ) = get_pdf_storage_paths(
        pdf_id
    )

    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    embedding = OpenAIEmbeddings()

    # --------------------------------------------------------
    # Load FAISS
    # --------------------------------------------------------

    vectorstore = FAISS.load_local(

        str(
            faiss_path
        ),

        embedding,

        allow_dangerous_deserialization=True,
    )

    # --------------------------------------------------------
    # Load parent store
    # --------------------------------------------------------

    store = load_docstore(
        docstore_path
    )

    # --------------------------------------------------------
    # Recreate MultiVectorRetriever
    # --------------------------------------------------------

    retriever = MultiVectorRetriever(

        vectorstore=vectorstore,

        docstore=store,

        id_key="doc_id",
    )

    # --------------------------------------------------------
    # Recreate conversational RAG
    # --------------------------------------------------------

    return build_conversational_chain(
        retriever
    )


# ============================================================
# Determine whether a string is an image
# ============================================================

def is_base64_image(
    value,
):
    """
    Determine whether a string contains one of the supported
    image formats.
    """

    if not isinstance(
        value,
        str,
    ):

        return False

    try:

        raw = base64.b64decode(
            value,
            validate=True,
        )

        if raw.startswith(
            b"\x89PNG"
        ):

            return True

        if raw.startswith(
            b"\xff\xd8"
        ):

            return True

        if raw.startswith(
            b"GIF"
        ):

            return True

        if raw.startswith(
            b"RIFF"
        ) and b"WEBP" in raw[:20]:

            return True

    except Exception:

        return False

    return False


# ============================================================
# Extract source information
# ============================================================

def extract_sources(
    docs,
):
    """
    Extract source information from the documents that were
    ALREADY retrieved by the existing RAG.

    No additional vector search is performed here.

    Therefore the existing retrieval flow is preserved.
    """

    sources = []

    for doc in docs:

        # ----------------------------------------------------
        # Source metadata attached to SourceString
        # ----------------------------------------------------

        if isinstance(
            doc,
            SourceString,
        ):

            metadata = getattr(
                doc,
                "source_metadata",
                {},
            )

            source_type = metadata.get(
                "type",
                "Document",
            )

            page_number = metadata.get(
                "page_number"
            )

        # ----------------------------------------------------
        # Normal Unstructured Document
        # ----------------------------------------------------

        elif hasattr(
            doc,
            "metadata",
        ):

            metadata = getattr(
                doc,
                "metadata",
                None,
            )

            source_type = "Text"

            page_number = (
                get_page_number(
                    metadata
                )
            )

        # ----------------------------------------------------
        # Plain string fallback
        # ----------------------------------------------------

        elif is_base64_image(
            doc
        ):

            source_type = "Image"

            page_number = None

        elif (
            isinstance(
                doc,
                str,
            )
            and "<table" in doc.lower()
        ):

            source_type = "Table"

            page_number = None

        else:

            source_type = "Text"

            page_number = None

        # ----------------------------------------------------
        # Format page
        # ----------------------------------------------------

        page_label = (
            format_page_number(
                page_number
            )
        )

        # ----------------------------------------------------
        # Create source object
        # ----------------------------------------------------

        source = {
            "type": source_type,

            "page": page_label,
        }

        # ----------------------------------------------------
        # Avoid duplicate sources
        # ----------------------------------------------------

        duplicate = any(
            s["type"] == source["type"]
            and s["page"] == source["page"]
            for s in sources
        )

        if not duplicate:

            sources.append(
                source
            )

    return sources


# ============================================================
# Build conversational RAG
# ============================================================

def build_conversational_chain(
    retriever,
):

    llm = ChatOpenAI(
        model="gpt-4.1-mini"
    )

    # ========================================================
    # STEP 1
    # Contextualize follow-up question
    # ========================================================

    contextualize_q_system_prompt = """

Given the chat history and the latest user question,
reformulate the question into a standalone question if needed.

Do NOT answer the question.

Return only the standalone question.
"""

    contextualize_q_prompt = (
        ChatPromptTemplate.from_messages(
            [

                (
                    "system",

                    contextualize_q_system_prompt,
                ),

                MessagesPlaceholder(
                    "chat_history"
                ),

                (
                    "human",

                    "{input}"
                ),
            ]
        )
    )

    # --------------------------------------------------------
    # Existing history-aware retriever
    # --------------------------------------------------------

    history_aware_retriever = (
        create_history_aware_retriever(

            llm,

            retriever,

            contextualize_q_prompt,
        )
    )

    # ========================================================
    # STEP 2
    # Parse retrieved documents
    # ========================================================

    def parse_docs(
        docs,
    ):

        images_b64 = []

        texts = []

        for doc in docs:

            try:

                base64.b64decode(
                    doc
                )

                images_b64.append(
                    doc
                )

            except Exception:

                texts.append(
                    doc
                )

        # ----------------------------------------------------
        # NEW:
        # Extract sources from the same docs.
        #
        # This does NOT perform another retrieval.
        # ----------------------------------------------------

        sources = extract_sources(
            docs
        )

        return {

            "images": images_b64,

            "texts": texts,

            "sources": sources,
        }

    # ========================================================
    # STEP 3
    # Existing multimodal prompt
    # ========================================================

    def build_prompt(
        inputs,
    ):

        context = inputs[
            "context"
        ]

        question = inputs[
            "input"
        ]

        combined_text = ""

        for text_element in (
            context["texts"]
        ):

            if hasattr(
                text_element,
                "text",
            ):

                combined_text += (
                    text_element.text
                    + "\n"
                )

            else:

                combined_text += (
                    str(
                        text_element
                    )
                    + "\n"
                )

        content = [

            {
                "type": "text",

                "text": (
                    "Answer using only this "
                    "context:\n\n"

                    f"{combined_text}\n\n"

                    f"Question: {question}"
                ),
            }
        ]

        # ----------------------------------------------------
        # Add images exactly as before
        # ----------------------------------------------------

        for image in context[
            "images"
        ]:

            content.append(

                {
                    "type": "image_url",

                    "image_url": {

                        "url": (
                            "data:image/png;base64,"
                            f"{image}"
                        )
                    },
                }
            )

        return (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "user",
                        content,
                    )
                ]
            )
        )

    # ========================================================
    # STEP 4
    # Existing final conversational RAG
    # ========================================================

    def multimodal_qa(
        inputs,
    ):

        # ----------------------------------------------------
        # Retrieve documents
        # ----------------------------------------------------

        docs = (
            history_aware_retriever.invoke(
                {

                    "input": inputs[
                        "input"
                    ],

                    "chat_history": inputs.get(
                        "chat_history",
                        [],
                    ),
                }
            )
        )

        # ----------------------------------------------------
        # Parse documents
        # ----------------------------------------------------

        parsed = parse_docs(
            docs
        )

        # ----------------------------------------------------
        # Build existing multimodal prompt
        # ----------------------------------------------------

        prompt = build_prompt(

            {
                "context": parsed,

                "input": inputs[
                    "input"
                ],
            }
        )

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        answer = (

            prompt

            | llm

            | StrOutputParser()

        ).invoke({})

        # ----------------------------------------------------
        # Return answer + existing context
        # + NEW sources metadata
        # ----------------------------------------------------

        return {

            "answer": answer,

            "context": parsed,
        }

    # ========================================================
    # Return LCEL Runnable
    # ========================================================

    return RunnableLambda(
        multimodal_qa
    )