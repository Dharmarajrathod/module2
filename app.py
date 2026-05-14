import io
import os
import re
from dataclasses import dataclass
from typing import List, Sequence

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader


APP_TITLE = "Module 2 Chatbot"
DEFAULT_PDF_CANDIDATES = [
    "/Users/dharmarajrathod/Downloads/Module 2 - Revised.pdf",
    os.path.join(os.path.dirname(__file__), "Module 2 - Revised.pdf"),
]
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
SYSTEM_PROMPT = (
    "You are PA Coach, a training chatbot based strictly on Module 2 Clinical "
    "Documentation That Gets Approved content. Only answer using the provided "
    "PDF context. Do not use external knowledge. Do not generate or assume "
    "clinical facts. Always enforce documentation verification, chart review, "
    "and evidence verification before submission."
)
PHI_WARNING = "Remove real patient information. Use fictional or deidentified data."
MISSING_INFO = "Please verify in the provider record."
OUT_OF_SCOPE = (
    "I can only answer questions that are covered in Module 2. Ask about medical "
    "necessity, SBAR letters, documentation, specialty-specific PA requirements, "
    "or the evidence tools used in this module."
)
WELCOME_MESSAGE = (
    "Ask a Module 2 question about medical necessity, SBAR letters, "
    "specialty-specific PA documentation, or evidence tools."
)
TEMPORARY_MODEL_ERROR = (
    "The Gemini model is temporarily unavailable due to high demand. Please try "
    "again in a moment."
)
GENERIC_API_ERROR = (
    "The chatbot could not get a response from Gemini right now. Please try again."
)
MAX_CHUNKS = 4
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MAX_FALLBACK_SENTENCES = 5
STOPWORDS = {
    "a", "about", "an", "and", "are", "can", "does", "for", "give", "hello",
    "help", "hi", "how", "i", "in", "is", "me", "module", "of", "please",
    "summarize", "summary", "tell", "the", "there", "to", "what",
}


@dataclass
class Chunk:
    text: str
    tokens: set[str]
    index: int


def init_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("knowledge_chunks", [])
    st.session_state.setdefault("knowledge_label", None)
    st.session_state.setdefault("knowledge_text", "")


def normalize_text(text: str) -> str:
    cleaned = text.replace("\x00", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return normalize_text("\n".join(pages))


def extract_text_from_pdf_path(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return normalize_text("\n".join(pages))


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", text.lower()))


def meaningful_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOPWORDS}


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    if not text:
        return []

    chunks: List[Chunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(text=chunk_text, tokens=tokenize(chunk_text), index=index))
            index += 1
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def retrieve_relevant_chunks(query: str, chunks: Sequence[Chunk], limit: int = MAX_CHUNKS) -> List[Chunk]:
    query_tokens = meaningful_tokens(query) or tokenize(query)
    if not query_tokens:
        return list(chunks[:limit])

    scored = []
    for chunk in chunks:
        overlap = len(query_tokens & chunk.tokens)
        if overlap == 0:
            continue
        score = overlap / max(len(query_tokens), 1)
        scored.append((score, chunk))

    scored.sort(key=lambda item: (item[0], -item[1].index), reverse=True)
    if scored:
        return [chunk for _, chunk in scored[:limit]]
    return list(chunks[:limit])


def build_extractive_fallback(query: str, context_chunks: Sequence[Chunk]) -> str:
    query_tokens = meaningful_tokens(query) or tokenize(query)
    scored_sentences = []
    seen = set()

    for chunk in context_chunks:
        for sentence in split_sentences(chunk.text):
            normalized = sentence.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            sentence_tokens = meaningful_tokens(sentence) or tokenize(sentence)
            overlap = len(query_tokens & sentence_tokens)
            if overlap == 0 and query_tokens:
                continue
            score = overlap / max(len(query_tokens), 1)
            scored_sentences.append((score, sentence))

    if not scored_sentences:
        excerpt = context_chunks[0].text[:500].strip(
        ) if context_chunks else ""
        if not excerpt:
            return OUT_OF_SCOPE
        return f"Based on Module 2:\n\n{excerpt}..."

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    top_sentences = []
    for _, sentence in scored_sentences:
        top_sentences.append(sentence)
        if len(top_sentences) >= MAX_FALLBACK_SENTENCES:
            break

    return "Based on Module 2:\n\n- " + "\n- ".join(top_sentences)


def likely_contains_phi(text: str) -> bool:
    patterns = [
        r"\b\d{3}-\d{2}-\d{4}\b",
        r"\b(?:dob|date of birth)\b",
        r"\b(?:mrn|member id|patient id|policy id)\b",
        r"\b\d{2}/\d{2}/\d{4}\b",
        r"\b\d{10}\b",
        r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
        r"\b\d{1,5}\s+\w+\s+(?:street|st|road|rd|avenue|ave|drive|dr|lane|ln|blvd)\b",
    ]
    lowered = text.lower()
    if any(re.search(pattern, lowered) for pattern in patterns):
        return True

    capitalized_words = re.findall(r"\b[A-Z][a-z]+\b", text)
    digit_groups = re.findall(r"\d{4,}", text)
    return len(capitalized_words) >= 2 and len(digit_groups) >= 1


def is_greeting_or_smalltalk(text: str) -> bool:
    normalized = re.sub(r"[^a-z\s]", " ", text.lower()).strip()
    phrases = {
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
        "thanks", "thank you",
    }
    return normalized in phrases


def asks_for_case_specific_details(text: str) -> bool:
    patterns = [
        r"\bpatient\b",
        r"\bmember\b",
        r"\bclaim\b",
        r"\bchart\b",
        r"\bprovider record\b",
        r"\bcase\b",
        r"\bdob\b",
        r"\bmrn\b",
    ]
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


def build_grounded_prompt(user_query: str, context_chunks: Sequence[Chunk], history: Sequence[dict]) -> str:
    context_text = "\n\n".join(
        f"[Module Chunk {chunk.index + 1}]\n{chunk.text}" for chunk in context_chunks
    )
    context_block = (
        "Use only the following PDF excerpts as your knowledge source.\n\n"
        f"{context_text}\n\n"
        "If the user asks for case-specific or missing patient/provider record details "
        f"that are not in these excerpts, reply exactly with: \"{MISSING_INFO}\"\n"
        "If the question is outside Module 2 or not supported by these excerpts, reply "
        f"exactly with: \"{OUT_OF_SCOPE}\""
    )
    recent_history = history[-8:]
    history_text = "\n".join(
        f"{item['role'].upper()}: {item['content']}" for item in recent_history
    )
    return (
        f"{context_block}\n\n"
        f"Conversation so far:\n{history_text or 'No previous conversation.'}\n\n"
        f"User question:\n{user_query}"
    )


def ask_pa_coach(user_query: str, context_chunks: Sequence[Chunk], history: Sequence[dict]) -> str:
    client = genai.Client(api_key=os.getenv(
        "GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=build_grounded_prompt(user_query, context_chunks, history),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
        ),
    )
    return (response.text or "").strip()


def format_api_error(exc: Exception) -> str:
    error_text = str(exc).lower()
    if "503" in error_text or "unavailable" in error_text or "high demand" in error_text:
        return TEMPORARY_MODEL_ERROR
    if "429" in error_text or "quota" in error_text or "rate limit" in error_text:
        return "Gemini API quota or rate limit reached. Please check your API plan and try again."
    if "api key" in error_text or "authentication" in error_text or "permission" in error_text:
        return "Gemini API authentication failed. Check the API key and try again."
    return GENERIC_API_ERROR


def load_knowledge_base_from_upload(uploaded_file) -> None:
    pdf_bytes = uploaded_file.getvalue()
    extracted_text = extract_text_from_pdf_bytes(pdf_bytes)
    if not extracted_text:
        raise ValueError("No text could be extracted from the uploaded PDF.")
    st.session_state.knowledge_text = extracted_text
    st.session_state.knowledge_chunks = split_into_chunks(extracted_text)
    st.session_state.knowledge_label = uploaded_file.name
    st.session_state.messages = []


def load_knowledge_base_from_default_path(pdf_path: str) -> None:
    extracted_text = extract_text_from_pdf_path(pdf_path)
    if not extracted_text:
        raise ValueError("No text could be extracted from the default PDF.")
    st.session_state.knowledge_text = extracted_text
    st.session_state.knowledge_chunks = split_into_chunks(extracted_text)
    st.session_state.knowledge_label = os.path.basename(pdf_path)
    st.session_state.messages = []


def ensure_default_pdf_loaded() -> None:
    if st.session_state.knowledge_chunks:
        return
    for pdf_path in DEFAULT_PDF_CANDIDATES:
        if os.path.exists(pdf_path):
            load_knowledge_base_from_default_path(pdf_path)
            return


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_session_state()

    try:
        ensure_default_pdf_loaded()
    except Exception as exc:
        st.session_state.knowledge_chunks = []
        st.session_state.knowledge_label = None
        st.error(f"Default PDF load failed: {exc}")

    st.title("PA Coach")
    st.write(
        "Ask questions about Module 2 clinical documentation, medical necessity, "
        "SBAR letters, and evidence-tool use. This assistant uses only the loaded "
        "Module 2 PDF."
    )
    st.caption(
        f"Knowledge source: `{st.session_state.knowledge_label or 'Not loaded'}` | "
        f"Active model: `{DEFAULT_MODEL}`"
    )

    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        st.warning(
            "Set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable before starting a chat.")

    if not st.session_state.knowledge_chunks:
        st.info(
            "Upload the Module 2 PDF to begin, or add the PDF file to the app repository.")
        uploaded_file = st.file_uploader("Upload Module 2 PDF", type=["pdf"])
        if uploaded_file is not None:
            try:
                load_knowledge_base_from_upload(uploaded_file)
                st.rerun()
            except Exception as exc:
                st.error(f"Unable to load uploaded PDF: {exc}")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask a Module 2 training question")
    if not user_query:
        return

    cleaned_query = user_query.strip()
    if not cleaned_query:
        st.warning("Enter a question to continue.")
        return

    st.session_state.messages.append(
        {"role": "user", "content": cleaned_query})
    with st.chat_message("user"):
        st.markdown(cleaned_query)

    if likely_contains_phi(cleaned_query):
        assistant_reply = PHI_WARNING
    elif is_greeting_or_smalltalk(cleaned_query):
        assistant_reply = WELCOME_MESSAGE
    else:
        relevant_chunks = retrieve_relevant_chunks(
            cleaned_query, st.session_state.knowledge_chunks)
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            assistant_reply = "Gemini API key not configured. Set GEMINI_API_KEY or GOOGLE_API_KEY and try again."
        else:
            try:
                assistant_reply = ask_pa_coach(
                    cleaned_query,
                    relevant_chunks,
                    st.session_state.messages[:-1],
                )
            except Exception as exc:
                assistant_reply = format_api_error(exc)
                if assistant_reply == GENERIC_API_ERROR:
                    assistant_reply = build_extractive_fallback(
                        cleaned_query, relevant_chunks)

        if assistant_reply == OUT_OF_SCOPE and asks_for_case_specific_details(cleaned_query):
            assistant_reply = MISSING_INFO

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply})
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)


if __name__ == "__main__":
    main()
