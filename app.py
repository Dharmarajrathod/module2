import io
import os
import re
import zipfile
from dataclasses import dataclass
from typing import List, Optional, Sequence
from xml.etree import ElementTree as ET

import streamlit as st
from google import genai
from google.genai import types


APP_TITLE = "Module 2 Chatbot"
DEFAULT_DOCX_CANDIDATES = [
    "/Users/dharmarajrathod/Downloads/Module 2 - Revised.docx",
    os.path.join(os.path.dirname(__file__), "Module 2 - Revised.docx"),
    os.path.join(os.path.dirname(__file__), "module2.docx"),
]
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
SYSTEM_PROMPT = (
    "You are PA Coach, a training chatbot based strictly on Module 2 Clinical "
    "Documentation That Gets Approved content. Only answer using the provided "
    "document context. Do not use external knowledge. Do not generate or assume "
    "clinical facts. Always enforce documentation verification, chart review, "
    "and evidence verification before submission."
)
PHI_WARNING = "Remove real patient information. Use fictional or deidentified data."
MISSING_INFO = "Please verify in the provider record."
OUT_OF_SCOPE = (
    "I can only answer questions that are covered in Module 2. Ask about "
    "medical necessity, SBAR letters, documentation quality, or the evidence tools "
    "used in this module."
)
WELCOME_MESSAGE = (
    "Ask a Module 2 question about medical necessity, SBAR letters, "
    "specialty-specific PA documentation, or the evidence tools."
)
TEMPORARY_MODEL_ERROR = (
    "The Gemini model is temporarily unavailable due to high demand. Please try "
    "again in a moment."
)
GENERIC_API_ERROR = (
    "The chatbot could not get a response from Gemini right now. Please try again."
)
MAX_CHUNKS = 4
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200
USE_GEMINI = os.getenv("MODULE2_USE_GEMINI", "false").lower() == "true"
STOPWORDS = {
    "a", "about", "an", "and", "are", "can", "does", "for", "give", "hello",
    "help", "hi", "how", "i", "in", "is", "me", "module", "of", "please",
    "summarize", "summary", "tell", "the", "there", "to", "what",
}
WORD_NAMESPACE = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
SECTION_HEADING_PATTERNS = [
    r"^MICRO MODULE \d",
    r"^AI TOOL DEMONSTRATION:",
    r"^CASE STUDY (?:EASY|HARD):",
    r"^Role Play:",
    r"^Quiz$",
    r"^Instructional Designer Notes$",
    r"^[A-Z][A-Za-z0-9 ,.&/\-]+ PA Requirements$",
    r"^Module Essence:",
    r"^Key Takeaways$",
]


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
    cleaned = cleaned.replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def read_document_xml_from_bytes(docx_bytes: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as archive:
        return archive.read("word/document.xml")


def read_document_xml_from_path(docx_path: str) -> bytes:
    with zipfile.ZipFile(docx_path) as archive:
        return archive.read("word/document.xml")


def extract_text_from_document_xml(document_xml: bytes) -> str:
    root = ET.fromstring(document_xml)
    paragraphs = []
    for paragraph in root.findall(".//w:p", WORD_NAMESPACE):
        text_nodes = [node.text for node in paragraph.findall(
            ".//w:t", WORD_NAMESPACE) if node.text]
        paragraph_text = " ".join(text_nodes).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)
    return normalize_text("\n".join(paragraphs))


def extract_text_from_docx_bytes(docx_bytes: bytes) -> str:
    return extract_text_from_document_xml(read_document_xml_from_bytes(docx_bytes))


def extract_text_from_docx_path(docx_path: str) -> str:
    return extract_text_from_document_xml(read_document_xml_from_path(docx_path))


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", text.lower()))


def meaningful_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOPWORDS}


def ordered_meaningful_tokens(text: str) -> List[str]:
    return [
        token for token in re.findall(r"[a-z0-9]{2,}", text.lower())
        if token not in STOPWORDS
    ]


def expand_query_tokens(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    if "documentation" in expanded:
        expanded.add("requirements")
    if "requirements" in expanded:
        expanded.add("documentation")
    if "sbar" in expanded:
        expanded.add("framework")
    if "trial" in expanded:
        expanded.add("evidence")
    return expanded


def normalize_phrase(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()


def is_section_heading(line: str) -> bool:
    return any(re.match(pattern, line) for pattern in SECTION_HEADING_PATTERNS)


def requested_specialty(tokens: set[str]) -> Optional[str]:
    for specialty in ("oncology", "cardiology", "neurology", "behavioral"):
        if specialty in tokens:
            return specialty
    return None


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    if not text:
        return []

    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    sections: List[str] = []
    current_section: List[str] = []

    for line in paragraphs:
        if current_section and is_section_heading(line):
            sections.append("\n\n".join(current_section))
            current_section = [line]
            continue
        current_section.append(line)

    if current_section:
        sections.append("\n\n".join(current_section))

    chunks: List[Chunk] = []
    index = 0

    for section in sections:
        if len(section) <= chunk_size:
            chunks.append(Chunk(text=section, tokens=tokenize(section), index=index))
            index += 1
            continue

        section_lines = [line.strip() for line in section.splitlines() if line.strip()]
        current_parts: List[str] = []
        current_length = 0
        for paragraph in section_lines:
            projected_length = current_length + len(paragraph) + (2 if current_parts else 0)
            if current_parts and projected_length > chunk_size:
                chunk_text = "\n\n".join(current_parts).strip()
                chunks.append(Chunk(text=chunk_text, tokens=tokenize(chunk_text), index=index))
                index += 1
                current_parts = []
                current_length = 0

            if current_parts:
                current_length += 2
            current_parts.append(paragraph)
            current_length += len(paragraph)

        if current_parts:
            chunk_text = "\n\n".join(current_parts).strip()
            chunks.append(Chunk(text=chunk_text, tokens=tokenize(chunk_text), index=index))
            index += 1

    return chunks


def retrieve_relevant_chunks(query: str, chunks: Sequence[Chunk], limit: int = MAX_CHUNKS) -> List[Chunk]:
    query_tokens = expand_query_tokens(meaningful_tokens(query) or tokenize(query))
    keyword_phrase = " ".join(ordered_meaningful_tokens(query))
    specialty = requested_specialty(query_tokens)
    if not query_tokens:
        return list(chunks[:limit])

    scored = []
    for chunk in chunks:
        overlap = len(query_tokens & chunk.tokens)
        if overlap == 0:
            continue
        coverage = overlap / max(len(query_tokens), 1)
        density = overlap / max(len(chunk.tokens), 1)
        lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
        heading = lines[0] if lines else ""
        heading_tokens = meaningful_tokens(heading) or tokenize(heading)
        heading_overlap = len(query_tokens & heading_tokens) / max(len(query_tokens), 1)
        best_line_overlap = 0.0
        for line in lines[:12]:
            line_tokens = meaningful_tokens(line) or tokenize(line)
            best_line_overlap = max(
                best_line_overlap,
                len(query_tokens & line_tokens) / max(len(query_tokens), 1),
            )

        normalized_query = normalize_phrase(keyword_phrase)
        normalized_heading = normalize_phrase(heading)
        phrase_bonus = 0.0
        if normalized_query and normalized_query in normalize_phrase(chunk.text):
            phrase_bonus += 0.2
        if normalized_query and normalized_heading.startswith(normalized_query):
            phrase_bonus += 0.25
        if specialty and specialty in normalized_heading and "requirements" in normalized_heading:
            phrase_bonus += 0.4

        score = (
            (coverage * 0.45)
            + (density * 0.1)
            + (heading_overlap * 0.3)
            + (best_line_overlap * 0.15)
            + phrase_bonus
        )
        scored.append((score, chunk))

    scored.sort(key=lambda item: (item[0], -item[1].index), reverse=True)
    if scored:
        return [chunk for _, chunk in scored[:limit]]
    return list(chunks[:limit])


def build_extractive_fallback(query: str, context_chunks: Sequence[Chunk]) -> str:
    if not context_chunks:
        return OUT_OF_SCOPE

    query_tokens = expand_query_tokens(meaningful_tokens(query) or tokenize(query))
    specialty = requested_specialty(query_tokens)
    if specialty:
        for chunk in context_chunks:
            heading = next((line.strip() for line in chunk.text.splitlines() if line.strip()), "")
            normalized_heading = normalize_phrase(heading)
            if specialty in normalized_heading and "requirements" in normalized_heading:
                excerpt, _ = extract_relevant_excerpt(query, chunk.text)
                if excerpt:
                    return f"Based on Module 2 (chunk {chunk.index + 1}):\n\n{excerpt}"

    ranked_excerpts = []
    for chunk in context_chunks:
        excerpt, excerpt_score = extract_relevant_excerpt(query, chunk.text)
        if excerpt:
            ranked_excerpts.append((excerpt_score, chunk, excerpt))

    if not ranked_excerpts:
        return OUT_OF_SCOPE

    ranked_excerpts.sort(key=lambda item: (item[0], -item[1].index), reverse=True)
    _, primary_chunk, excerpt = ranked_excerpts[0]
    if len(excerpt) > 1800:
        excerpt = excerpt[:1800].rstrip() + "..."
    return f"Based on Module 2 (chunk {primary_chunk.index + 1}):\n\n{excerpt}"


def extract_relevant_excerpt(query: str, text: str, window: int = 8) -> tuple[str, float]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "", -1.0

    query_tokens = expand_query_tokens(meaningful_tokens(query) or tokenize(query))
    keyword_phrase = " ".join(ordered_meaningful_tokens(query))
    best_index = 0
    best_score = -1.0

    for i, line in enumerate(lines):
        line_tokens = meaningful_tokens(line) or tokenize(line)
        overlap = len(query_tokens & line_tokens)
        score = overlap / max(len(query_tokens), 1)
        normalized_line = normalize_phrase(line)
        if keyword_phrase and keyword_phrase in normalized_line:
            score += 0.5
        if normalized_line == normalize_phrase(keyword_phrase):
            score += 0.75
        if i == 0:
            score += 0.15
        if score > best_score:
            best_score = score
            best_index = i

    start = max(best_index - 2, 0)
    end = min(best_index + window, len(lines))
    return "\n\n".join(lines[start:end]).strip(), best_score


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
        "Use only the following document excerpts as your knowledge source.\n\n"
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


def load_knowledge_base_from_upload(uploaded_file) -> None:
    docx_bytes = uploaded_file.getvalue()
    extracted_text = extract_text_from_docx_bytes(docx_bytes)
    if not extracted_text:
        raise ValueError("No text could be extracted from the uploaded DOCX.")
    st.session_state.knowledge_text = extracted_text
    st.session_state.knowledge_chunks = split_into_chunks(extracted_text)
    st.session_state.knowledge_label = uploaded_file.name
    st.session_state.messages = []


def load_knowledge_base_from_default_path(docx_path: str) -> None:
    extracted_text = extract_text_from_docx_path(docx_path)
    if not extracted_text:
        raise ValueError("No text could be extracted from the default DOCX.")
    st.session_state.knowledge_text = extracted_text
    st.session_state.knowledge_chunks = split_into_chunks(extracted_text)
    st.session_state.knowledge_label = os.path.basename(docx_path)
    st.session_state.messages = []


def ensure_default_docx_loaded() -> None:
    if st.session_state.knowledge_chunks:
        return
    for docx_path in DEFAULT_DOCX_CANDIDATES:
        if os.path.exists(docx_path):
            load_knowledge_base_from_default_path(docx_path)
            return


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_session_state()

    try:
        ensure_default_docx_loaded()
    except Exception as exc:
        st.session_state.knowledge_chunks = []
        st.session_state.knowledge_label = None
        st.error(f"Default document load failed: {exc}")

    st.title("PA Coach")
    st.write(
        "Ask questions about Module 2 clinical documentation, medical necessity, "
        "SBAR letters, and evidence-tool use. This assistant uses only the loaded "
        "Module 2 document."
    )
    st.caption(
        f"Knowledge source: `{st.session_state.knowledge_label or 'Not loaded'}` | "
        f"Answer mode: `{'Gemini grounded rewrite' if USE_GEMINI else 'Document extractive'}`"
    )

    if USE_GEMINI and not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        st.warning(
            "Set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable before starting a chat.")

    if not st.session_state.knowledge_chunks:
        st.info(
            "Upload the Module 2 DOCX to begin, or add the DOCX file to the app repository.")
        uploaded_file = st.file_uploader("Upload Module 2 DOCX", type=["docx"])
        if uploaded_file is not None:
            try:
                load_knowledge_base_from_upload(uploaded_file)
                st.rerun()
            except Exception as exc:
                st.error(f"Unable to load uploaded DOCX: {exc}")
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
        assistant_reply = build_extractive_fallback(
            cleaned_query, relevant_chunks)
        if USE_GEMINI and (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            try:
                gemini_reply = ask_pa_coach(
                    cleaned_query,
                    relevant_chunks,
                    st.session_state.messages[:-1],
                )
                if gemini_reply and gemini_reply not in {OUT_OF_SCOPE, MISSING_INFO}:
                    assistant_reply = gemini_reply
            except Exception:
                pass

        if assistant_reply == OUT_OF_SCOPE and asks_for_case_specific_details(cleaned_query):
            assistant_reply = MISSING_INFO

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply})
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)


if __name__ == "__main__":
    main()
