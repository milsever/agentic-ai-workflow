from __future__ import annotations

import re

import requests
import streamlit as st

DEFAULT_API_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 180

def filter_referenced_sources(answer: str, sources: list[dict]) -> tuple[list[dict], bool]:
    """Return sources cited as [S1], [S2], etc. in the answer.

    The second value indicates whether any citation IDs were found in the answer.
    If no citation IDs are found, callers can choose a safe fallback display.
    """
    cited_ids = set(re.findall(r"\[(S\d+)\]", answer, flags=re.IGNORECASE))
    if not cited_ids:
        return [], False

    filtered = [
        source
        for source in sources
        if str(source.get("id", "")).upper() in cited_ids
    ]
    return filtered, True

st.set_page_config(
    page_title="Enterprise RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Enterprise RAG Knowledge Assistant")
st.caption("Akbank'ın kamuya açık dokümanları üzerinde çalışan belge tabanlı soru-cevap asistanı.")

with st.sidebar:
    st.header("Bağlantı ayarları")
    api_url = st.text_input(
        "FastAPI adresi",
        value=DEFAULT_API_URL,
        help="FastAPI sunucusunun adresi. Örneğin: http://127.0.0.1:8000",
    ).strip().rstrip("/")

    if st.button("API durumunu kontrol et", use_container_width=True):
        try:
            response = requests.get(
                f"{api_url}/health",
                timeout=5,
            )
            response.raise_for_status()
            health_data = response.json()
            if health_data.get("status") == "ok":
                st.success("API erişilebilir.")
            else:
                st.warning(f"API yanıtı: {health_data}")
        except requests.RequestException as exc:
            st.error(f"API'ye bağlanılamadı: {exc}")

    st.divider()
    st.caption("Önce FastAPI sunucusunu çalıştırın. Streamlit, soruları HTTP üzerinden /ask endpoint'ine gönderir.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        sources = message.get("sources", [])
        visible_sources, citations_found = filter_referenced_sources(
            message["content"], sources
        )
        if visible_sources:
            with st.expander(f"Kaynaklar ({len(visible_sources)})"):
                for source in visible_sources:
                    source_id = source.get("id", "S?")
                    source_name = source.get("source", "Bilinmeyen kaynak")
                    page = source.get("page", "Bilinmiyor")
                    text = source.get("content", "")

                    st.markdown(f"**[{source_id}] {source_name}** — Sayfa: {page}")
                    if text:
                        st.text(text)
                    st.divider()
        elif sources:
            st.caption(
                "Bu yanıtta eşleşen kaynak referansı bulunamadı; kaynak listesi gizlendi."
            )

question = st.chat_input("Dokümanlar hakkında bir soru sorun...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Dokümanlar aranıyor ve yanıt hazırlanıyor..."):
            try:
                response = requests.post(
                    f"{api_url}/ask",
                    json={"question": question},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                result = response.json()

                answer = result.get("answer") or "Bir yanıt döndürülmedi."
                sources = result.get("sources", [])
                if not isinstance(sources, list):
                    sources = []

                st.markdown(answer)

                visible_sources, citations_found = filter_referenced_sources(
                    answer, sources
                )
                if visible_sources:
                    with st.expander(
                            f"Kaynaklar ({len(visible_sources)})", expanded=False
                    ):
                        for source in visible_sources:
                            source_id = source.get("id", "S?")
                            source_name = source.get("source", "Bilinmeyen kaynak")
                            page = source.get("page", "Bilinmiyor")
                            text = source.get("content", "")

                            st.markdown(f"**[{source_id}] {source_name}** — Sayfa: {page}")
                            if text:
                                st.text(text)
                            st.divider()
                elif sources:
                    st.caption(
                        "Bu yanıtta eşleşen kaynak referansı bulunamadı; kaynak listesi gizlendi."
                    )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except requests.Timeout:
                error_message = (
                    "İstek zaman aşımına uğradı. LLM yanıt üretimi uzun sürebilir; "
                    "gerekirse REQUEST_TIMEOUT_SECONDS değerini artırın."
                )
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message, "sources": []}
                )
            except requests.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.response.json().get("detail", "")
                except (ValueError, AttributeError):
                    pass
                error_message = (
                        f"API isteği başarısız oldu (HTTP {exc.response.status_code})."
                        + (f" Ayrıntı: {detail}" if detail else "")
                )
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message, "sources": []}
                )
            except requests.RequestException as exc:
                error_message = (
                    f"FastAPI sunucusuna bağlanılamadı: {exc}. "
                    "API adresini ve sunucunun çalıştığını kontrol edin."
                )
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message, "sources": []}
                )
