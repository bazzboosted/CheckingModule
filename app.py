"""
Главный файл приложения.
Запуск: streamlit run app.py
"""

import streamlit as st
from similarity import extract_text, compute_similarity
from preprocessing import preprocess
from highlighting import highlight_text


st.set_page_config(
    page_title="Анализ сходства текстов",
    page_icon="📄",
    layout="wide"
)

st.title("Анализ текстового сходства документов")
st.markdown("Загрузите несколько файлов — приложение определит, насколько они похожи друг на друга.")
st.divider()

uploaded_files = st.file_uploader(
    "Загрузите документы (.docx, .pdf, .txt)",
    type=["docx", "pdf", "txt"],
    accept_multiple_files=True
)


def similarity_color(pct: float) -> str:
    if pct >= 70:
        return "🔴"
    elif pct >= 40:
        return "🟡"
    else:
        return "🟢"


def show_preview(fname: str, text: str, pair_f1: str, pair_f2: str, texts_by_name: dict):
    """
    Показывает содержимое файла с подсветкой совпадений относительно парного документа.
    """
    # Определяем с каким файлом сравнивать
    other_fname = pair_f2 if fname == pair_f1 else pair_f1
    other_text = texts_by_name.get(other_fname, "")

    st.markdown(f"** {fname}** — совпадения с *{other_fname}*")

    st.subheader("Текст с подсветкой совпадений")

    if other_text:
        with st.spinner("Поиск совпадений..."):
             html = highlight_text(text, other_text)
        st.components.v1.html(html, height=450, scrolling=True)
    else:
        st.text_area("", value=text[:5000], height=300, disabled=True)




if uploaded_files:
    if len(uploaded_files) < 2:
        st.warning("⚠ Загрузите хотя бы **2 файла** для сравнения.")
    else:
        st.success(f" Загружено файлов: {len(uploaded_files)}")

        if st.button(" Анализировать", type="primary", use_container_width=True):
            with st.spinner("Обработка текстов..."):
                texts, filenames, texts_by_name, errors = [], [], {}, []

                for f in uploaded_files:
                    text = extract_text(f)
                    if text.strip():
                        texts.append(text)
                        filenames.append(f.name)
                        texts_by_name[f.name] = text
                    else:
                        errors.append(f.name)

                if errors:
                    st.error(f"Не удалось прочитать файлы: {', '.join(errors)}")

                if len(texts) >= 2:
                    sim_df = compute_similarity(texts, filenames, method="auto")
                    pairs = []
                    for i in range(len(filenames)):
                        for j in range(i + 1, len(filenames)):
                            f1, f2 = filenames[i], filenames[j]
                            pairs.append((f1, f2, sim_df.loc[f1, f2]))
                    pairs.sort(key=lambda x: x[2], reverse=True)

                    st.session_state["pairs"] = pairs
                    st.session_state["texts_by_name"] = texts_by_name
                    st.session_state["open_preview"] = None

        if "pairs" in st.session_state:
            pairs = st.session_state["pairs"]
            texts_by_name = st.session_state["texts_by_name"]

            if "open_preview" not in st.session_state:
                st.session_state["open_preview"] = None

            st.subheader(" Рейтинг сходства документов")
            st.divider()

            for rank, (f1, f2, score) in enumerate(pairs, start=1):
                emoji = similarity_color(score)

                col_rank, col_info, col_score = st.columns([0.5, 5, 1.5])
                with col_rank:
                    st.markdown(f"### {rank}")
                with col_info:
                    st.markdown(f"**{f1}** — **{f2}**")
                    st.progress(int(score) / 100)
                with col_score:
                    st.markdown(f"## {emoji} {score:.1f}%")

                btn1, btn2, btn_close = st.columns([2, 2, 1])
                with btn1:
                    if st.button(f" {f1}", key=f"btn_{rank}_1"):
                        current = st.session_state["open_preview"]
                        st.session_state["open_preview"] = None if current == (rank, f1) else (rank, f1)
                        st.rerun()
                with btn2:
                    if st.button(f" {f2}", key=f"btn_{rank}_2"):
                        current = st.session_state["open_preview"]
                        st.session_state["open_preview"] = None if current == (rank, f2) else (rank, f2)
                        st.rerun()
                with btn_close:
                    open_prev = st.session_state["open_preview"]
                    if open_prev and open_prev[0] == rank:
                        if st.button("✖ Закрыть", key=f"close_{rank}"):
                            st.session_state["open_preview"] = None
                            st.rerun()

                # Превью прямо под кнопками своей пары
                open_prev = st.session_state["open_preview"]
                if open_prev and open_prev[0] == rank:
                    fname_to_show = open_prev[1]
                    show_preview(
                        fname_to_show,
                        texts_by_name[fname_to_show],
                        f1, f2,
                        texts_by_name
                    )

                st.divider()
