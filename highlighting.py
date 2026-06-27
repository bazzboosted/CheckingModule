"""
Модуль поиска и подсветки совпадающих фрагментов между двумя текстами.

Жёлтый — целое предложение ИЛИ 15+ значимых слов подряд
Красный — несколько предложений подряд ИЛИ 30+ значимых слов подряд

Подсвечивается весь диапазон текста (включая предлоги, союзы и пр.)
"""

import re
from preprocessing import clean_text, is_russian, stemmer_en, morph, STOP_WORDS
from nltk.tokenize import word_tokenize

SHINGLE_YELLOW = 15
SHINGLE_RED    = 30


# ── Базовые утилиты ──────────────────────────────────────────────────────────

def normalize_word(token: str) -> str:
    if is_russian(token):
        return morph.parse(token)[0].normal_form
    return stemmer_en.stem(token)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


def words_of_text(text: str) -> list[str]:
    """Все слова текста в нижнем регистре (включая стоп-слова)."""
    cleaned = clean_text(text)
    return word_tokenize(cleaned, language="russian")


def significant_words(word_list: list[str]) -> list[tuple[int, str]]:
    """
    Из списка всех слов возвращает только значимые (не стоп-слова, длина > 2).
    Возвращает список (индекс_в_исходном_списке, нормализованная_форма).
    """
    result = []
    for i, w in enumerate(word_list):
        if len(w) > 2 and w not in STOP_WORDS:
            result.append((i, normalize_word(w)))
    return result


# ── Поиск совпадений ─────────────────────────────────────────────────────────

def find_ranges_by_shingles(
    all_words_a: list[str],
    all_words_b: list[str],
    min_significant: int
) -> list[tuple[int, int]]:
    """
    Ищет диапазоны [start, end] в all_words_a где подряд идут
    min_significant совпадающих значимых слов с all_words_b.
    Возвращает диапазоны по индексам all_words_a (включая стоп-слова внутри).
    """
    sig_a = significant_words(all_words_a)
    sig_b = significant_words(all_words_b)

    normals_b = [w[1] for w in sig_b]
    shingles_b = set()
    for i in range(len(normals_b) - min_significant + 1):
        shingles_b.add(tuple(normals_b[i:i + min_significant]))

    normals_a = [w[1] for w in sig_a]
    matched_sig_indices = set()  # индексы в sig_a
    for i in range(len(normals_a) - min_significant + 1):
        if tuple(normals_a[i:i + min_significant]) in shingles_b:
            for pos in range(i, i + min_significant):
                matched_sig_indices.add(pos)

    if not matched_sig_indices:
        return []

    # Переводим индексы значимых слов → индексы в all_words_a
    matched_real_indices = {sig_a[i][0] for i in matched_sig_indices}

    # Склеиваем в непрерывные диапазоны (с допуском на стоп-слова между ними)
    return indices_to_ranges(matched_real_indices, gap=6)


def find_ranges_by_sentences(
    text_a: str,
    all_words_a: list[str],
    text_b: str,
    multi: bool = False
) -> list[tuple[int, int]]:
    """
    Ищет предложения из text_b которые целиком есть в text_a.
    multi=True — только блоки из 2+ предложений подряд (для красного).
    Возвращает диапазоны по индексам all_words_a.
    """
    sentences_b = [s for s in split_sentences(text_b) if len(s.split()) >= 5]
    sig_b_sets = {}
    for sent in sentences_b:
        words = words_of_text(sent)
        sig = tuple(w[1] for w in significant_words(words))
        if sig:
            sig_b_sets[sig] = words  # значимые слова → все слова предложения

    sig_a = significant_words(all_words_a)
    normals_a = [w[1] for w in sig_a]

    matched_real = set()
    # Храним какие позиции sig_a входят в совпавшие предложения
    sentence_hit_positions = []  # список: (start_sig_idx, end_sig_idx)

    for sent_sig in sig_b_sets:
        sent_len = len(sent_sig)
        for i in range(len(normals_a) - sent_len + 1):
            if tuple(normals_a[i:i + sent_len]) == sent_sig:
                sentence_hit_positions.append((i, i + sent_len - 1))

    if not sentence_hit_positions:
        return []

    if multi:
        # Оставляем только позиции где два попадания перекрываются или идут подряд
        merged = merge_sig_ranges(sentence_hit_positions, gap=3)
        long_enough = [(s, e) for s, e in merged if e - s >= 15]
        if not long_enough:
            return []
        for s, e in long_enough:
            for i in range(s, e + 1):
                if i < len(sig_a):
                    matched_real.add(sig_a[i][0])
    else:
        for s, e in sentence_hit_positions:
            for i in range(s, e + 1):
                if i < len(sig_a):
                    matched_real.add(sig_a[i][0])

    if not matched_real:
        return []

    return indices_to_ranges(matched_real, gap=6)


def merge_sig_ranges(ranges: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    """Сливает перекрывающиеся или близкие диапазоны."""
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [ranges[0]]
    for s, e in ranges[1:]:
        if s <= merged[-1][1] + gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def indices_to_ranges(indices: set[int], gap: int) -> list[tuple[int, int]]:
    """Превращает множество индексов в список диапазонов [start, end]."""
    if not indices:
        return []
    sorted_idx = sorted(indices)
    ranges = []
    start = end = sorted_idx[0]
    for idx in sorted_idx[1:]:
        if idx <= end + gap:
            end = idx
        else:
            ranges.append((start, end))
            start = end = idx
    ranges.append((start, end))
    return ranges


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Сливает пересекающиеся диапазоны."""
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [list(ranges[0])]
    for s, e in ranges[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(r) for r in merged]


# ── Сборка HTML ──────────────────────────────────────────────────────────────

def highlight_text(text: str, other_text: str) -> str:
    """
    Возвращает HTML с подсветкой и навигацией по совпадениям.
    """
    all_words_a = words_of_text(text)

    if not all_words_a:
        return f"<div style='padding:16px'>{text}</div>"

    all_words_b = words_of_text(other_text)

    red_ranges = merge_ranges(
        find_ranges_by_shingles(all_words_a, all_words_b, SHINGLE_RED) +
        find_ranges_by_sentences(text, all_words_a, other_text, multi=True)
    )
    yellow_ranges_raw = merge_ranges(
        find_ranges_by_shingles(all_words_a, all_words_b, SHINGLE_YELLOW) +
        find_ranges_by_sentences(text, all_words_a, other_text, multi=False)
    )

    red_set = set()
    for s, e in red_ranges:
        for i in range(s, e + 1):
            red_set.add(i)

    yellow_ranges = []
    for s, e in yellow_ranges_raw:
        chunk_start = None
        for i in range(s, e + 1):
            if i not in red_set:
                if chunk_start is None:
                    chunk_start = i
            else:
                if chunk_start is not None:
                    yellow_ranges.append((chunk_start, i - 1))
                    chunk_start = None
        if chunk_start is not None:
            yellow_ranges.append((chunk_start, e))

    mark_map = {}
    match_counter = 0

    for s, e in red_ranges:
        match_counter += 1
        for i in range(s, e + 1):
            mark_map[i] = ('red', match_counter, i == s)

    for s, e in yellow_ranges:
        match_counter += 1
        for i in range(s, e + 1):
            mark_map[i] = ('yellow', match_counter, i == s)

    total_red    = len(red_ranges)
    total_yellow = len(yellow_ranges)
    total_matches = total_red + total_yellow

    # Собираем HTML текста
    result_parts = []
    i = 0
    while i < len(all_words_a):
        word = all_words_a[i]
        if i in mark_map:
            color_type, match_num, is_start = mark_map[i]
            bg     = "#e53935" if color_type == "red" else "#fdd835"
            fg     = "white"   if color_type == "red" else "#333"
            anchor = f'id="match{match_num}"' if is_start else ""
            result_parts.append(
                f'<mark {anchor} style="background:{bg};color:{fg};'
                f'padding:1px 2px;border-radius:2px;">{word}</mark>'
            )
        else:
            result_parts.append(word)

        if i + 1 < len(all_words_a) and re.match(r'[а-яёa-z]', all_words_a[i + 1]):
            result_parts.append(" ")
        else:
            result_parts.append("")
        i += 1

    html_text = "".join(result_parts)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: sans-serif; font-size: 14px; background: white; }}

        #nav {{
            position: sticky;
            top: 0;
            z-index: 10;
            background: #f0f0f0;
            border-bottom: 1px solid #ddd;
            padding: 8px 12px;
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }}

        #nav .stat {{
            font-size: 13px;
            margin-right: 6px;
        }}

        #nav button {{
            padding: 4px 14px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            font-weight: bold;
            background: #555;
            color: white;
            transition: background 0.15s;
        }}
        #nav button:hover {{ background: #333; }}
        #nav button:disabled {{ background: #bbb; cursor: default; }}

        #counter {{
            font-size: 13px;
            color: #555;
            min-width: 60px;
        }}

        #content {{
            padding: 14px;
            line-height: 1.9;
            white-space: pre-wrap;
            overflow-y: auto;
            height: calc(100vh - 52px);
        }}

        mark {{
            border-radius: 2px;
            padding: 1px 2px;
        }}

        .current-match {{
            outline: 3px solid #1565c0;
            outline-offset: 1px;
        }}
    </style>
    </head>
    <body>

    <div id="nav">
        <span class="stat">🟡 <b>{total_yellow}</b></span>
        <span class="stat">🔴 <b>{total_red}</b></span>
        <button id="btn-prev" onclick="navigate(-1)">← Пред.</button>
        <span id="counter">–</span>
        <button id="btn-next" onclick="navigate(1)">След. →</button>
    </div>

    <div id="content">{html_text}</div>

    <script>
        const total = {total_matches};
        let current = 0;  // 0 = ничего не выбрано

        function getMatch(n) {{
            return document.getElementById('match' + n);
        }}

        function navigate(dir) {{
            if (total === 0) return;

            // Снимаем подсветку с текущего
            if (current > 0) {{
                const el = getMatch(current);
                if (el) el.classList.remove('current-match');
            }}

            // Вычисляем следующий (по кругу)
            current = current + dir;
            if (current < 1) current = total;
            if (current > total) current = 1;

            const el = getMatch(current);
            if (el) {{
                el.classList.add('current-match');
                el.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}

            document.getElementById('counter').textContent = current + ' / ' + total;
            document.getElementById('btn-prev').disabled = false;
            document.getElementById('btn-next').disabled = false;
        }}

        // Автоматически переходим к первому совпадению при открытии
        if (total > 0) {{
            setTimeout(() => navigate(1), 150);
        }}
    </script>

    </body>
    </html>
    """