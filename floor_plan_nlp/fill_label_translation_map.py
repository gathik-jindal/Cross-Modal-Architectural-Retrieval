import argparse
import asyncio
import re
import time
from pathlib import Path


CJK_PATTERN = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]+")
ENGLISH_END_PATTERN = re.compile(r"([A-Za-z]+)\s*$")
ENGLISH_COMPOUND_END_PATTERN = re.compile(r"([A-Za-z][A-Za-z0-9_-]*)\s*$")


def has_cjk(text):
    return bool(CJK_PATTERN.search(text))


def normalize_spaces(text):
    return re.sub(r"\s+", " ", text).strip()


def normalize_english_value(text):
    text = text.replace("_", " ").replace("-", " ")
    text = normalize_spaces(text)
    return text.lower()


def extract_last_dollar_term(key):
    parts = [p.strip() for p in key.split("$") if p.strip()]
    return parts[-1] if parts else key


def extract_terminal_english_word(key):
    match = ENGLISH_END_PATTERN.search(key)
    if not match:
        return None
    return match.group(1).lower()


def extract_terminal_english_compound(key):
    match = ENGLISH_COMPOUND_END_PATTERN.search(key)
    if not match:
        return None
    return match.group(1)


class TranslationBackend:
    def __init__(self):
        self.mode = None
        self.backend = None

        try:
            # Fast and stable sync wrapper over Google Translate endpoints.
            from deep_translator import GoogleTranslator

            self.mode = "deep-translator"
            self.backend = GoogleTranslator(source="auto", target="en")
            return
        except Exception:
            pass

        try:
            from googletrans import Translator

            self.mode = "googletrans"
            self.backend = Translator()
            return
        except Exception as exc:
            raise RuntimeError(
                "No translation backend found. Install one of:\n"
                "  pip install deep-translator\n"
                "  pip install googletrans==4.0.0rc1"
            ) from exc

    def translate(self, text, src_lang, dest_lang):
        if self.mode == "deep-translator":
            # deep-translator uses configured target; recreate only when needed.
            if dest_lang != "en":
                from deep_translator import GoogleTranslator

                self.backend = GoogleTranslator(source=src_lang, target=dest_lang)
            return self.backend.translate(text)

        result = self.backend.translate(text, src=src_lang, dest=dest_lang)
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
        return getattr(result, "text", str(result))


def safe_translate_text(translator, text, src_lang, dest_lang, cache):
    if text in cache:
        return cache[text]

    last_error = None
    for attempt in range(3):
        try:
            translated = translator.translate(text, src_lang=src_lang, dest_lang=dest_lang)
            translated = normalize_english_value(translated)
            cache[text] = translated
            return translated
        except Exception as exc:
            last_error = exc
            # Small backoff improves stability on intermittent API/rate issues.
            time.sleep(0.4 * (attempt + 1))

    raise RuntimeError(f"translate failed for '{text}': {last_error}")


def translate_cjk_chunks(translator, text, src_lang, dest_lang, cache):
    def replace_chunk(match):
        chunk = match.group(0)
        translated = safe_translate_text(translator, chunk, src_lang, dest_lang, cache)
        return translated

    replaced = CJK_PATTERN.sub(replace_chunk, text)
    return normalize_spaces(replaced)


def derive_value_for_key(key, translator, src_lang, dest_lang, cache):
    # Rule 2: If key contains '$', use only the last term.
    if "$" in key:
        tail = extract_last_dollar_term(key)
        if has_cjk(tail):
            return translate_cjk_chunks(translator, tail, src_lang, dest_lang, cache).lower()
        return normalize_english_value(tail)

    # Rule 1: If key ends with a proper English word, use that word directly.
    # For compound endings like 'window_text', keep the full compound as phrase.
    terminal_compound = extract_terminal_english_compound(key)
    if terminal_compound and ("_" in terminal_compound or "-" in terminal_compound):
        return normalize_english_value(terminal_compound)

    terminal_word = extract_terminal_english_word(key)
    if terminal_word:
        return terminal_word

    # Default: translate CJK chunks while keeping digits and symbols intact.
    if has_cjk(key):
        return translate_cjk_chunks(translator, key, src_lang, dest_lang, cache).lower()

    return normalize_english_value(key)


def collect_required_chunks_for_key(key):
    if "$" in key:
        tail = extract_last_dollar_term(key)
        if has_cjk(tail):
            return set(CJK_PATTERN.findall(tail))
        return set()

    terminal_compound = extract_terminal_english_compound(key)
    if terminal_compound and ("_" in terminal_compound or "-" in terminal_compound):
        return set()

    terminal_word = extract_terminal_english_word(key)
    if terminal_word:
        return set()

    if has_cjk(key):
        return set(CJK_PATTERN.findall(key))

    return set()


def fill_map_file(map_path, src_lang, dest_lang, overwrite_existing, sleep_seconds):
    path = Path(map_path)
    lines = path.read_text(encoding="utf-8").splitlines()

    translator = TranslationBackend()
    print(f"Using translation backend: {translator.mode}")
    cache = {}

    total_pairs = 0
    updated_pairs = 0
    skipped_existing = 0

    keys_to_update = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value.strip() and not overwrite_existing:
            continue
        keys_to_update.append(key.strip())

    # Prefetch unique CJK chunks once to avoid repeated network calls per line.
    required_chunks = set()
    for key in keys_to_update:
        required_chunks.update(collect_required_chunks_for_key(key))

    if required_chunks:
        print(f"Prefetching {len(required_chunks)} unique CJK chunks...")
        for idx, chunk in enumerate(sorted(required_chunks), start=1):
            try:
                safe_translate_text(translator, chunk, src_lang, dest_lang, cache)
            except Exception as exc:
                print(f"[WARN] Prefetch failed for '{chunk}': {exc}")
            if idx % 200 == 0 or idx == len(required_chunks):
                print(f"Prefetched {idx}/{len(required_chunks)} chunks")

    output_lines = []
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()

        if not stripped or stripped.startswith("#") or "=" not in line:
            output_lines.append(line)
            continue

        key, value = line.split("=", 1)
        total_pairs += 1

        if value.strip() and not overwrite_existing:
            skipped_existing += 1
            output_lines.append(line)
            continue

        try:
            new_value = derive_value_for_key(
                key=key.strip(),
                translator=translator,
                src_lang=src_lang,
                dest_lang=dest_lang,
                cache=cache,
            )
            updated_pairs += 1
        except Exception as exc:
            print(f"[WARN] Line {idx}: failed to translate key '{key.strip()}': {exc}")
            new_value = value.strip()

        output_lines.append(f"{key}={new_value}")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        if updated_pairs > 0 and updated_pairs % 100 == 0:
            print(f"Updated {updated_pairs} entries so far...")

    path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    print(f"Processed mappings: {total_pairs}")
    print(f"Updated values: {updated_pairs}")
    print(f"Skipped existing values: {skipped_existing}")
    print(f"Translation cache size: {len(cache)}")
    print(f"Updated map file: {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fill label translation map values using Google Translate and rule-based key parsing."
    )
    parser.add_argument(
        "--map-path",
        default="label_translation_map.txt",
        help="Path to label translation map file",
    )
    parser.add_argument(
        "--src-lang",
        default="auto",
        help="Source language code for Google Translate",
    )
    parser.add_argument(
        "--dest-lang",
        default="en",
        help="Destination language code",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite lines that already have a non-empty value",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between requests to reduce rate-limit risk",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fill_map_file(
        map_path=args.map_path,
        src_lang=args.src_lang,
        dest_lang=args.dest_lang,
        overwrite_existing=args.overwrite_existing,
        sleep_seconds=args.sleep_seconds,
    )
