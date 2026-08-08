"""
Language configuration for CTM experiments.

FLEURS language IDs, tonal status, language family, and resource level
for the 20 languages selected for the paper (Table 1).

whisper_code: the language string Whisper's tokenizer actually uses.
  - Usually the first part of the FLEURS ID (e.g. "yo" from "yo_ng")
  - Exceptions that must be hardcoded:
      jv_id -> "jw"  (Whisper uses legacy ISO 639-1 "jw" for Javanese, not "jv")
      ig_ng -> None   (Igbo is NOT in Whisper-medium's language vocabulary;
                       Whisper was not trained on Igbo. WER evaluation will be
                       skipped for this language — similarity analysis still runs.)

Note on excluded languages:
  my_mm (Burmese) and lo_la (Lao) were removed because OpenAI's official evaluation
  uses CER not WER for these languages — they have no word boundaries and no
  established Python word segmentation library equivalent to PyThaiNLP for Thai.
  WER on them would produce invalid metrics. Replaced with Lingala and Shona.
  eu_es (Basque) was removed because it is not present in FLEURS as a dataset
  config. Replaced with Afrikaans (af_za), a Germanic language in FLEURS with
  Latin script, standard word boundaries, and Whisper code 'af'.
"""

# Full language list for paper experiments
LANGUAGES = {
    # --- TONAL ---
    "yo_ng": {
        "name": "Yoruba",
        "family": "Niger-Congo",
        "tonal": True,
        "resource": "Low",
        "tones": 3,
        "whisper_code": "yo",
    },
    "ig_ng": {
        "name": "Igbo",
        "family": "Niger-Congo",
        "tonal": True,
        "resource": "Very Low",
        "tones": 2,
        "whisper_code": None,   # Not in Whisper-medium vocab — skip WER eval
    },
    "vi_vn": {
        "name": "Vietnamese",
        "family": "Austroasiatic",
        "tonal": True,
        "resource": "Medium",
        "tones": 6,
        "whisper_code": "vi",
    },
    "th_th": {
        "name": "Thai",
        "family": "Tai-Kadai",
        "tonal": True,
        "resource": "Medium",
        "tones": 5,
        "whisper_code": "th",
    },
    "ha_ng": {
        "name": "Hausa",
        "family": "Chadic",
        "tonal": True,
        "resource": "Low",
        "tones": 2,
        "whisper_code": "ha",
    },
    "ln_cd": {
        "name": "Lingala",
        "family": "Bantu",
        "tonal": True,
        "resource": "Low",
        "tones": 2,
        "whisper_code": "ln",
    },
    "sn_zw": {
        "name": "Shona",
        "family": "Bantu",
        "tonal": True,
        "resource": "Low",
        "tones": 2,
        "whisper_code": "sn",
    },
    # --- NON-TONAL ---
    "sw_ke": {
        "name": "Swahili",
        "family": "Bantu",
        "tonal": False,
        "resource": "Medium",
        "tones": 0,
        "whisper_code": "sw",
    },
    "am_et": {
        "name": "Amharic",
        "family": "Semitic",
        "tonal": False,
        "resource": "Low",
        "tones": 0,
        "whisper_code": "am",
    },
    "jv_id": {
        "name": "Javanese",
        "family": "Austronesian",
        "tonal": False,
        "resource": "Very Low",
        "tones": 0,
        "whisper_code": "jw",   # Whisper uses legacy "jw", not "jv"
    },
    "ta_in": {
        "name": "Tamil",
        "family": "Dravidian",
        "tonal": False,
        "resource": "Medium",
        "tones": 0,
        "whisper_code": "ta",
    },
    "pa_in": {
        "name": "Punjabi",
        "family": "Indo-Aryan",
        "tonal": False,
        "resource": "Low",
        "tones": 0,
        "whisper_code": "pa",
    },
    "uz_uz": {
        "name": "Uzbek",
        "family": "Turkic",
        "tonal": False,
        "resource": "Low",
        "tones": 0,
        "whisper_code": "uz",
    },
    "kk_kz": {
        "name": "Kazakh",
        "family": "Turkic",
        "tonal": False,
        "resource": "Very Low",
        "tones": 0,
        "whisper_code": "kk",
    },
    "so_so": {
        "name": "Somali",
        "family": "Cushitic",
        "tonal": False,
        "resource": "Very Low",
        "tones": 0,
        "whisper_code": "so",
    },
    "is_is": {
        "name": "Icelandic",
        "family": "Germanic",
        "tonal": False,
        "resource": "Low",
        "tones": 0,
        "whisper_code": "is",
    },
    "cy_gb": {
        "name": "Welsh",
        "family": "Celtic",
        "tonal": False,
        "resource": "Low",
        "tones": 0,
        "whisper_code": "cy",
    },
    "af_za": {
        "name": "Afrikaans",
        "family": "Germanic",
        "tonal": False,
        "resource": "Low",
        "tones": 0,
        "whisper_code": "af",
    },
    "ka_ge": {
        "name": "Georgian",
        "family": "Kartvelian",
        "tonal": False,
        "resource": "Low",
        "tones": 0,
        "whisper_code": "ka",
    },
    "mt_mt": {
        "name": "Maltese",
        "family": "Semitic",
        "tonal": False,
        "resource": "Low",
        "tones": 0,
        "whisper_code": "mt",
    },
    # --- HIGH-RESOURCE ANCHORS (added for low- vs high-resource comparison) ---
    # Whisper-medium has tens of thousands of hours of training data per these.
    "en_us": {
        "name": "English",
        "family": "Germanic",
        "tonal": False,
        "resource": "High",
        "tones": 0,
        "whisper_code": "en",
    },
    "es_419": {
        "name": "Spanish",
        "family": "Romance",
        "tonal": False,
        "resource": "High",
        "tones": 0,
        "whisper_code": "es",
    },
    "de_de": {
        "name": "German",
        "family": "Germanic",
        "tonal": False,
        "resource": "High",
        "tones": 0,
        "whisper_code": "de",
    },
    "fr_fr": {
        "name": "French",
        "family": "Romance",
        "tonal": False,
        "resource": "High",
        "tones": 0,
        "whisper_code": "fr",
    },
    "cmn_hans_cn": {
        "name": "Mandarin Chinese",
        "family": "Sino-Tibetan",
        "tonal": True,
        "resource": "High",
        "tones": 4,
        "whisper_code": "zh",
    },
    # Note: Cantonese (yue_hant_hk) is NOT in whisper-medium's tokenizer
    # — the <|yue|> token resolves to UNK. Cantonese as a distinct language
    # tag was only added in whisper-large-v3. Skipping it here means the
    # "tonal-high-resource" cell has n=1 (Mandarin only).
}

# Step 1 subset: 5 languages for initial WER + similarity analysis
# All 5 are confirmed present in Whisper-medium's language vocab.
STEP1_LANGUAGES = ["yo_ng", "vi_vn", "th_th", "sw_ke", "uz_uz"]

# Helper views
TONAL_LANGS = [k for k, v in LANGUAGES.items() if v["tonal"]]
NONTONAL_LANGS = [k for k, v in LANGUAGES.items() if not v["tonal"]]

# Languages that can be evaluated with Whisper WER (whisper_code is not None)
WER_EVAL_LANGS = [k for k, v in LANGUAGES.items() if v["whisper_code"] is not None]


def get_lang_name(lang_id: str) -> str:
    return LANGUAGES[lang_id]["name"]


def is_tonal(lang_id: str) -> bool:
    return LANGUAGES[lang_id]["tonal"]


def get_whisper_code(lang_id: str) -> str | None:
    """
    Returns the language code string that Whisper's tokenizer expects.
    Returns None if this language is not in Whisper-medium's vocabulary
    (currently only Igbo / ig_ng).
    """
    return LANGUAGES[lang_id]["whisper_code"]
