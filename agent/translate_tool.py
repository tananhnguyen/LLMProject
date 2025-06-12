from transformers import pipeline
from deep_translator import GoogleTranslator  

def translate_ko_to_en(text: str) -> str:
    """Translates Korean text to English."""
    translated = GoogleTranslator(source='ko', target='en').translate(text)
    return translated

def translate_en_to_ko(text: str) -> str:
    """Translates Korean text to English."""
    translated = GoogleTranslator(source='en', target='ko').translate(text)
    return translated

# translator_eng_to_kr = pipeline(
#     "translation",
#     model="facebook/nllb-200-distilled-600M",
#     src_lang="eng_Latn",
#     tgt_lang="kor_Hang",
#     device=0  # Use GPU 0
# )

# translator_kr_to_eng = pipeline(
#     "translation",
#     model="facebook/nllb-200-distilled-600M",
#     src_lang="eng_Latn",
#     tgt_lang="kor_Hang",
#     device=0  # Use GPU 0
# )


