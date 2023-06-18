import spacy
from src.constants import NUMBER_WORDS

nlp = spacy.load("es_core_news_sm")


def remove_stopwords_punctuation(text):
    if not text == "":
        doc = nlp(text)
        filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        filtered_text = " ".join(filtered_tokens)
        return filtered_text
    else:
        return text


def replace_number_words_with_ordinals(text):
    if not text == "":
        words = text.split()
        converted_words = []

        for word in words:
            if word.lower() in NUMBER_WORDS:
                number = NUMBER_WORDS[word.lower()]
                converted_words.append(number)
            else:
                converted_words.append(word)

        converted_text = " ".join(converted_words)
        return converted_text
    else:
        return text
