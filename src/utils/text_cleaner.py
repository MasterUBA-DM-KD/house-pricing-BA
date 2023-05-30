import spacy

nlp =  spacy.load('es_core_news_sm')

def remove_stopwords_punctuaction(text):
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def replace_number_words_with_ordinals(text):
    number_words = {
        "cero": "0",
        "uno": "1",
        "dos": "2",
        "tres": "3",
        "cuatro": "4",
        "cinco": "5",
        "seis": "6",
        "siete": "7",
        "ocho": "8",
        "nueve": "9",
        # Add more number words as needed
    }
    
    words = text.split()
    converted_words = []
    
    for word in words:
        if word.lower() in number_words:
            number = number_words[word.lower()]
            converted_words.append(number)
        else:
            converted_words.append(word)
    
    converted_text = ' '.join(converted_words)
    return converted_text