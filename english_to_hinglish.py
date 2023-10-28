import nltk
from nltk import word_tokenize, pos_tag
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"

checkpoint = 'facebook/nllb-200-3.3B'
# checkpoint = ‘facebook/nllb-200–1.3B’
# checkpoint = ‘facebook/nllb-200–3.3B’
# checkpoint = ‘facebook/nllb-200-distilled-1.3B’

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

source_lang = "eng_Latn"
target_lang = "hin_Deva"

# For CPU inference please remove device parameter
translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=source_lang, tgt_lang=target_lang, device=0, max_length = 400)

#function that returns nouns
def return_nouns(text : str) -> list:
    tokens = word_tokenize(text)
    parts_of_speech = nltk.pos_tag(tokens)
    nouns = list(filter(lambda x: x[1] == "NN" or x[1] == "NNS" or x[1] == "NNP" or x[1] == "NNPS", parts_of_speech))
    
    return nouns
    
#map key value pairs
def key_value_pair(nouns: list) -> dict:
    temp_dict = dict()
    for noun in nouns:
        value = translator(noun[0]+".")[0]['translation_text']
        if value[-1] == "." or value[-1] == "।":
            value = value[:-1]
        temp_dict[noun[0]] = value
        
    return temp_dict
    
#function for converting hindi text to hinglish
def hindi_to_hinglish(en_text: str, hi_text: str) -> str:
    #find nouns from english text
    nouns = return_nouns(en_text)
    #pair it with its hindi translation
    en_hi_transliteration = key_value_pair(nouns)
    
    hindi_text = hi_text.split()
    
    for k, v in en_hi_transliteration.items():
        for i in range(len(hindi_text)):
            if v in hindi_text[i]:
                hindi_text[i] = k
                
    return " ".join(hindi_text)
    
#main pipeline function
def english_to_hinglish(en_text: str) -> str:
    hi_text = translator(en_text)[0]['translation_text']
    hinglish_text = hindi_to_hinglish(en_text, hi_text)
    
    return hinglish_text
    
# Feel free to edit below input
input_text = "So even if it's a big video, I will clearly mention all the products."

if __name__ == "__main__":
    print(english_to_hinglish(input_text))
