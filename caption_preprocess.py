"""
CLASS DÙNG ĐỄ PRE PROCESS CAPTIONS (CLEAN, TOKEN, PADDING, ETC...)
"""
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from caption_load import image_dict
from caption_split import training_image_names


# clean caption
def clean_captions(image_dict):
    # loop qua image dict
    for key, captions in image_dict.items():
        # loop qua từng caption
        for i, caption in enumerate(captions):
            # cho xuống lowercase, bỏ kí tự đặc biệt và punctuation (ly,es,...)
            caption_nopunct = re.sub(r"[^a-zA-Z0-9]+", ' ', caption.lower())

            # clean word, chỉ lấy word trên 1 chữ bỏ những chữ "a" và chữ có số
            clean_words = [word for word in caption_nopunct.split() if ((len(word) > 1) and (word.isalpha()))]

            # gắn lại thành chuỗi mới
            caption_new = ' '.join(clean_words)

            # đổi chuỗi cũ thành chuỗi đã clean
            captions[i] = caption_new


# add token bắt đầu và end caption
def add_token(captions):
    for i, caption in enumerate(captions):
        captions[i] = 'startseq ' + caption + ' endseq'
    return captions


# cho tên ảnh, return ảnh cùng caption như bên token
def subset_data_dict(image_dict, image_names):
    dict = {image_name: add_token(captions) for image_name, captions in image_dict.items() if image_name in image_names}
    return dict


# trả nguyên 1 list caption (ko có key, flat list)
def all_captions(data_dict):
    return [caption for key, captions in data_dict.items() for caption in captions]


# lấy max len của caption để padding
def max_caption_length(captions):
    return max(len(caption.split()) for caption in captions)


# tokenize dùng đễ encode caption
def create_tokenizer(data_dict):
    # lấy caption list
    captions = all_captions(data_dict)
    # lấy maxlen
    max_caption_words = max_caption_length(captions)

    # init token
    tokenizer = Tokenizer()

    # tokenize
    tokenizer.fit_on_texts(captions)

    vocab_size = len(tokenizer.word_index) + 1

    return tokenizer, vocab_size, max_caption_words


# padding text
def pad_text(text, max_length):
    text = pad_sequences([text], maxlen=max_length, padding='post')[0]
    return text


clean_captions(image_dict)
training_dict = subset_data_dict(image_dict, training_image_names)

# Tokenize
tokenizer, vocab_size, max_caption_words = create_tokenizer(training_dict)
