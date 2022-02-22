from numpy import array

from caption_preprocess import pad_text, training_dict, max_caption_words, tokenizer, vocab_size
from image_preprocess import img_path

def data_prep(data_dict, tokenizer, max_length, vocab_size):
    X = list() #features
    y = list() #clean & tokenized captions

    # For each image and list of captions
    for image_name, captions in data_dict.items():
        image_name = img_path + image_name + '.jpg'

        # For each caption in the list of captions
        for caption in captions:
            # tokenize
            word_idxs = tokenizer.texts_to_sequences([caption])[0]

            # Pad caption
            pad_idxs = pad_text(word_idxs, max_length)

            X.append(image_name)
            y.append(pad_idxs)
    return array(X), array(y)
    return X, y


train_X, train_y = data_prep(training_dict, tokenizer, max_caption_words, vocab_size)
