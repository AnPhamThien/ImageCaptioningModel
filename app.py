import flask
import numpy as np
import tensorflow as tf
import os
from flask import Flask, render_template, request
from caption_load import image_dict
from caption_preprocess import tokenizer, vocab_size, max_caption_words
from caption_split import subset_image_name
from image_preprocess import load_image, image_features_extract_model
from model import RNN_Decoder, CNN_Encoder


def evaluate(encoder, decoder, image, max_length):
    attention_plot = np.zeros((max_length, 64))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['startseq']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input,
                                                         features,
                                                         hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == 'endseq':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def check_test(encoder, decoder, test_image_names, image_dict, image_dir, max_caption_words):
    # captions on the validation set

    image_path = r"C:\Users\Pham Thien An\PycharmProjects\data\Images\667626_18933d713e.jpg"
    result, attention_plot = evaluate(encoder, decoder, image_path, max_caption_words)

    # from IPython.display import Image, display
    # display(Image(image_path))
    print('Image path:', image_path)
    #print('Real Caption:', real_caption)
    print('Prediction Caption:', ' '.join(result))


# if __name__ == '__main__':
#     embedding_dim = 256
#     units = 512
#     vocab_size = vocab_size
#     optimizer = tf.keras.optimizers.Adam()
#     encoder = CNN_Encoder(embedding_dim)
#     decoder = RNN_Decoder(embedding_dim, units, vocab_size)
#     checkpoint_path = './checkpoints'
#     string = 'abc'
#     ckpt = tf.train.Checkpoint(encoder=encoder,
#                                decoder=decoder,
#                                optimizer=optimizer)
#     #print(tf.train.CheckpointManager.latest_checkpoint())
#     ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)
#     ckpt.restore(ckpt_manager.latest_checkpoint)
#
#     test_image_name_file = "C:/Users/Pham Thien An/PycharmProjects/data/testImages.txt"
#     test_image_names = subset_image_name(test_image_name_file)
#     image_dir = "data/Images/"
#     print(max_caption_words)
#     check_test(encoder, decoder, list(test_image_names), image_dict, image_dir, max_caption_words)
#

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/after', methods=['GET', 'POST'])
def after():
    global vocab_size,CNN_Encoder,RNN_Decoder
    img = flask.request.files.get('file1', '')
    img.save('static/file.jpg')
    embedding_dim = 256
    units = 512
    vocab_size = vocab_size
    optimizer = tf.keras.optimizers.Adam()
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)
    checkpoint_path = './checkpoints'
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    result, attention_plot = evaluate(encoder, decoder, 'static/file.jpg', max_caption_words)
    print(result)
    return render_template('after.html', data=' '.join(result))
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0',port=port)
