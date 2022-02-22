import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from caption_load import captions_dict, count_words, images_features

# lấy max_lenght để padding
MAX_LEN = 0
for k, vv in captions_dict.items():
    for v in vv:
        if len(v) > MAX_LEN:
            MAX_LEN = len(v)

VOCAB_SIZE = len(count_words) + 1


# padding
def generator(photo, caption):
    X = []  # vetor ảnh
    y_in = []  # word fetch
    y_out = []  # generated word
    for k, vs in caption.items():  # từng ảnh
        for v in vs:  # từng caption trong ảnh
            for i in range(1, len(v)):  # từng từ trong caption
                X.append(photo[k])  # gán vector ảnh vào X
                in_seq = [v[:i]]  # split theo từng từ ví dụ 1,0,0,0,0 rồi tới 1,2,0,0,0,0

                out_seq = v[i]  # số thứ tự từ tiếp theo trong chuỗi ví dụ 0,1,0,0,0,0 rồi tới 0,0,1,0,0,0 ...

                in_seq = pad_sequences(in_seq, maxlen=MAX_LEN, padding='post', truncating='post')[
                    0]  # padding, số 0 để dằng sau
                out_seq = to_categorical([out_seq], num_classes=VOCAB_SIZE)[0]  # đánh số thứ tự cũng như categorize

                y_in.append(in_seq)
                y_out.append(out_seq)
    return X, y_in, y_out


X, y_in, y_out = generator(images_features, captions_dict)

X = np.array(X)
y_in = np.array(y_in, dtype='float64')
y_out = np.array(y_out, dtype='float64')
