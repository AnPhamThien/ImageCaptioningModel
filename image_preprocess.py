import numpy as np
import tensorflow as tf
from caption_split import training_image_names
from tqdm import tqdm
from tensorflow.keras.applications import inception_v3

img_path = "data/Images/"


# load ảnh và preprocess ảnh
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = inception_v3.preprocess_input(img)
    return img, image_path


# transfer learning từ model CNN của imagenet
image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
# chỉ cần extract features chứ không cần classify layer nên bỏ layer cuối (classifier)
hidden_layer = image_model.layers[-1].output

# khai báo model lấy features
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

training_image_paths = [img_path + name + '.jpg' for name in training_image_names]

# Lấy ảnh trong tập train
encode_train = sorted(set(training_image_paths))  # sort dataset

image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)  # slice thành từng tuple

image_dataset = image_dataset.map(
    load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
    30)  # tuning dataset, 1 batch = 30 ảnh, xử lí song song tùy theo độ khỏe của CPU

# lấy feature của từng ảnh
for img, path in tqdm(image_dataset):  # thanh progressbar
    batch_features = image_features_extract_model(img)  # móc feature ra
    batch_features = tf.reshape(batch_features,
                                (batch_features.shape[0], -1, batch_features.shape[3]))  # reshape

    for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        np.save(path_of_feature, bf.numpy())  # lưu fetures vào file ảnh
