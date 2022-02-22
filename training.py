import time
import tensorflow as tf

from caption_prepare import train_X, vocab_size, tokenizer
from model import CNN_Encoder, RNN_Decoder
from prepare_dataset import dataset


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


loss_plot = []


@tf.function
def train_step(img_tensor, target):
    loss = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([tokenizer.word_index['startseq']] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)

            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


if __name__ == '__main__':
    start_epoch = 0
    epochs = 10000
    embedding_dim = 256
    units = 512
    vocab_size = vocab_size
    num_steps = len(train_X)  # BATCH_SIZE
    # Shape of the vector extracted from InceptionV3 is (64, 2048)
    # These two variables represent that vector shape
    features_shape = 2048
    attention_features_shape = 64
    checkpoint = None

    optimizer = tf.keras.optimizers.Adam(lr=0.01)

    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    checkpoint_path = "./checkpoints/"
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer,

                               )
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        print(f'Restored models from checkpoint: {ckpt_manager.latest_checkpoint}')
    else:
        print('Initialized from scratch.')

    for epoch in range(start_epoch, epochs):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(img_tensor, target)
            total_loss += t_loss

            if batch % 100 == 0:
                average_batch_loss = batch_loss.numpy() / int(target.shape[1])
                print(f'Epoch {epoch + 1} Batch {batch} Loss {average_batch_loss:.4f}')
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)

        print(f'Epoch {epoch + 1} Loss {total_loss / num_steps:.6f}')
        print(f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')

        # Checkpoint
        ckpt_manager.save()
