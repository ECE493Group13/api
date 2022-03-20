import io

import astroid
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm

SEED = 42
MAX_INFERRED = 500

astroid.context.InferenceContext.max_inferred = MAX_INFERRED


class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hparams):
        super().__init__()

        # Create embedding layers
        self.target_embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=1,
            name="w2v_embedding",
        )
        self.context_embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=hparams["num_neg_samples"] + 1,
        )

    def call(self, pair):
        target, context = pair
        # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
        # context: (batch, context)
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        # target: (batch,)
        word_emb = self.target_embedding(target)
        # word_emb: (batch, embed)
        context_emb = self.context_embedding(context)
        # context_emb: (batch, context, embed)
        dots = tf.einsum("be,bce->bc", word_emb, context_emb)
        # dots: (batch, context)
        return dots


def read_data(corpus_filename):
    data: pd.DataFrame = pd.read_csv(
        corpus_filename, sep="\t", names=["ngram_lc", "ngram_count"]
    )

    # Determine number of unique words and scale with ngram_count
    words = {}
    for row in data.itertuples():
        tokens = row.ngram_lc.split(" ")
        count = row.ngram_count
        for token in tokens:
            if token not in words:
                words[token] = count
            else:
                words[token] += count
    num_words = len(words.keys())
    data: pd.DataFrame = data.loc[data.index.repeat(data.ngram_count)].reset_index(
        drop=True
    )
    data = data.ngram_lc
    return data, num_words


def prepare_data(data, num_words):

    # Convert to tensorflow object
    ngrams_tf = tf.data.Dataset.from_tensor_slices((tf.cast(data.values, tf.string)))

    # Create text vectorziation layer
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=num_words,
        split="whitespace",
        output_mode="int",
        output_sequence_length=5,
    )
    vectorize_layer.adapt(ngrams_tf)

    text_vector_ds = (
        ngrams_tf.batch(1024).prefetch(tf.data.AUTOTUNE).map(vectorize_layer).unbatch()
    )
    sequences = list(text_vector_ds.as_numpy_iterator())
    return sequences, vectorize_layer


def generate_training_data(sequences, num_words, hparams, seed):
    targets, contexts, labels = [], [], []

    # Build the sampling table for `vocab_size` tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(
        num_words, sampling_factor=hparams["subsample"]
    )

    # Iterate over all sequences (sentences) in the dataset.
    for sequence in tqdm.tqdm(sequences):

        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=num_words,
            sampling_table=sampling_table,
            window_size=hparams["window_size"],
            negative_samples=0,
        )

        # Iterate over each positive skip-gram pair to produce training examples
        # with a positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype="int64"), 1
            )
            (
                negative_sampling_candidates,
                _,
                _,
            ) = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=hparams["num_neg_samples"],
                unique=True,
                range_max=num_words,
                seed=seed,
                name="negative_sampling",
            )

            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(
                negative_sampling_candidates, 1
            )

            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0] * hparams["num_neg_samples"], dtype="int64")

            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    targets = np.array(targets)
    contexts = np.array(contexts)[:, :, 0]
    labels = np.array(labels)
    training_data = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    training_data = training_data.shuffle(len(training_data)).batch(
        hparams["batch_size"]
    )
    # training_data = training_data.cache().prefetch(
    #     buffer_size=tf.data.AUTOTUNE
    # )
    return training_data


def train(corpus_filename, embeddings_filename, hparams):
    data, num_words = read_data(corpus_filename)
    sequences, vectorize_layer = prepare_data(data, num_words)
    training_data = generate_training_data(sequences, num_words, hparams, seed=SEED)
    word2vec = Word2Vec(num_words, hparams["embedding_size"], hparams)

    optimizer = tf.keras.optimizers.Adam(learning_rate=hparams["learning_rate"])
    word2vec.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        run_eagerly=True,
    )

    word2vec.fit(training_data, epochs=hparams["epochs_to_train"])

    weights = word2vec.get_layer("w2v_embedding").get_weights()[0]
    vocab = vectorize_layer.get_vocabulary()

    # out_v = io.open("vectors1.tsv", "w", encoding="utf-8")
    # out_m = io.open("metadata1.tsv", "w", encoding="utf-8")
    out_kv = io.open(embeddings_filename, "w", encoding="utf-8")

    for index, word in enumerate(vocab):
        if index == 0:
            out_kv.write(
                f"{vectorize_layer.vocabulary_size() - 1} {hparams['embedding_size']}\n"
            )
            continue
        vec = weights[index]
        # out_v.write("\t".join([str(x) for x in vec]) + "\n")
        # out_m.write(word + "\n")
        out_kv.write(f"{word} " + " ".join([str(x) for x in vec]) + "\n")
    # out_v.close()
    # out_m.close()
    out_kv.close()


# if __name__ == "__main__":
#     print("starting")
#     hparams = {
#         "embedding_size": 200,
#         "epochs_to_train": 20,
#         "learning_rate": 0.025,
#         "num_neg_samples": 25,
#         "batch_size": 500,
#         "concurrent_steps": 12,
#         "window_size": 5,
#         "min_count": 1,
#         "subsample": 1e-3,
#     }

#     train("test.txt", "embeddings.txt", hparams)
