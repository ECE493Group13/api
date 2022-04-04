"""
The implmentation of Word2Vec was based on the following tutorial by
TensorFlow: https://www.tensorflow.org/tutorials/text/word2vec and adapted
to fit our use case
"""
import astroid
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm

from api.word2vec.stopwords import stopwords

SEED = 42
MAX_INFERRED = 500

astroid.context.InferenceContext.max_inferred = MAX_INFERRED


class Word2Vec(tf.keras.Model):
    """Implmentation of this class is taken from TensorFlow's Word2Vec tutorial.
    https://www.tensorflow.org/tutorials/text/word2vec
    """

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
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        word_emb = self.target_embedding(target)
        # word_emb: (batch, embed)
        context_emb = self.context_embedding(context)
        # context_emb: (batch, context, embed)

        # dots = np.zeros((word_emb.shape[0], context_emb.shape[1]))
        # for b in range(word_emb.shape[0]):
        #    for c in range(context_emb.shape[1]):
        #        total = 0
        #        for e in range(word_emb.shape[1]):
        #            total += word_emb[b, e] * context_emb[b, c, e]
        #        dots[b, c] = total
        dots = tf.einsum("be,bce->bc", word_emb, context_emb)
        # dots: (batch, context)
        return dots


def process_data(corpus_filename, hparams):
    # Read in data
    data = read_data(corpus_filename)
    # Prepare data
    sequences, vectorize_layer, num_words = prepare_data(data, hparams)
    return sequences, vectorize_layer, num_words


def read_data(corpus_filename):
    data: pd.DataFrame = pd.read_csv(
        corpus_filename,
        sep="\t",
        names=["ngram_lc", "ngram_count"],
        dtype={"ngram_lc": str, "ngram_count": np.int32},
    )
    return data


def strip_punctuation(input_data):
    punctuation = r'[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^`{|}~\']'
    return tf.strings.regex_replace(input_data, punctuation, "")


def prepare_data(data, hparams):
    """This function processes the data by:
    1. Drop rows that are less than min_count
    2. Drop nan rows
    3. Remove stop words
    4. Duplicate rows by ngram_count
    5. Vectorize"""

    data = data.drop(data[data.ngram_count <= hparams["min_count"]].index)
    data = data.dropna()
    data["ngram_lc"] = data["ngram_lc"].apply(
        lambda x: " ".join([word for word in x.split() if word not in stopwords])
    )
    data: pd.DataFrame = data.loc[data.index.repeat(data.ngram_count)].reset_index(
        drop=True
    )
    data = data.ngram_lc
    num_words = len(data.str.split().explode().unique())

    # Convert to tensorflow object
    ngrams_tf = tf.data.Dataset.from_tensor_slices((tf.cast(data.values, tf.string)))

    # Create text vectorziation layer
    vectorize_layer = tf.keras.layers.TextVectorization(
        standardize=strip_punctuation,
        max_tokens=num_words,
        split="whitespace",
        output_mode="int",
        output_sequence_length=5,
    )
    vectorize_layer.adapt(ngrams_tf)

    text_vector_ds = (
        ngrams_tf.batch(hparams["batch_size"])
        .prefetch(tf.data.AUTOTUNE)
        .map(vectorize_layer)
        .unbatch()
    )
    sequences = list(text_vector_ds.as_numpy_iterator())
    return sequences, vectorize_layer, num_words


def generate_negative_skipgrams(skip_gram, num_words, hparams):
    """This function generates negative cases for each positive skip gram"""

    targets, contexts, labels = [], [], []
    for target_word, context_word in skip_gram:
        context_class = tf.expand_dims(tf.constant([context_word], dtype="int64"), 1)
        (negative_samples, _, _,) = tf.random.log_uniform_candidate_sampler(
            true_classes=context_class,
            num_true=1,
            num_sampled=hparams["num_neg_samples"],
            unique=True,
            range_max=num_words,
            seed=SEED,
            name="negative_sampling",
        )

        # Build context and label vectors (for one target word)
        negative_samples = tf.expand_dims(negative_samples, 1)
        negative_samples = np.array(negative_samples).flatten()
        context_class = np.array(context_class).flatten()

        context = np.concatenate((context_class, negative_samples), axis=0)
        label = np.array([1] + [0] * hparams["num_neg_samples"], dtype="int64")
        targets.append(target_word)
        contexts.append(context)
        labels.append(label)
    return targets, contexts, labels


def generate_positve_skipgrams(sequences, num_words, hparams):
    """This function loops through the sequences and generates
    skipgrams for each of the words"""

    skip_grams = []
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(
        num_words, sampling_factor=hparams["subsample"]
    )
    for sequence in tqdm.tqdm(sequences):
        # Generate skipgram for each sequence
        if len(sequence) == 0:
            continue
        skip_gram, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=num_words,
            sampling_table=sampling_table,
            window_size=hparams["window_size"],
            negative_samples=hparams["num_neg_samples"],
        )
        skip_grams.append(skip_gram)
    return skip_grams


def generate_training_data(sequences, num_words, hparams):
    """This function iterates through the sequences and generates positive and
    negative skipgrams and combines them into a tensor dataset"""

    targets, contexts, labels = [], [], []

    # Generate positive skipgrams
    positive_skip_grams = generate_positve_skipgrams(sequences, num_words, hparams)

    # For each positive skipgram, use the positive to generate a negative skipgram
    for skip_gram in tqdm.tqdm(positive_skip_grams):
        target, context, label = generate_negative_skipgrams(
            skip_gram, num_words, hparams
        )

        # Concatenate the results to a list
        targets = targets + target
        contexts = contexts + context
        labels = labels + label

    targets = np.array(targets)
    contexts = np.array(contexts)
    labels = np.array(labels)
    training_data = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    training_data = training_data.shuffle(len(training_data)).batch(
        hparams["batch_size"]
    )

    return training_data


def train(corpus_filename, embeddings_filename, hparams):
    """
    1. Process Data
        a. Read data
        b. Prepare data
    2. Generate Training data
        a. Generate positive skipgrams
        b. Generate negative skipgrams
    3. Run word2vec on training data
        a. Create a word2vec model
        b. fit the data
    4. Write the word embeddings to a file
    """
    sequences, vectorize_layer, num_words = process_data(corpus_filename, hparams)
    training_data = generate_training_data(sequences, num_words, hparams)
    word2vec = Word2Vec(num_words, hparams["embedding_size"], hparams)

    word2vec.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hparams["learning_rate"]),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        run_eagerly=True,
    )

    word2vec.fit(training_data, epochs=hparams["epochs_to_train"])

    weights = word2vec.get_layer("w2v_embedding").get_weights()[0]
    vocab = vectorize_layer.get_vocabulary()

    with open(embeddings_filename, "w", encoding="utf-8") as out_kv:
        for index, word in enumerate(vocab):
            if index == 0:
                out_kv.write(
                    f"{vectorize_layer.vocabulary_size() - 1} "
                    f"{hparams['embedding_size']}\n"
                )
                continue
            out_kv.write(f"{word} " + " ".join([str(x) for x in weights[index]]) + "\n")
