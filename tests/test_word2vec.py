from tempfile import NamedTemporaryFile

from api.word2vec import word2vec

CORPUS = """
chain or of nonconjugate polymer	1
nonconjugate polymer with fluorescent dye	1
dye in the side chain	1
allow for a precise adjustment	1
precise adjustment of emission color	1
focus of the research activity	1
research activity in the field	1
polymer led be the improvement	1
improvement of the quantum efficiency	1
quantum efficiency of both photoluminescence	1
strong increase of device efficiency	1
trapping of the radiative exciton	1
radiative exciton on the conjugate	1
exciton on the conjugate segment	1
"""


class TestWord2Vec:
    def test_word2vec(self):
        hparams = {
            "embedding_size": 1,
            "epochs_to_train": 1,
            "learning_rate": 0.025,
            "num_neg_samples": 1,
            "batch_size": 1,
            "concurrent_steps": 12,
            "window_size": 2,
            "min_count": 0,
            "subsample": 1e-3,
        }

        with NamedTemporaryFile("w") as data_input, NamedTemporaryFile() as emb:
            data_input.write(CORPUS)
            data_input.flush()
            word2vec.train(data_input.name, emb.name, hparams)
            assert emb.read().decode().splitlines()[:1] == ["27 1"]
