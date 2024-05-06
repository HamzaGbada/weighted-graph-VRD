import nltk
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from nltk.tokenize import RegexpTokenizer
from torch import from_numpy

nltk.download('punkt')


# TODO: - add another embedding approaches
#       - apply other processing approaches
#       - add more params to the embedding model
#       - Check the word embedder of the transformer in PICK
#       - try to split the text before applying the embedding to get the same key as the train
class Doc2VecEncoder:

    def __init__(self, dataloader_set, out_dim: int) -> None:

        # remove punctuation
        text_units = [[i[0] for i in dataloader_set.data[j][1]['labels']] for j in range(len(dataloader_set.data))]
        tokenizer = RegexpTokenizer(r'\w+')
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(text_units)]
        model = Doc2Vec(documents, vector_size=out_dim, window=2, min_count=1)
        # logger.debug(f"The New Vec: {len(new_vec)}")

        self.model = model

    def __getitem__(self, word: str):
        word_vetor = from_numpy(self.model.wv[word])
        word_similarity = self.model.wv.most_similar(word)
        return word_vetor, word_similarity


