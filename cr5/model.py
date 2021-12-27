from typing import List, Optional, Tuple
from collections import Counter
import pathlib
import gzip
import shutil

from cr5.settings import *
from cr5.errors import *
from cr5.utils import *

try:
    import plyvel
except ImportError:
    plyvel = None
import numpy as np
import faiss
from faiss.contrib.ondisk import merge_ondisk


class Cr5Model:
    def __init__(self,
                 model_name: str,
                 model_dir: Optional[str] = None,
                 level_db_dir: Optional[str] = None,
                 search_indexes_dir: Optional[str] = None):
        """
        Initialize a Cr5 model. The Cr5 model can work in-memory by loading embeddings from the original model lazily,
        or work on-disk by using the level db as backbone to retrieve embeddings.
        To enable the model to work in-memory, `model_dir` must be defined.
        To enable the model to work on-disk, `level_db_dir` must be defined.
        Either `model_dir` or `level_db_dir` must be defined to ensure the model work properly.
        Optionally, `search_indexes_dir` could be defined to store search indexes.
        :param model_name: The name of the Cr5 model. Available names include pairwise_2_*, joint_4, and joint_28.
        :param model_dir: The directory where the original Cr5 model is stored.
        :param level_db_dir: The directory where level db files are stored.
        :param search_indexes_dir: The directory where search indexes are stored.
        """
        if model_name not in SUPPORTED_LANGUAGES_PER_MODEL.keys():
            raise ModelNotSupportedError(model_name=model_name)
        self.model_name = model_name

        if model_dir and not pathlib.Path(model_dir).is_dir():
            raise PathError(path=model_dir)
        self.model_dir = model_dir

        if level_db_dir and not plyvel:
            raise LevelDbInitError()
        if level_db_dir and not pathlib.Path(level_db_dir).is_dir():
            raise PathError(path=level_db_dir)
        self.level_db_dir = level_db_dir

        if not is_valid_str(model_dir) and not is_valid_str(level_db_dir):
            raise NoModelFoundException()

        if search_indexes_dir and not pathlib.Path(search_indexes_dir).is_dir():
            raise PathError(path=search_indexes_dir)
        self.search_indexes_dir = search_indexes_dir

        self.lang_to_embeddings = {}
        self.in_memory_index = {}
        self.in_memory_keys = {}

    def get_document_embedding(self,
                               document: str,
                               lang_code: str,
                               in_memory_model: bool = False) -> np.ndarray:
        """
        Embed a document of given language. Errors or exceptions could be thrown if arguments or conditions are not met.
        The method works either in memory or on disk, controlled by the flag `in_memory_model`.
        If in memory model is used, the respective embedding file will be loaded lazily.
        Please be cautious with the memory usage if you want to use in memory model, since there is no emission policy
        to emit unused embeddings.
        :param document: The document to embed, and should not be empty.
        :param lang_code: The language of the given document.
        :param in_memory_model: Whether to use in memory model or not.
                                If `True`, `self.model_dir` must be set.
                                If `False`, `level_db_dir` must be set, and the related dependencies must be installed.
        :return: a numpy array representing the document embedding
        """
        if not is_valid_str(document):
            raise EmptyStringError(field='document')
        if not is_valid_str(lang_code):
            raise EmptyStringError(field='lang_code')

        if lang_code not in SUPPORTED_LANGUAGES_PER_MODEL[self.model_name]:
            raise LanguageNotSupportedInModelError(lang_code=lang_code, model_name=self.model_name)

        if in_memory_model and not self.model_dir:
            raise ModelDirectoryNotDefinedException()
        if not in_memory_model and not plyvel:
            raise LevelDbInitError()
        if not in_memory_model and not self.level_db_dir:
            raise LevelDbDirectoryNotDefinedException()

        tokens = tokenize_document(document)
        tf_tokens = dict(Counter(tokens))

        if in_memory_model and lang_code not in self.lang_to_embeddings.keys():
            file_name = _model_file_name(model_name=self.model_name, lang_code=lang_code)
            path = pathlib.Path(self.model_dir, file_name)
            if not path.is_file():
                raise ModelFileMissingException(model_file_name=file_name, model_file_dir=path.as_posix())

            embeddings = {}
            with gzip.open(path, mode='rt', encoding='utf-8') as file:
                for line in file:
                    parts = line.split(EMBEDDING_FILE_SEPARATOR)
                    word = EMBEDDING_FILE_SEPARATOR.join(parts[:-EMBEDDING_DIM])
                    embeddings[word] = np.array(parts[-EMBEDDING_DIM:], dtype=np.float32)
            self.lang_to_embeddings[lang_code] = embeddings

        words_in_vocab = []
        embeds_dict = {}

        if in_memory_model:
            words_in_vocab = [word for word in tf_tokens.keys() if word in self.lang_to_embeddings[lang_code]]
            embeds_dict = self.lang_to_embeddings[lang_code]
        else:
            # Assume that we have the top level `level_db_dir`, and the current model name is `model_name`,
            # embedding of each language belonging to the current model will be stored in:
            #               level_db_dir/model_name/models/
            db_path = pathlib.Path(self.level_db_dir, self.model_name, LEVEL_DB_MODEL_DIR)
            if not db_path.exists():
                raise LevelDbFileMissingException(db_file_dir=db_path.as_posix())
            db = plyvel.DB(db_path.as_posix())

            # Each language is prefixed with the prefix `{lang_code}-`
            pf_db = db.prefixed_db(_level_db_prefix(lang_code))
            if _is_empty(pf_db):
                raise EmptyDbException(lang_code=lang_code, db_file_dir=db_path.as_posix())

            for word in tf_tokens.keys():
                if word in embeds_dict.keys():
                    words_in_vocab.append(word)
                    continue
                emb = _embed_one_token_with_db(pf_db, word)
                if emb is None:
                    continue
                words_in_vocab.append(word)
                embeds_dict[word] = emb

        if len(words_in_vocab) <= 0:
            raise EmbeddingException()

        tfs = np.array([tf_tokens[word] for word in words_in_vocab])
        embs = np.array([embeds_dict[word] for word in words_in_vocab])
        for i in range(len(tfs)):
            embs[i] = embs[i] * tfs[i]
        doc_emb = embs.sum(axis=0)
        doc_emb = doc_emb / np.linalg.norm(tfs)
        return doc_emb

    def store_model_in_db(self,
                          lang_code: str,
                          model_dir: Optional[str] = None,
                          level_db_dir: Optional[str] = None):
        """
        Helper method to store the original Cr5 model into the local level db instance.
        If `self.model_dir` or `self.level_db_dir` is not defined before, they can be supplied here.
        :param lang_code: The language of the embedding file to store.
        :param model_dir: The path to the original Cr5 model.
        :param level_db_dir: The path to the level db file storage.
        """
        if lang_code not in SUPPORTED_LANGUAGES_PER_MODEL[self.model_name]:
            raise LanguageNotSupportedInModelError(lang_code=lang_code, model_name=self.model_name)
        if model_dir and not pathlib.Path(model_dir).is_dir():
            raise PathError(path=model_dir)
        if level_db_dir and not pathlib.Path(level_db_dir).is_dir():
            raise PathError(path=level_db_dir)
        if plyvel is None:
            raise LevelDbInitError()
        if model_dir:
            self.set_search_indexes_dir(model_dir)
        if level_db_dir:
            self.set_level_db_dir(level_db_dir)
        if not self.model_dir:
            raise ModelDirectoryNotDefinedException()
        if not self.level_db_dir:
            raise LevelDbDirectoryNotDefinedException()

        # Here we define the actual path to store for this model in level db.
        # Assume that we have the top level `level_db_dir`, and the current model name is `model_name`,
        # embedding of each language belonging to the current model will be stored in:
        #               level_db_dir/model_name/models/
        # The relevant directory will be created on demand.
        db_path = pathlib.Path(self.level_db_dir, self.model_name, LEVEL_DB_MODEL_DIR)
        if not db_path.exists():
            db_path.mkdir(parents=True)
        db = plyvel.DB(db_path.as_posix(), create_if_missing=True)

        # Instead of creating a separate db per language, we use the prefix db here in level db.
        # This means that different languages of the same Cr5 will be put in the same db.
        # Each language is prefixed with the prefix `{lang_code}-`
        # Keys and values will be stored in bytes,
        # with key being the encoded token and value being the byte representation of the numpy array.
        pf_db = db.prefixed_db(_level_db_prefix(lang_code))
        with pf_db.write_batch() as wb:
            file_name = _model_file_name(model_name=self.model_name, lang_code=lang_code)
            path = pathlib.Path(self.model_dir, file_name)
            if not path.is_file():
                raise ModelFileMissingException(model_file_name=file_name, model_file_dir=path.as_posix())

            with gzip.open(path, mode='rt', encoding='utf-8') as file:
                for line in file:
                    parts = line.split(EMBEDDING_FILE_SEPARATOR)
                    word = EMBEDDING_FILE_SEPARATOR.join(parts[:-EMBEDDING_DIM])
                    wb.put(word.encode(), np.array(parts[-EMBEDDING_DIM:], dtype=np.float32).tobytes())

    def create_search_indexes_on_disk(self,
                                      document_path: str,
                                      document_language: str,
                                      search_indexes_dir: Optional[str] = None,
                                      level_db_dir: Optional[str] = None,
                                      in_memory_model: bool = True,
                                      normalize: bool = True):
        """
        Given the path to a file containing a collection of documents, create the search index based on the file for
        similarity search.
        The search index will be stored on disk.
        If `self.search_indexes_dir` or `self.level_db_dir` is not defined before, they can be supplied here.
        :param document_path: The path to the file which contains a collection of documents of the same language.
        :param document_language: The language used in the file.
        :param search_indexes_dir: The path to store the search indexes.
        :param level_db_dir: The path to store level db files.
        :param in_memory_model: Whether to use in memory model for embedding or not.
                                For better performance, this is turned on by default.
        :param normalize: Whether to normalize each document embedding based on l-2 norm or not.
        """
        if not pathlib.Path(document_path).is_file():
            raise PathError(path=document_path)
        if document_language not in SUPPORTED_LANGUAGES_PER_MODEL[self.model_name]:
            raise LanguageNotSupportedInModelError(lang_code=document_language, model_name=self.model_name)
        if search_indexes_dir and not pathlib.Path(search_indexes_dir).is_dir():
            raise PathError(path=search_indexes_dir)
        if not search_indexes_dir and not self.search_indexes_dir:
            raise SearchIndexDirectoryNotDefinedException()
        if level_db_dir and not pathlib.Path(level_db_dir).is_dir():
            raise PathError(path=level_db_dir)
        if not level_db_dir and not self.level_db_dir:
            raise LevelDbDirectoryNotDefinedException()
        if plyvel is None:
            raise LevelDbInitError()

        if search_indexes_dir:
            self.set_search_indexes_dir(search_indexes_dir)
        if level_db_dir:
            self.set_level_db_dir(level_db_dir)

        document_keys, document_embeddings = self._process_document(document_path,
                                                                    document_language,
                                                                    in_memory_model,
                                                                    normalize)
        if normalize:
            index = faiss.index_factory(EMBEDDING_DIM, SEARCH_INDEX_ON_DISK_TYPE, faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.index_factory(EMBEDDING_DIM, SEARCH_INDEX_ON_DISK_TYPE)

        # Since we are using the index based on an inverted file (IVF), we need to initial clusters
        # The training size for 4096 clusters is by default 200,000.
        chosen_indices = np.random.choice(
            document_embeddings.shape[0],
            size=min(SEARCH_INDEX_TRAINING_DATA_SIZE, document_embeddings.shape[0]),
            replace=False
        )
        index.train(document_embeddings[chosen_indices])

        # We adopt the following file structure:
        #           search_indexes_dir/model_name/document_language
        # I.e., for each language of the model, we will have a separate directory
        # Thus, we will remove the existing directory (if any) before proceeding
        file_dir = pathlib.Path(search_indexes_dir, self.model_name, document_language)
        if file_dir.exists():
            shutil.rmtree(file_dir.as_posix())
        file_dir.mkdir(parents=True)
        faiss.write_index(index, file_dir.joinpath(SEARCH_INDEX_TRAINING_FILE_NAME).as_posix())

        # Split all embeddings based on a predefined `SEARCH_INDEX_TRUNK_SIZE`.
        # Increasing the trunk size would increase the memory usage.
        # To change the default setting, simply change the value defined in `SEARCH_INDEX_TRUNK_SIZE`.
        split_embeddings = np.split(document_embeddings,
                                    np.arange(
                                        min(SEARCH_INDEX_TRUNK_SIZE, document_embeddings.shape[0]),
                                        document_embeddings.shape[0],
                                        min(SEARCH_INDEX_TRUNK_SIZE, document_embeddings.shape[0]))
                                    )

        # Based on the split embeddings, write each block data in to file
        start_index, end_index = 0, 0
        for block, data in enumerate(split_embeddings):
            index = faiss.read_index(file_dir.joinpath(SEARCH_INDEX_TRAINING_FILE_NAME).as_posix())
            end_index = start_index + data.shape[0]
            index.add_with_ids(data, np.arange(start_index, end_index))
            faiss.write_index(index, file_dir.joinpath(_search_index_block_file_name(block)).as_posix())
            start_index = end_index

        index = faiss.read_index(file_dir.joinpath(SEARCH_INDEX_TRAINING_FILE_NAME).as_posix())
        block_file_names = [
            file_dir.joinpath(_search_index_block_file_name(block)).as_posix()
            for block in range(len(split_embeddings))
        ]

        # Merge blocks into one unified file, and remove the old block files
        merge_ondisk(index, block_file_names, file_dir.joinpath(SEARCH_INDEX_META_FILE_NAME).as_posix())
        faiss.write_index(index, file_dir.joinpath(SEARCH_INDEX_OUTPUT_FILE_NAME).as_posix())
        assert index.ntotal == document_embeddings.shape[0]

        for block in range(len(split_embeddings)):
            file_to_remove = file_dir.joinpath(_search_index_block_file_name(block))
            file_to_remove.unlink()

        # Store document id information into level db
        # Here we assume the following file structure:
        #           level_db_dir/model_name/ids
        # Each language will be stored in a prefixed db.
        db_path = pathlib.Path(self.level_db_dir, self.model_name, LEVEL_DB_DOCUMENT_ID_DIR)
        if not db_path.exists():
            db_path.mkdir(parents=True)
        db = plyvel.DB(db_path.as_posix(), create_if_missing=True)
        pf_db = db.prefixed_db(_level_db_prefix(document_language))
        with pf_db.write_batch() as wb:
            for i, v in enumerate(document_keys):
                wb.put(str(i).encode(), v.encode())

    def create_index_in_memory(self,
                               document_path: str,
                               document_language: str,
                               in_memory_model: bool = True,
                               normalize: bool = True):
        """
        Given the path to a file containing a collection of documents, create the search index based on the file for
        similarity search.
        The search index will be stored in memory.
        :param document_path: The path to the file which contains a collection of documents of the same language.
        :param document_language: The language used in the file.
        :param in_memory_model: Whether to use in memory model for embedding or not.
                                For better performance, this is turned on by default.
        :param normalize: Whether to normalize each document embedding based on l-2 norm or not.
        """
        if not pathlib.Path(document_path).is_file():
            raise PathError(path=document_path)
        if document_language not in SUPPORTED_LANGUAGES_PER_MODEL[self.model_name]:
            raise LanguageNotSupportedInModelError(lang_code=document_language, model_name=self.model_name)

        document_keys, document_embeddings = self._process_document(document_path,
                                                                    document_language,
                                                                    in_memory_model,
                                                                    normalize)
        if normalize:
            self.in_memory_index[document_language] = faiss.index_factory(EMBEDDING_DIM, SEARCH_INDEX_IN_MEMORY_TYPE,
                                                                          faiss.METRIC_INNER_PRODUCT)
        else:
            self.in_memory_index[document_language] = faiss.index_factory(EMBEDDING_DIM, SEARCH_INDEX_IN_MEMORY_TYPE)

        # Since we are using in memory search index here, there is no IVF optimization.
        # Every embedding will be put in memory
        self.in_memory_index[document_language].add(document_embeddings)
        self.in_memory_keys[document_language] = document_keys

    def set_search_indexes_dir(self, search_indexes_dir: str):
        """
        Set the search index directory
        :param search_indexes_dir: The search index directory to set
        :return:
        """
        if not pathlib.Path(search_indexes_dir).is_dir():
            raise PathError(path=search_indexes_dir)
        self.search_indexes_dir = search_indexes_dir

    def set_level_db_dir(self, level_db_dir: str):
        """
        Set the level db directory
        :param level_db_dir: The level db directory to set
        :return:
        """
        if not pathlib.Path(level_db_dir).is_dir():
            raise PathError(path=level_db_dir)
        self.level_db_dir = level_db_dir

    def search_similar_documents_on_disk(self,
                                         document: str,
                                         src_lang: str,
                                         dst_lang: str,
                                         normalize: bool = True) -> Tuple[List[Tuple[str, str]], int]:
        """
        Given a document of `src_lang`, search in the on disk document space of language `dst_lang`,
        and return the top results.
        :param document: The document represented in a string.
        :param src_lang: The language of the document.
        :param dst_lang: The language of the document to search for.
        :param normalize: Whether to normalize the document based on l-2 norm.
        :return: A tuple of search results and the size of search space.
                Inside the search results:
                            there is a list of tuples representing top matches.
                            Each tuple contains two fields: ID of the document and the title of the document.
        """
        if self.search_indexes_dir is None:
            raise SearchIndexDirectoryNotDefinedException()
        if self.level_db_dir is None:
            raise LevelDbDirectoryNotDefinedException()

        if src_lang not in SUPPORTED_LANGUAGES_PER_MODEL[self.model_name]:
            raise LanguageNotSupportedInModelError(lang_code=src_lang, model_name=self.model_name)

        if dst_lang not in SUPPORTED_LANGUAGES_PER_MODEL[self.model_name]:
            raise LanguageNotSupportedInModelError(lang_code=dst_lang, model_name=self.model_name)

        # Here we read the index located in:
        #       search_indexes_dir/model_name/dst_lang
        index_path = pathlib.Path(self.search_indexes_dir,
                                  self.model_name,
                                  dst_lang,
                                  SEARCH_INDEX_OUTPUT_FILE_NAME)
        if not index_path.exists():
            raise SearchIndexFileMissingException(search_index_file=index_path.as_posix())
        index = faiss.read_index(index_path.as_posix(), faiss.IO_FLAG_ONDISK_SAME_DIR)
        size = index.ntotal

        # The clusters to probe is defined in settings
        # With faiss, we get the numbered ids of these documents.
        # Numbers are in range: [0, document_size - 1]
        index.nprobe = SEARCH_INDEX_NPROBE
        query = normalize_vector(self.get_document_embedding(document, src_lang)) \
            if normalize else self.get_document_embedding(document, src_lang)
        _, ids = index.search(query.reshape(1, -1), SEARCH_RESULT_SIZE)

        # In order to return meaning for information of these documents,
        # we need to retrieve the metadata of these documents with level db, in folder:
        # level_db_dir/model_name/ids
        document_titles = []
        titles_path = pathlib.Path(self.level_db_dir, self.model_name, LEVEL_DB_DOCUMENT_ID_DIR)
        if not titles_path.exists():
            raise LevelDbFileMissingException(db_file_dir=titles_path.as_posix())
        db = plyvel.DB(titles_path.as_posix())
        pf_db = db.prefixed_db(_level_db_prefix(dst_lang))
        if _is_empty(pf_db):
            raise EmptyDbException(lang_code=dst_lang, db_file_dir=titles_path.as_posix())
        for _id in ids.flatten():
            title = pf_db.get(str(_id).encode()).decode().split(DOCUMENT_LINE_SEPARATORS)
            document_titles.append((str(title[0]), str(title[1])))
        return document_titles, size

    def search_similar_documents_in_memory(self,
                                           document: str,
                                           src_lang: str,
                                           dst_lang: str,
                                           normalize: bool = True) -> Tuple[List[Tuple[str, str]], int]:
        """
        Given a document of `src_lang`, search in the in memory document space of language `dst_lang`,
        and return the top results.
        :param document: The document represented in a string.
        :param src_lang: The language of the document.
        :param dst_lang: The language of the document to search for.
        :param normalize: Whether to normalize the document based on l-2 norm.
        :return: A tuple of search results and the size of search space.
                Inside the search results:
                            there is a list of tuples representing top matches.
                            Each tuple contains two fields: ID of the document and the title of the document.
        """
        if src_lang not in SUPPORTED_LANGUAGES_PER_MODEL[self.model_name]:
            raise LanguageNotSupportedInModelError(lang_code=src_lang, model_name=self.model_name)

        if dst_lang not in SUPPORTED_LANGUAGES_PER_MODEL[self.model_name]:
            raise LanguageNotSupportedInModelError(lang_code=dst_lang, model_name=self.model_name)

        if not self.in_memory_index[dst_lang] or not self.in_memory_keys[dst_lang]:
            raise InMemoryIndexMissingException()

        query = normalize_vector(self.get_document_embedding(document, src_lang)) \
            if normalize else self.get_document_embedding(document, src_lang)

        # We perform in memory search here with:
        #       - `self.in_memory_index[dst_lang]`
        #       - `self.in_memory_keys[dst_lang]`
        _, ids = self.in_memory_index[dst_lang].search(query.reshape(1, -1), SEARCH_RESULT_SIZE)
        size = self.in_memory_index[dst_lang].ntotal
        document_titles = []
        for _id in ids.flatten():
            title = self.in_memory_keys[dst_lang][_id].split(DOCUMENT_LINE_SEPARATORS)
            document_titles.append((str(title[0]), str(title[1])))
        return document_titles, size

    def _process_document(self,
                          document_path: str,
                          document_language: str,
                          in_memory_model: bool,
                          normalize: bool) -> Tuple[List[str], np.ndarray]:
        """
        Process the document at given path into a tuple of ids and numpy matrix.
        For the given document, we assume the following structure.
            - For each line, it represents a document
            - Inside each line, it is separated by `DOCUMENT_LINE_SEPARATORS`, which is `\t` in our case.
            - Each line has three columns, namely:
                - The title of the document of the current line
                - A unique identifier of the document of the current line
                - The document itself
        This can be achieved by preprocessing the document, or tweak this function,
        as long as the method signature remains the same.
        :param document_path: The path to the file.
        :param document_language: The language of the file.
        :param in_memory_model: Whether to use in memory model or not.
        :param normalize: Whether to normalize embeddings based on l-2 norm or not.
        :return: a tuple consisting of: the list of document keys, and a stacked numpy array of embeddings.
        """

        # `document_keys` uniquely identifies a document, `document_embeddings` stacks all embeddings
        document_keys, document_embeddings = [], []
        with open(pathlib.Path(document_path)) as f:
            for line in f:
                line_dsv = line.split(DOCUMENT_LINE_SEPARATORS)
                if len(line_dsv) != DOCUMENT_LINE_COLUMNS:
                    continue
                try:
                    emb = self.get_document_embedding(line_dsv[DOCUMENT_LINE_COLUMNS - 1],
                                                      document_language,
                                                      in_memory_model)
                    if normalize:
                        emb = normalize_vector(emb)
                    document_embeddings.append(emb)
                    document_keys.append(DOCUMENT_LINE_SEPARATORS.join(line_dsv[:DOCUMENT_LINE_COLUMNS - 1]))
                except ValueError:
                    pass

        assert len(document_keys) == len(document_embeddings)
        return document_keys, np.stack(document_embeddings)


def _model_file_name(model_name: str, lang_code: str) -> str:
    """
    Helper method to generate the original Cr5 model file name
    :param model_name: The name of the model.
    :param lang_code: The language to use.
    :return: The original Cr5 model file name.
    """
    return f'{model_name}_{lang_code}.txt.gz'


def _search_index_block_file_name(b: int) -> str:
    """
    Helper method to generate search index block file name.
    :param b: Block number.
    :return: The search index block file name.
    """
    return f'block_{b}.index'


def _level_db_prefix(lang: str) -> bytes:
    """
    Helper method to get the name of the prefix db of the current model.
    :param lang: The language belonging to the current model.
    :return: The name of the prefix db of the current model.
    """
    return f'{lang}-'.encode()


def _embed_one_token_with_db(db, word: str) -> Optional[np.ndarray]:
    """
    Helper method to embed one token with level db.
    :param db: The level db handler.
    :param word: The token to embed.
    :return: Either the embedding of the token or None if the token cannot be embedded.
    """
    value = db.get(word.encode())
    if not value:
        return None
    return np.frombuffer(value, dtype=np.float32)


def _is_empty(db) -> bool:
    """
    Helper method to check if the level db is empty (without any records).
    :param db: The level db handler.
    :return: True if the current level db is empty.
    """
    empty = True
    with db.iterator() as it:
        for _, _ in it:
            empty = False
            break
    return empty
