# `EMBEDDING_DIM` defines the dimension of the embedding. This is a static value used by the original Cr5 model.
# `EMBEDDING_FILE_SEPARATOR` The file separator of the original Cr5 model.
EMBEDDING_DIM, EMBEDDING_FILE_SEPARATOR = 300, ' '

# `DOCUMENT_LINE_COLUMNS` defines the number of columns in the file to be indexed.
# `DOCUMENT_LINE_SEPARATORS` defines the line separator of each line.
DOCUMENT_LINE_COLUMNS, DOCUMENT_LINE_SEPARATORS = 3, '\t'

# Here we define the on disk search index characteristics
SEARCH_INDEX_ON_DISK_TYPE = 'IVF4096,Flat'
SEARCH_INDEX_TRAINING_DATA_SIZE = 200000
SEARCH_INDEX_TRUNK_SIZE = 500000
SEARCH_INDEX_TRAINING_FILE_NAME = 'trained.index'
SEARCH_INDEX_META_FILE_NAME = 'merged_index.ivfdata'
SEARCH_INDEX_OUTPUT_FILE_NAME = 'populated.index'

# The default type to be used in the in memory search index
SEARCH_INDEX_IN_MEMORY_TYPE = 'Flat'

# The parameters to be used when performing the search
SEARCH_INDEX_NPROBE = 32
MAX_SEARCH_RESULT_SIZE = 100

# The directory names inside each model's level db path
LEVEL_DB_DOCUMENT_ID_DIR = 'ids'
LEVEL_DB_MODEL_DIR = 'models'

# All available models and their supported languages
SUPPORTED_LANGUAGES_PER_MODEL = {
    'pairwise_2_en-it': {'en', 'it'},
    'pairwise_2_da-en': {'da', 'en'},
    'pairwise_2_da-vi': {'da', 'vi'},
    'joint_4': {'da', 'vi', 'en', 'it'},
    'joint_28': {'bg', 'ca', 'cs', 'da',
                 'de', 'el', 'en', 'es',
                 'et', 'fi', 'fr', 'hr',
                 'hu', 'id', 'it', 'mk',
                 'nl', 'no', 'pl', 'pt',
                 'ro', 'ru', 'sk', 'sl',
                 'sv', 'tr', 'uk', 'vi'},
}
