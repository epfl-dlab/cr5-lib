# Cr5 Library

This project implements a memory efficient Cr5 library. [Cr5](https://github.com/epfl-dlab/Cr5) provides cross-lingual document embedding and typically, it supports embedding documents from up to 28 different languages into a common vector space.

In addition to the basic usage of Cr5 model, which is to get a document embedding, we support multiple other usages of Cr5 model. In summary, usages include:
- Get a document embedding, based on the model name and the language.
- Store Cr5 models into a local database instance. In our case, we use [LevelDB](https://github.com/google/leveldb). This allows the user to use Cr5 models in an "off-line" mode, without the need to load model files in memory.
- Create search indexes for a corpus of documents. Thus, the user can find out the most similar document in the corpus given a query document. Cr5 supports two flavors of such search indexes, either on-disk or in-memory. In both cases, Cr5 uses [faiss](https://github.com/facebookresearch/faiss).
  - On-disk indexes help reduce the memory usage when searching for similar vectors in the search space. In addition, on-disk indexes uses inverted file (IVF) which performs approximate search.
  - In-memory indexes return the exact results. However, using in-memory indexes will consume much memory space, given that the corpus is huge.
- Search with indexes. Likewise, searching supports both in-memory mode and on-disk mode. With the on-disk indexes, searching in huge search space costs relatively minimal memory usage.


## Dependencies
The library depends on the following python packages:
- ```numpy```, for performing vector operations
- ```nltk```, for tokenizing documents
- ```faiss```, for creating indexes and searching in the indexes

If you want to run with the "off-line" mode without loading the Cr5 models in memory, both ```LevelDB``` and the python package ```plyvel``` are required.

To create the environment named `cr5` and install the dependencies, run the script below:

```bash
conda create -n cr5 python=3.6 -y
conda activate cr5

pip install numpy nltk
python -c "import nltk; nltk.download('punkt')"

# If you have LevelDB installed, proceed with installing plyvel
pip install plyvel

# The official way to install faiss
# Installing via pip is not recommended due to obsolete versions
conda install -c pytorch faiss-cpu -y
```

## Installation
To install the library, simply use:
```bash
pip install "git+https://github.com/epfl-dlab/cr5-lib"
```

## Example Usage
```python
from cr5.model import Cr5Model
from cr5.settings import SUPPORTED_LANGUAGES_PER_MODEL

MODEL_NAME = 'joint_4'

model = Cr5Model(
    model_name=MODEL_NAME, 
    model_dir='/path/to/original/Cr5/models',
    level_db_dir='/path/to/leveldb/files', # could be an empty directory, to be created later
    search_indexes_dir='/path/to/search_indexes/files', # could be an empty directory, to be created later
)

# First, we try with the model that is in-memory
emb = model.get_document_embedding(
    document='Some document in English', 
    lang_code='en', 
    in_memory_model=True,
)

# Next, let us store the model in LevelDB
for lang in SUPPORTED_LANGUAGES_PER_MODEL[MODEL_NAME]:
    model.store_model_in_db(lang)

# Now we can use the model in "off-line" mode
emb = model.get_document_embedding(
    document='Some document in English', 
    lang_code='en', 
    in_memory_model=False,
)

# We can create a search index for a corpus of documents
# The corpus is represented as a file as shown below
model.create_search_indexes_on_disk(
    document_path='/path/to/some/huge/corpus',
    document_language='language_of_the_corpus',
)

# Now let's search for the most similar documents in the corpus
top_matches, size_of_documents_in_corpus = model.search_similar_documents_on_disk(
    document="Some document to search for",
    src_lang="language_of_the_document_to_search",
    dst_lang="target_language_to_search",
)

# Creating an in-memory index is similar
model.create_search_indexes_in_memory(
    document_path='/path/to/some/huge/corpus',
    document_language='language_of_the_corpus',
)

# Searching in-memory index is similar
top_matches, size_of_documents_in_corpus = model.search_similar_documents_in_memory(
    document="Some document to search for",
    src_lang="language_of_the_document_to_search",
    dst_lang="target_language_to_search",
)
```

## Conversion to Database Records
The Cr5 Library reaches an efficient memory usage by utilizing the external database service to look up embeddings of tokens. Since the query itself is simple, we use a simple on-disk key-value storage: `LevelDB`.

The conversion is transparent to the user and is done by the call to the method `store_model_in_db(lang)`.

Under the hood, the conversion creates an entry per token:
- Key: the token itself, encoded in bytes
- Value: the serialized `numpy` array

For example, the original [Cr5 model file](https://zenodo.org/record/2597441#.Yco3xRPMJhE) uses a line to represent the token and its corresponding embedding, separated by the space. Each embedding has 300 dimensions. To convert the model to database records,
```python
# Create the database at some path
db = plyvel.DB(some_path, create_if_missing=True)
# Create a specific prefixed database for the English language
pf_db = db.prefixed_db('en'.encode())
with pf_db.write_batch() as wb:
    # Open the original model file
    with gzip.open(model_path, mode='rt', encoding='utf-8') as file:
        for line in file:
            parts = line.split(' ')
            # The token itself
            word = ' '.join(parts[:-300])
            # Here we store it in LevelDB
            wb.put(word.encode(), np.array(parts[-300:]).tobytes())
```

To read from `LevelDB`,
```python
# Get the serialized numpy bytes of a given token
value = db.get(word.encode())
# Restore it to a numpy array
vector = np.frombuffer(value, dtype=np.float32)
```

## Data Storage
The Cr5 Library depends on three directories.
- ```model_dir```: The path to the original [Cr5 model file](https://zenodo.org/record/2597441#.Yco3xRPMJhE). This is needed when storing models as LevelDB files or running the model in-memory.
- ```level_db_dir```: The path to store and read LevelDB files.
- ```search_indexes_dir```: The path to store and read faiss search indexes.

A typical and recommended layout would be:

------------

    └── data                              <- The root of data folder
        ├── models                        <- The directory which stores original models
        │   ├──joint_28_en.txt.gz
        │   ├──joint_4_fr.txt.gz
        │   └──...
        └── level_db                      <- The directory which stores LevelDB files
        │   ├──joint_4                    <- LevelDB files for model `joint_4`
        │   │    ├──ids                   <- Document IDs generated when creating search indexes
        │   │    └──models                <- Cr5 models stored as LevelDB files
        │   └──joint_28
        │        ├──ids
        │        └──models
        └── search_indexes                <- The directory which stores on-disk search indexes
            ├──joint_4                    <- Search indexes created using model `joint_4`
            │    ├──en                    <- Indexes for English corpus
            │    ├──it
            │    └──...
            └──joint_28
                 ├──en
                 ├──fr
                 └──...

------------
