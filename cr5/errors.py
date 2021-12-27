"""
This module defines some common errors or exceptions that may be thrown from the library.
This will help the caller to check meaningful error messages and types.
"""


class ModelNotSupportedError(ValueError):
    def __init__(self, model_name: str):
        self.message = f'The given model name: {model_name} is not supported.'

    def __str__(self) -> str:
        return str(self.message)


class PathError(ValueError):
    def __init__(self, path: str):
        self.message = f'The given path: {path} is not a valid directory.'

    def __str__(self) -> str:
        return str(self.message)


class LanguageNotSupportedInModelError(ValueError):
    def __init__(self, lang_code: str, model_name: str):
        self.message = f'The input language code {lang_code} is not supported in model {model_name}.'

    def __str__(self) -> str:
        return str(self.message)


class EmptyStringError(ValueError):
    def __init__(self, field: str):
        self.message = f'The field: {field} should be an non-empty string.'

    def __str__(self) -> str:
        return str(self.message)


class LevelDbInitError(Exception):
    def __init__(self):
        self.message = 'Level db fails to initialize. Please check your installation of leveldb and plyvel.'

    def __str__(self) -> str:
        return str(self.message)


class NoModelFoundException(Exception):
    def __init__(self):
        self.message = 'No usable model found. Either model_dir or level_db_dir must be defined.'

    def __str__(self) -> str:
        return str(self.message)


class ModelDirectoryNotDefinedException(Exception):
    def __init__(self):
        self.message = 'Model directory must be defined.'

    def __str__(self) -> str:
        return str(self.message)


class LevelDbDirectoryNotDefinedException(Exception):
    def __init__(self):
        self.message = 'Level db directory must be defined.'

    def __str__(self) -> str:
        return str(self.message)


class SearchIndexDirectoryNotDefinedException(Exception):
    def __init__(self):
        self.message = 'Search index directory must be defined.'

    def __str__(self) -> str:
        return str(self.message)


class ModelFileMissingException(Exception):
    def __init__(self, model_file_name: str, model_file_dir: str):
        self.message = f'Cannot find the required embedding file {model_file_name} under path: {model_file_dir}.'

    def __str__(self) -> str:
        return str(self.message)


class LevelDbFileMissingException(Exception):
    def __init__(self, db_file_dir: str):
        self.message = f'Cannot find the level db file under path: {db_file_dir}.'

    def __str__(self) -> str:
        return str(self.message)


class SearchIndexFileMissingException(Exception):
    def __init__(self, search_index_file: str):
        self.message = f'Cannot find the search index file under path: {search_index_file}.'

    def __str__(self) -> str:
        return str(self.message)


class InMemoryIndexMissingException(Exception):
    def __init__(self):
        self.message = f'In memory index is missing. You need to create in memory index first.'

    def __str__(self) -> str:
        return str(self.message)


class EmptyDbException(Exception):
    def __init__(self, lang_code: str, db_file_dir: str):
        self.message = f'The level db for language: {lang_code} under path: {db_file_dir} is empty.'

    def __str__(self) -> str:
        return str(self.message)


class EmbeddingException(Exception):
    def __init__(self):
        self.message = 'The vocabulary of the input document cannot be embedded.'

    def __str__(self) -> str:
        return str(self.message)
