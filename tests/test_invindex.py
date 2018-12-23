from contextlib import contextmanager
from os.path import join
import shutil
import tempfile

import pytest

from schema import SchemaError
from zhtools.utils.inverted_index import InvertedIndex, FieldNotExistsError


@contextmanager
def tempdir():
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path)
        except IOError:
            pass


class TestInvertedIndex():

    def setup(self):
        self.schema = {'id': str, 'text': str, 'desc': str, 'cnt': int}
        self.index = InvertedIndex(self.schema)
        documents = [
            {'id': '1', 'text': 'first doc', 'desc': 'first desc', 'cnt': 4},
            {'id': '2', 'text': 'second doc', 'desc': 'second desc', 'cnt': 3},
            {'id': '3', 'text': 'third doc', 'desc': 'third desc', 'cnt': 2},
            {'id': '4', 'text': 'fourth doc', 'desc': 'fourth desc', 'cnt': 1},
        ]
        for doc in documents:
            self.index.add_document(doc)

    def test_dump_load(self):
        with tempdir() as base_dir:
            self.index.dump(join(base_dir, 'test.index'))
            self.index = InvertedIndex.load(join(base_dir, 'test.index'))

    def test_add_doc_warning(self):
        documents = [
            {'id': '1', 'text': 'some bad doc', 'desc': 'some desc', 'cnt': 1},
        ]
        for doc in documents:
            self.index.add_document(doc)

    def test_add_doc_error(self):
        bad_documents = [
            {'id': 5, 'text': 'some bad doc', 'desc': 'some desc', 'cnt': 1},
            {'id': '6', 'text': 666666, 'cnt': 1},
        ]
        with pytest.raises(SchemaError):
            for doc in bad_documents:
                self.index.add_document(doc)

    def test_retrieve(self):
        results = self.index.retrieve('first', limit=1)
        assert len(results) == 1 and results[0]['document']['id'] == '1'

        results = self.index.retrieve('first', metric_base='query', threshold=0.9)
        assert len(results) == 1 and results[0]['document']['id'] == '1'

        another_results = self.index.retrieve('first', metric_base='document', limit=1)
        assert len(another_results) == 1 and \
            another_results[0]['document']['id'] == '1' and \
            results[0]['score'] > another_results[0]['score']

        results = self.index.retrieve('first', rank_metric='cosine', limit=1)
        assert len(results) == 1 and results[0]['document']['id'] == '1'

    def test_retrieve_on_field(self):
        results = self.index.retrieve_on_field('first', 'text', limit=1)
        assert len(results) == 1 and results[0]['document']['id'] == '1'

        results = self.index.retrieve_on_field(1, 'text')
        assert not results

        results = self.index.retrieve_on_field('第一篇文章', 'text')
        assert not results

    def test_retrieve_on_field_error(self):
        with pytest.raises(FieldNotExistsError):
            self.index.retrieve_on_field('first', 'some field', limit=1)

    def test_match_on_field(self):
        results = self.index.match_on_field('id', '1')
        assert len(results) == 1 and results[0]['id'] == '1'

        results = self.index.match_on_field('text', 'first doc')
        assert len(results) == 1 and results[0]['id'] == '1'

        results = self.index.match_on_field('cnt', 4)
        assert len(results) == 1 and results[0]['id'] == '1'

        results = self.index.match_on_field('id', 1)
        assert not results

        results = self.index.match_on_field('text', 'first game')
        assert not results

        results = self.index.match_on_field('cnt', 5)
        assert not results

    def test_match_on_field_error(self):
        with pytest.raises(FieldNotExistsError):
            self.index.match_on_field('some field', 'some value')
