from operator import itemgetter
from collections import defaultdict
import logging
import pickle

from schema import Schema

from zhtools.preprocess import to_halfwidth
from zhtools.tokenize import get_tokenizer
from zhtools.similarity import compute_similarity
from zhtools.utils.storage import MemoryDocumentStorage


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class FieldNotExistsError(KeyError):
    def __repr__(self):
        return '<FieldNotExistsError>'


class InvertedIndex():

    """
    Parameters
    ----------
    schema: dict
        文档的结构定义，指定文档每个字段的值类型，要求其中有 id 字段，
        且除 id 以外必须有其他字段，否则认为 schema 无效，目前仅支 int/float/str 三种类型

    storage: zhtools.utils.storage.Storage(optional)
        用于将存储实际的 document，并在检索时通过它来获得文档的实际内容（InvertedIndex内不存储
        document 的实际内容），若不指定则使用内建的 MemoryDocumentStorage。


    Instance Methods
    -------
    add_document(document)
        将一个文档添加到索引中

    retrieve(query, fields=None, limit=None, rank_metric='jaccard',
             metric_base='both', threshold=None)
        检索与 query 相关的文档

    retrieve_on_field(query, fields=None, limit=None, rank_metric='jaccard',
                      metric_base='both', threshold=None)
        根据指定 field 检索与 query 相关的文档

    match_on_field(field, value)
        获取指定字段值与 value 相等的文档

    dump(filename)
        将索引及 storage 保存到文件，storage 的保存行为由对应的类决定，如 MemoryDocumentStorage
        会将数据本身也一起保存到文件


    Class Methods
    -------------
    preprocess(text)
        在建立索引或检索文档时，对文本进行预处理以去除一些对检索可能无用的信息

    load(filename)
        从 dump 方法导出的索引文件中恢复 InvertedIndex 对象


    Examples
    --------
    In [1]: schema = {"id": str, "content": str}
    In [2]: inv_index = InvertedIndex(schema)
    In [3]: inv_index.add_document({"id": "1", "content": "hello world"})
    In [4]: inv_index.add_document({"id": "2", "content": "world wide web"})
    In [5]: inv_index.retrieve("world")
    Out[5]: [{"score": 0.5, "document": {"id": "1", "content": "hello world"}},
             {"score": 0.33, "document": {"id": "2", "content": "world wide web"}}]
    In [6]: inv_index.match("content", "hello world")
    Out[6]: [{"id": "2", "content": "world wide web"}]
    """

    FIELD_ID = 'id'
    METRICS = set(['lcs', 'jaccard', 'dice', 'cosine'])
    PREPROCESSORS = [to_halfwidth]

    __slots__ = ('schema', 'fields', 'storage', 'term_dict', 'index', 'tokenizer')

    def __init__(self, schema, storage=None):
        assert isinstance(schema, dict)
        assert self.FIELD_ID in schema and len(schema) > 1
        assert all(field_type in (str, int, float) for _, field_type in schema.items())

        self.schema = Schema(schema)
        self.fields = sorted(set(schema))

        self.storage = storage or MemoryDocumentStorage()

        self.term_dict = dict()
        self.index = [defaultdict(set) for _ in self.fields]
        self.tokenizer = get_tokenizer("ngram", level=2)

    @classmethod
    def preprocess(cls, text):
        for func in cls.PREPROCESSORS:
            text = func(text)

        return text

    def add_document(self, document):
        """将一个文档添加到索引中，同时会尝试将文档写入 storage 中"""
        self.schema.validate(document)

        if self.storage.get_by_id(document[self.FIELD_ID]):
            LOGGER.warning("document is already in storage: %r", document)

        docid = document[self.FIELD_ID]
        for field, value in document.items():
            if field == self.FIELD_ID:
                continue

            terms = []
            # 仅对 str 类型的 value 进行 tokenization
            if isinstance(value, str) and field != self.FIELD_ID:
                value = self.preprocess(value)
                terms = self.tokenizer.lcut(value)
            else:
                terms = [value]

            field_id = self.fields.index(field)
            for term in terms:
                if term in self.term_dict:
                    term_id = self.term_dict[term]
                else:
                    term_id = len(self.term_dict)
                    self.term_dict[term] = term_id
                self.index[field_id][term_id].add(docid)

        # 对接 mysql/ES 这种实际已经有文档数据的存储后端时，可以
        # 将 storage 设置为 read only，仅仅用 storage 来配合检索
        self.storage.add_document(document)

    def retrieve(self, query, fields=None, limit=None,
                 rank_metric='jaccard', metric_base='both', threshold=None):
        """检索与 query 相关的文档

        Parameters
        ----------
        query: any
            用于检索文档的值
        fields: list of str(optional)
            检索时考虑的文档字段，若不指定则使用所有字段
        limit: int(optional)
            返回结果的最大数量限制，若不设置则返回全部
        rank_metric: str(optional), default 'jaccard'
            计算检索结果与 query 相似度的方法，最终结果将按此进行排序
        metric_base: str(optional), default 'both'
            计算文档与 query 相似度时，以哪一方为准，有三个选项
            1. both: 计算对称的相似度，即 S(query, document)=S(document, query)
            2. query: 计算 document 与 query 的有偏的相似度
            3. document: 计算 query 与 document 的有偏的相似度
            例: query='abc', document='abdc', 使用 jaccard
            1. 选项为 both 时，得到的结果为 0.6
            2. 选项为 query 时，得到的结果为 1.0
            3. 选项为 document 时，得到的结果为 0.75
        threshold: float(optional)
            query 与文档的相似度阈值，若相似度低于阈值则不会被返回

        Return
        ------
        matches: list, 如: [{"document": <Document>, "score": 1.0}, ...]
        """
        documents = {}
        scores = defaultdict(float)
        for field in fields or self.fields:
            field_results = self.retrieve_on_field(query, field, limit=limit,
                                                   rank_metric=rank_metric,
                                                   metric_base=metric_base,
                                                   threshold=threshold)
            for match in field_results:
                document = match["document"]
                score = match["score"]
                docid = document[self.FIELD_ID]
                if docid not in documents:
                    documents[docid] = document
                    scores[docid] = score
                else:
                    scores[docid] = max(score, scores[docid])

        results = []
        for docid, score in scores.items():
            results.append(dict(document=documents[docid], score=score))

        limit = limit or len(documents)
        results.sort(key=itemgetter("score"), reverse=True)
        return results[:limit]

    def retrieve_on_field(self, query, field, limit=None,
                          rank_metric='jaccard', metric_base='both', threshold=None):
        """根据指定 field 检索与 query 相关的文档

        Parameters
        ----------
        query: any
            需匹配的字段的值
        field: str
            要匹配的文档的字段，若不存在触发 FieldNotExistsError 异常
        limit: int(optional)
            返回结果的最大数量限制，若不设置则返回全部
        rank_metric: str(optional), default 'jaccard'
            计算检索结果与 query 相似度的方法，最终结果将按此进行排序
        metric_base: str(optional), default 'both'
            计算文档与 query 相似度时，以哪一方为准，有三个选项
            1. both: 计算对称的相似度，即 S(query, document)=S(document, query)
            2. query: 计算 document 与 query 的有偏的相似度
            3. document: 计算 query 与 document 的有偏的相似度
            例: query='abc', document='abdc', 使用 jaccard
            1. 选项为 both 时，得到的结果为 0.6
            2. 选项为 query 时，得到的结果为 1.0
            3. 选项为 document 时，得到的结果为 0.75
        threshold: float(optional)
            query 与文档的相似度阈值，若相似度低于阈值则不会被返回

        Return
        ------
        matches: list, 如: [{"document": <Document>, "score": 1.0}, ...]
        """
        # field 不存在则抛异常
        if field not in self.fields:
            raise FieldNotExistsError

        # 若 value 与 schema 中 field value 的类型不一致，则认为无匹配结果
        if not isinstance(query, self.schema._schema[field]):
            return []

        # 若指定 field 为 id 或者 query 不是 str 类型，那么进行严格匹配
        if field == self.FIELD_ID or not isinstance(query, str):
            documents = self.match_on_field(field, query)
            results = [dict(document=doc, score=1.0) for doc in documents][:limit]
            return results if not limit else results[:limit]

        related, field_id = set(), self.fields.index(field)

        # 切分 terms 后寻找相关文档
        query = self.preprocess(query)
        terms = self.tokenizer.lcut(query)
        for term in terms:
            term_id = self.term_dict.get(term)
            if term_id is None:
                continue

            for docid in self.index[field_id][term_id]:
                related.add(docid)

        assert rank_metric in set(['lcs', 'jaccard', 'dice', 'lcs', 'cosine'])
        assert metric_base in set(['query', 'document', 'both'])

        # 准备 compute_similarity 的参数
        parameters = {"method": rank_metric}
        if metric_base in ('query', 'document'):
            parameters["partial"] = True
        if rank_metric == 'cosine':
            parameters["tokenizer"] = self.tokenizer

        # 对结果进行排序
        # TODO: 使用小顶堆优化内存占用和速度
        results = []
        for docid in related:
            document = self.storage.get_by_id(docid)
            text = self.preprocess(document[field])

            if metric_base == 'document':
                score = compute_similarity(text, query, **parameters)
            else:
                score = compute_similarity(query, text, **parameters)

            if threshold and score < threshold:
                continue

            results.append(dict(document=document, score=score))

        results.sort(key=itemgetter('score'), reverse=True)
        return results if not limit else results[:limit]

    def match_on_field(self, field, value):
        """查找对应字段值与 value 完全相等的文档

        Parameters
        ----------
        field: str
            要匹配的文档的字段，若不存在触发 FieldNotExistsError 异常
        value: any
            需匹配的字段的值

        Return
        ------
        documents: list
            匹配到的文档列表
        """
        # field 不存在则抛异常
        if field not in self.fields:
            raise FieldNotExistsError

        # 若 value 与 schema 中 field value 的类型不一致，则认为无匹配结果
        if not isinstance(value, self.schema._schema[field]):
            return []

        # 当 field 为 id 时，直接使用 storage 的方法来获取
        if field == self.FIELD_ID:
            document = self.storage.get_by_id(value)
            return [document] if document else []

        documents = []
        field_id = self.fields.index(field)

        # 当 value 为非字符串内容时，先从 index 中获得文档的 id，再取得文档
        if not isinstance(value, str):
            term_id = self.term_dict.get(value)
            if term_id is None:
                return []
            for docid in self.index[field_id].get(term_id, set()):
                documents.append(self.storage.get_by_id(docid))

            return documents

        # 当 value 为字符串内容时，将字符串切分为 term，检索出相关 docid
        text = self.preprocess(value)
        terms = self.tokenizer.lcut(text)
        related = defaultdict(int)
        for term in terms:
            # 文本中存在未索引的 term，认为不会有匹配的结果
            term_id = self.term_dict.get(term)
            if term_id is None:
                return []

            for docid in self.index[field_id][term_id]:
                related[docid] += 1

        for docid, score in related.items():
            # 若 docid 被匹配到的次数比 terms 少，说明 terms 中某个 term 在
            # 对应的 field value 中不存在，可以认为两者无法匹配
            if score < len(terms):
                continue

            document = self.storage.get_by_id(docid)
            if document[field] == value:
                documents.append(document)

        return documents

    def dump(self, filename):
        with open(filename, 'wb') as fout:
            pickle.dump(self, fout)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as fin:
            return pickle.load(fin)
