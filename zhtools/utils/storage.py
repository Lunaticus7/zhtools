from copy import deepcopy


class MemoryDocumentStorage():

    __slots__ = ('data')

    def __init__(self, data=None):
        self.data = {}

    def get_by_id(self, docid):
        return self.data.get(docid)

    def add_document(self, document, override=False):
        docid = document['id']
        if override or docid not in self.data:
            self.data[docid] = deepcopy(document)
            return True

        return False
