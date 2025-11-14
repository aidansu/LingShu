from elasticsearch_dsl import Document, Long


class BaseDoc(Document):
    id = Long()
