from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.schema import Document

def split_text_by_markdown(input_md: str) -> list:
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = splitter.split_text(input_md)
    documents = [Document(page_content=chunk.page_content, metadata=chunk.metadata) for chunk in chunks]
    return documents