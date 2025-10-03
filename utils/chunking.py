from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
import tiktoken

def split_text_by_markdown(input_md: str, max_tokens: int = 2048, model: str = "cl100k_base") -> list:
    # Step 1: Split by headers
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    header_chunks = md_splitter.split_text(input_md)

    # Step 2: Tokenizer (OpenAI/Groq style)
    encoding = tiktoken.get_encoding(model)

    # Step 3: For each header chunk, further split if it’s too long
    final_docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # characters per chunk (roughly ~500 tokens, safe buffer)
        chunk_overlap=100   # overlap to preserve context
    )

    for chunk in header_chunks:
        token_count = len(encoding.encode(chunk.page_content))

        if token_count > max_tokens:
            # Split into smaller parts
            sub_chunks = text_splitter.split_text(chunk.page_content)
            for sub in sub_chunks:
                final_docs.append(
                    Document(page_content=sub, metadata=chunk.metadata)
                )
        else:
            # Keep as is
            final_docs.append(
                Document(page_content=chunk.page_content, metadata=chunk.metadata)
            )

    return final_docs
