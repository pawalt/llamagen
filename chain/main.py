import sys
sys.path.append('/Users/peyton/projects/llmkit')

from langchain.llms import CTransformers, LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from localgen import model_profiles, prompting
import os
import pathlib
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.embeddings import GPT4AllEmbeddings
from tqdm import tqdm
import json
import chromadb

ACTIVE_MODEL = model_profiles.CODE_LLAMA_7B
PERSIST_DIR = "code-db"
EMBEDDING_FUNCTION = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# EMBEDDING_FUNCTION = GPT4AllEmbeddings()

def get_chroma_store():
    return Chroma( 
        embedding_function=EMBEDDING_FUNCTION,
        collection_name="code-docs",
        persist_directory=PERSIST_DIR,
    )

def get_repo_docs():
    repo_base = "/Users/peyton/go/src/github.com/cockroachlabs/managed-service"
    subdirs = [
        "console",
        "pkg",
        "intrusion",
        "migration",
        "docs",
        "gen",
    ]
    total_files = []
    for subdir in subdirs:
        joined_path = os.path.join(repo_base, subdir)
        p = pathlib.Path(joined_path)
        go_files = p.glob("**/*.go")
        md_files = p.glob("**/*.md")
        total_files = total_files + list(go_files) + list(md_files)
    for fpath in total_files:
        with open(fpath, "r") as f:
            relpath = fpath.relative_to(repo_base)
            metadata_url = f"managed-service/{relpath}"
            yield Document(page_content=f.read(), metadata={"source": metadata_url})

def build_repo_chunks():
    source_chunks = []
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
    for doc in get_repo_docs():
        for chunk in splitter.split_text(doc.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=doc.metadata))
    return source_chunks

def create_db():
    docs_store = get_chroma_store()
    print("chunking")
    chunks = build_repo_chunks()
    print(f"found {len(chunks)} chunks")
    print("adding")
    batch_size = 1000
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]
        docs_store.add_documents(documents=batch, embeddings=EMBEDDING_FUNCTION)

    # save to disk
    print("persisting")
    docs_store.persist()

    print("done persisting")

def get_llm():
    # https://python.langchain.com/docs/integrations/llms/llamacpp
    return LlamaCpp(
        model_path=f"models/{ACTIVE_MODEL.model_path}",
        callbacks=[StreamingStdOutCallbackHandler()],
        n_gpu_layers=1,
#        grammar_path="grammars/json.gbnf",
        n_ctx=8192,
        max_tokens=512,
        f16_kv=True,
        verbose=False,
    )

def run_rag():
    llm = get_llm()
    docs_store = get_chroma_store()

    query = input("what would you like to know: ").strip()

    query_prompt = prompting.get_prompt_for_model(
        prompt_style=ACTIVE_MODEL.prompt_style,
        system_prompt="""you are an expert over the cockroachdb cloud codebase.
the user has asked you a question, and the first step to answering it is to retrieve
information about the question. respond with a natural-language search question to use
to get more information about the codebase.
""".strip(),
        message=query,
    ) + "Natural language query: "
    nl_query = llm(query_prompt).strip()

    found_docs = docs_store.similarity_search(nl_query, k=4)
    system_sum_prompt = """you are an expert over the cockroachdb cloud codebase.
the user has asked a question, and the following chunks from files have been provided as context.
use the information from the chunks to answer the user's question. respond with an answer
to the user's question.
"""
    for doc in found_docs:
        fname = doc.metadata["source"]
        content = doc.page_content
        system_sum_prompt += f"""
Filename: {fname}
Content: {content}
"""
    rag_prompt = prompting.get_prompt_for_model(
        prompt_style=ACTIVE_MODEL.prompt_style,
        system_prompt=system_sum_prompt,
        message=query,
    )
    answer = llm(rag_prompt)
    print(answer)

run_rag()
