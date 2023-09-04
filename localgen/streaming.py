from typing import Generator
import ctransformers

def stream_response(llm: ctransformers.llm, g: Generator) -> str:
    resp = ""
    for token in g:
        detok = llm.detokenize(token)
        resp += detok
        print(detok, end="", flush=True)

    return resp