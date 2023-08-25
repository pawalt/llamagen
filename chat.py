import prompting
import threading
from typing import List, Dict
import ctransformers
import streaming

def format_messages(messages) -> str:
    ret = ""
    for message in messages:
        ret += f"{message['role']}: {message['message']}\n"
    return ret

def get_chat_prompt() -> str:
    system_prompt = f"""
You are a helpful, smart assistant that can answer any question with perfect accuracy.
You are an expert Golang developer.
Respond with solely the answer to the user's question with no decoration text.
""".strip()

    return prompting.get_llama_system_prompt(system_prompt)

def run_chat(llm: ctransformers.LLM):
    cp = get_chat_prompt()
    print(cp)

    tokens = llm.tokenize(cp)
    system_preload = lambda: llm.eval(tokens)
    t = threading.Thread(target=system_preload)
    t.start()

    while True:
        user_input = input("human: ").strip()
        user_input += " [/INST]"

        tokens = llm.tokenize(user_input)

        t.join()

        print("ai: ", end="", flush=True)
        streaming.stream_response(llm, llm.generate(
            tokens=tokens,
            temperature=1,
            reset=False,
        ))


        streaming.stream_response(llm, llm.generate(
            tokens=llm.tokenize("what programming languages do you know"),
            temperature=1,
            reset=False,
        ))

        before_next = "<s>[INST]"
        tokens = llm.tokenize(before_next)
        llm.eval(tokens)

        print("")
