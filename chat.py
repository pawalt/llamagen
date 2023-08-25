import prompting
from typing import List, Dict
import ctransformers
import streaming

def format_messages(messages) -> str:
    ret = ""
    for message in messages:
        ret += f"{message['role']}: {message['message']}\n"
    return ret

def get_chat_prompt(messages: List[Dict], user_input: str) -> str:
    system_prompt = f"""
You are a helpful, smart assistant that can answer any question with perfect accuracy.
You are an expert Golang developer.
Respond with solely the answer to the user's question with no decoration text.

Previous messages:
{format_messages(messages)}
""".strip()

    return prompting.get_llama_prompt(
        system_prompt=system_prompt,
        message=user_input,
    )

def run_chat(llm: ctransformers.LLM):
    messages = [{
        "role": "assistant",
        "message": "How can I help you today?"
    }]
    while True:
        user_input = input("human: ")

        new_prompt = get_chat_prompt(messages, user_input)

        print(new_prompt)

        tokens = llm.tokenize(new_prompt)

        print("ai: ", end="", flush=True)
        resp = streaming.stream_response(llm, llm.generate(
            tokens=tokens,
            temperature=1,
        ))

        messages.append({
            "role": "human",
            "message": user_input,
        })

        messages.append({
            "role": "assistant",
            "message": resp,
        })

        print("")
