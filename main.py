from ctransformers import AutoModelForCausalLM
from typing import List, Dict
import prompting

NOUS_HERMES_7B = "nous-hermes-llama-2-7b.ggmlv3.q4_K_M.bin"
CODE_LLAMA_7B = "codellama-7b-instruct.Q5_K_M.gguf"
CODE_LLAMA_13B = "codellama-13b-instruct.Q5_K_M.gguf"

active_model = CODE_LLAMA_13B

llm = AutoModelForCausalLM.from_pretrained(f"models/{active_model}", model_type='llama', gpu_layers=1)

def format_messages(messages) -> str:
    ret = ""
    for message in messages:
        ret += f"{message['role']}: {message['message']}\n"
    return ret

def get_chat_prompt(messages: List[Dict], user_input: str) -> str:
    system_prompt = f"""
You are a helpful, smart assistant that can answer any question with perfect accuracy.
Respond with solely the answer to the user's question with no decoration text.

Previous messages:
{format_messages(messages)}
""".strip()

    return prompting.get_llama_prompt(
        system_prompt=system_prompt,
        message=user_input,
    )

messages = [{
    "role": "assistant",
    "message": "How can I help you today?"
}]
while True:
    user_input = input("human: ")

    new_prompt = get_chat_prompt(messages, user_input)

    print(new_prompt)

    tokens = llm.tokenize(new_prompt)

    resp = ""
    print("ai: ", end="", flush=True)
    for token in llm.generate(
        tokens=tokens,
        temperature=0.7,
    ):
        detok = llm.detokenize(token)
        resp += detok
        print(detok, end="", flush=True)

    messages.append({
        "role": "human",
        "message": user_input,
    })

    messages.append({
        "role": "assistant",
        "message": resp,
    })

    print("")
