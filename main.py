from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained('models/nous-hermes-llama-2-7b.ggmlv3.q4_K_M.bin', model_type='llama')


ALPACA_FORMAT = """
### Instruction:
You are a helpful, smart assistant that can answer questions about the world. You are also a bit of a
jokester and like to make people laugh.

Previous messages:
{}

### Input:
user: {}

### Response:
assistant: 
"""

def format_messages(messages) -> str:
    ret = ""
    for message in messages:
        ret += f"{message['role']}: {message['message']}\n"
    return ret

messages = [{
    "role": "assistant",
    "message": "How can I help you today?"
}]
while True:
    prompt = input("human: ")

    msgfmt = format_messages(messages)
    new_prompt = ALPACA_FORMAT.format(format_messages(messages), prompt)

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
        "message": prompt,
    })

    messages.append({
        "role": "assistant",
        "message": resp,
    })

    print("")
