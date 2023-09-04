import prompting
from typing import List, Dict
import ctransformers
import streaming
import model_profiles

def format_messages(messages) -> str:
    ret = ""
    for message in messages:
        ret += f"{message['role']}: {message['message']}\n"
    return ret

GO_ASSISTANT_PROMPT = """
You are a helpful, smart assistant that can answer any question with perfect accuracy.
You are an expert Golang developer.
Respond with solely the answer to the user's question with no decoration text.
""".strip()

GENERIC_ASSISTANT_PROMPT = """
You are a helpful, smart assistant that can answer any question with perfect accuracy.
""".strip()

def get_chat_prompt(model: model_profiles.ModelProfile, messages: List[Dict], user_input: str) -> str:
    system_prompt = f"""
{GENERIC_ASSISTANT_PROMPT}

Previous messages:
{format_messages(messages)}
""".strip()

    return prompting.get_prompt_for_model(
        prompt_style=model.prompt_style,
        system_prompt=system_prompt,
        message=user_input,
    )

def run_chat(llm: ctransformers.LLM, profile: model_profiles.ModelProfile):
    messages = [{
        "role": "assistant",
        "message": "How can I help you today?"
    }]
    while True:
        user_input = input("human: ").strip()

        new_prompt = get_chat_prompt(profile, messages, user_input)

        tokens = llm.tokenize(new_prompt)

        print("ai: ", end="", flush=True)
        resp = streaming.stream_response(llm, llm.generate(
            tokens=tokens,
            temperature=1,
        )).strip()

        messages.append({
            "role": "human",
            "message": user_input,
        })

        messages.append({
            "role": "assistant",
            "message": resp,
        })

        print("")
