import model_profiles

def get_prompt_for_model(prompt_style: str, system_prompt: str, message: str):
    match prompt_style:
        case "alpaca":
            return ALPACA_FORMAT.format(system_prompt, message)
        case "llama":
            return LLAMA_INSTRUCT_FORMAT.format(system_prompt, message)

    raise Exception(f"unrecognized prompt style: {prompt_style}")

ALPACA_FORMAT = """
### Instruction:
{}

### Input:
{}

### Response:
""".strip()

def get_alpaca_prompt(system_prompt: str, message: str) -> str:
    return ALPACA_FORMAT.format(system_prompt, message)

LLAMA_INSTRUCT_FORMAT = """
[INST] <<SYS>>
{}
<</SYS>>

{} [/INST]
""".strip()

def get_llama_prompt(system_prompt: str, message: str) -> str:
    return LLAMA_INSTRUCT_FORMAT.format(system_prompt, message)