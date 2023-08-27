class ModelProfile:
    model_type = None
    model_path = None
    prompt_style = None
    def __init__(self, model_path, model_type="llama", prompt_style="alpaca"):
        self.model_path=model_path
        self.model_type=model_type
        self.prompt_style=prompt_style

NOUS_HERMES_7B = ModelProfile(
    model_path="nous-hermes-llama-2-7b.ggmlv3.q4_K_M.bin",
    model_type="llama",
    prompt_style="alpaca",
)

CODE_LLAMA_7B = ModelProfile(
    model_path="codellama-7b-instruct.Q5_K_M.gguf",
    model_type="llama",
    prompt_style="llama",
)

CODE_LLAMA_13B = ModelProfile(
    model_path="codellama-13b-instruct.Q5_K_M.gguf",
    model_type="llama",
    prompt_style="llama",
)
