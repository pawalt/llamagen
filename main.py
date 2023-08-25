from ctransformers import AutoModelForCausalLM
from typing import List, Dict
import prompting
from chat import run_chat
import refactor
import streaming

NOUS_HERMES_7B = "nous-hermes-llama-2-7b.ggmlv3.q4_K_M.bin"
CODE_LLAMA_7B = "codellama-7b-instruct.Q5_K_M.gguf"
CODE_LLAMA_13B = "codellama-13b-instruct.Q5_K_M.gguf"

active_model = CODE_LLAMA_13B

llm = AutoModelForCausalLM.from_pretrained(f"models/{active_model}", model_type='llama', gpu_layers=1)

refactor_prompt = llm.tokenize(refactor.get_refactor_prompt())

streaming.stream_response(llm, llm.generate(
    tokens=refactor_prompt,
    temperature=1,
))