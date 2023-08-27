from ctransformers import AutoModelForCausalLM
from typing import List, Dict
import prompting
from chat import run_chat
import refactor
import streaming
import model_profiles

active_model = model_profiles.NOUS_HERMES_7B

llm = AutoModelForCausalLM.from_pretrained(
    f"models/{active_model.model_path}",
    model_type=active_model.model_type,
    gpu_layers=1,
)

run_chat(llm, active_model)

# refactor_prompt = llm.tokenize(refactor.get_refactor_prompt())
# 
# streaming.stream_response(llm, llm.generate(
#     tokens=refactor_prompt,
#     temperature=1,
# ))