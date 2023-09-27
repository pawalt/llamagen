from ctransformers import AutoModelForCausalLM
from typing import List, Dict
import prompting
from chat import run_chat
import streaming
import model_profiles

active_model = model_profiles.NOUS_HERMES_7B

llm = AutoModelForCausalLM.from_pretrained(
    f"models/{active_model.model_path}",
    model_type=active_model.model_type,
    gpu_layers=1,
    context_length=2048,
)

run_chat(llm, active_model)

# refactor_prompt = refactor.build_full_prompt(active_model)
# print(refactor_prompt)
# 
# tokens = llm.tokenize(refactor_prompt)
# print(len(tokens))
# 
# streaming.stream_response(llm, llm.generate(
#     tokens=tokens,
#     temperature=.7,
# ))
