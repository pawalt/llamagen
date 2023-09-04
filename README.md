installing:

```
// standalone
$ CT_METAL=1 poetry add ctransformers
// for langchain
$ CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 poetry add llama-cpp-python
```