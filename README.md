# text-extract-2
Pulling structured data out of freetext using LLMs

# Options 

## 1. vllm guided generation

This is what I have tried so far - implementing my own output parser. Mine has some things I like, such as the ability to prompt a model to update an existing model, based on which fields are absent.

But it would be good to have something a bit more robust and general purpose.

### current vllm limitations
Vllm is great but doesnt yet support function calling. This [reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1ck0jt8/function_calling_using_vllm_and_llama3/) had quite a good summary of current status:
* function calling is on the [Q2 2024 roadmap](https://github.com/vllm-project/vllm/issues/3861) for vllm
* there is some thing called [functionary](https://github.com/MeetKai/functionary) which has adapted vllm and added function calling.
    - they also maintain their [own models](https://github.com/MeetKai/functionary?tab=readme-ov-file#models-available) for this.
    - these are mistral 7b fine tunes, based on the [config](https://huggingface.co/meetkai/functionary-small-v2.4/blob/main/config.json)
    - the company looks like it has gone for metaverse, crypto, digital twins and now gen ai... [meetkai](https://meetkai.com/)


## 2. `python-llama-cpp` guided generation

I think this is quite close to what I have tried already.

### llama-cpp-python json mode
llama-cpp itself doesnt do function calling, but has a json mode which gets you a good chunk of the way to implementing it. This is what instructor uses to patch in function calling. There are some optimisations that make it not horribly slow.


### llama-cpp-python function calling

llama-cpp-python implements some things on top of llama-cpp, for example, chat templates, which as a jinja thing are hard to do in a lightweight way in c++. 

llama-cpp-python also implements function calling, however it only currently supports the [_functionary_](https://huggingface.co/meetkai) models according to the [docs](https://llama-cpp-python.readthedocs.io/en/latest/server/#function-calling).

Someone has had a go at implementing a function calling chat template for llama 3 in a python-llama-cpp fork. See [reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1c7jtwh/function_calling_template_for_llama_3/)


## 3. langchain output parsers

Langchain offers different options for postprocessing llm output. Not sure any of them hit the spot at the moment.

[OpenAI Tools](https://python.langchain.com/docs/modules/model_io/output_parsers/types/openai_tools/)
* requires OpenAI tool calling API, so limited/no use with open models at present.

[JSON](https://python.langchain.com/docs/modules/model_io/output_parsers/types/json/) / [Pydantic](https://python.langchain.com/docs/modules/model_io/output_parsers/types/pydantic/)
* I dont think these do any form of guided generation to enforce the structure of the output, they would just fail when the LLM doesnt manage it. (though you can [retry](https://python.langchain.com/docs/modules/model_io/output_parsers/types/retry/))


## 4. Instructor
[Instructor](https://python.useinstructor.com) looks like it aims to work well with both open and closed models.

* It can integrate nicely with SQLmodel.
* has some neat examples like [citing RAG sources](https://python.useinstructor.com/examples/exact_citations/)
* It feels like learning the instructor way will be the most transferrable.


Works by [patching](https://python.useinstructor.com/concepts/patching/) clients for different LLM libraries. There are a few different core patching methods.


### Instructor Patching Modes

https://github.com/jxnl/instructor/blob/main/instructor/mode.py

```
JSON = "json_mode"
JSON_SCHEMA = "json_schema_mode"
```

Cant find documentation of what the difference is, but would guess that "json_mode" means the LLM api just verifies if the response is syntacticaly valid JSON, whereas "json_schema_mode" would hopefully validate against a schema - which would be preferable I expect.

A few options for open / available weights models
* `ollama` -> `llama-cpp` ([examples](https://python.useinstructor.com/examples/ollama/#ollama), [docs](https://python.useinstructor.com/hub/ollama/)) which is more mainstream and worth looking at.
    - uses `instructor.Mode.JSON`
    * uses ollama via a localhost API interface so can deploy ollama in docker
    * Ollama has an official [docker image](https://hub.docker.com/r/ollama/ollama) identified [in the github readme](https://github.com/ollama/ollama?tab=readme-ov-file#docker).


* `llama-cpp-python` -> `llama-cpp` ([docs](https://python.useinstructor.com/hub/llama-cpp-python/#llama-cpp-python))
    - uses `instructor.Mode.JSON_SCHEMA` 
    * instructor uses `llama-cpp-python` python lib directly
        * I think I want to keep model running stuff in its own container ideally? 
        * Or perhaps this doesnt matter too much if we arent pulling in all of the pytorch stuff though?


# Open weight function calling models

* Meta - Llama 3
    * according to this [reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1ce63ql/comment/l1hjcua/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) should do ok out of the box
    * [GGUF versions](https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf)
    * [_more_ GGUF versions](https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/tree/main)
    * [_even more_ GGUF versions](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-GGUF/tree/main)


* NousResearch - Hermes
    * https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B
    * meant to be a bit better than base model according to [their own benchmarks](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B#benchmarks)
    * has a specific [prompt format](https://github.com/NousResearch/Hermes-Function-Calling?tab=readme-ov-file#prompt-format-for-function-calling)   
    * [OpenHermes-2.5-Mistral-7B-GGUF](https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF) is what they appear to be using in the [Instructor docs example](https://python.useinstructor.com/hub/llama-cpp-python/#patching)

