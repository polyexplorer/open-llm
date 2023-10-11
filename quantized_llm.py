model_name = "TheBloke/falcon-40b-instruct-GPTQ"
model_basename = "model"
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer, pipeline, logging
from langchain.llms import HuggingFacePipeline

def create_model(llm_model_name):
    use_triton = False
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name, use_fast=True)
    return AutoGPTQForCausalLM.from_quantized(
            llm_model_name,
            model_basename=model_basename,
            use_safetensors=True,
            trust_remote_code=True,
            device="cuda:0",
            use_triton=use_triton,
            quantize_config=None), tokenizer


def create_langchain_llm(huggingface_model, tokenizer, generation_config =None):
    if not generation_config:
      generation_config = huggingface_model.generation_config
      generation_config.temperature = 0.1
      generation_config.do_sample = True
      generation_config.num_return_sequences = 1
      generation_config.max_new_tokens = 256
      generation_config.use_cache = False
      generation_config.repetition_penalty = 1
      generation_config.pad_token_id = tokenizer.eos_token_id
      generation_config.eos_token_id = tokenizer.eos_token_id

    pipe = pipeline(
        model=huggingface_model,
        tokenizer=tokenizer,
        return_full_text=True,
        task="text-generation",
        # stopping_criteria=stopping_criteria,
        generation_config=generation_config,
    )
    return HuggingFacePipeline(pipeline=pipe)

model,tokenizer = create_model(model_name)
open_source_llm = create_langchain_llm(model,tokenizer)
print(open_source_llm("""
TASK: Summarize The given input
INPUT:
Ghana  and Côte d’Ivoire  are the world´s  leading  cocoa (Thebroma  cacao) producing  countries;
together they produce 53% of the world’s cocoa. Cocoa contributes 7.5% of the Gross Domestic
Product (GDP) of Côte d'Ivoire and 3.4% of that of Ghana and is an important cash crop for the
rural population in the forest zones of these countries. If progressive climate change affected the
climatic  suitability  for  cocoa  of  West  Africa,  this  would  have  implications  for  the  national
economies and farmer livelihoods, with potential repercussions for forests and natural habitat as
cocoa  growing regions  expand, shrink  or shift.  The objective  of this  paper  is to present future
climate scenarios for the main cocoa growing regions of Ghana and Côte d’Ivoire and to predict
their  impact  on the relative suitability  of these regions for  growing  cocoa.  These  analyses  are
intended to support the respective countries and supply chain actors in developing strategies for
reducing  the  vulnerability  of  the  cocoa  sector  to  climate  change.  Based  on  the  current
distribution  of  cocoa  growing  areas  and  climate  change  predictions from  19 Global Circulation
Models, we predict changes in relative climatic suitability for cocoa for 2050 using the MAXENT
method. According to the model, some  current  cocoa  producing  areas will  become  unsuitable
(Lagunes and Sud‐Comoe in Côte d'Ivoire) requiring crop change, while other areas will require
adaptations  in  agronomic  management,  and  in  yet  others  the  climatic suitability  for  growing
cocoa will increase (Kwahu Plateu in Ghana and southwestern Côte d´Ivoire). We recommend the
development  of  site‐specific  strategies  to  reduce  the  vulnerability  of  cocoa  farmers  and  the
sector to future climate change.
"""))

# Task: Create question answer pairs from the given corpus:
# Example:
# Question: Who are the leadin cocoa producing countries ?
# Answer: Ghana and Côte d’Ivoire.
