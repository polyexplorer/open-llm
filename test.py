from langchain.llms import OpenLLM, HuggingFacePipeline

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)
from peft import LoraConfig, get_peft_model


import torch

MODEL_NAME = "Falcon-7B"
MODEL_ID = "tiiuae/falcon-7b"

class StopGenerationCriteria(StoppingCriteria):
    def __init__(
        self, tokens: list[list[str]], tokenizer: AutoTokenizer, device: torch.device
    ):
        stop_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
        self.stop_token_ids = [
            torch.tensor(x, dtype=torch.int8, device=device) for x in stop_token_ids
        ]
 
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                return True
        return False



def langchain_llm():
    print("Loading LLM...")
    llm = OpenLLM(model_name=MODEL_NAME, model_id=MODEL_ID)
    print("Loaded Open Source LLM: ",llm.model_name)
    with torch.cuda.amp.autocast():
        print("In Autocast, Inferencing")
        llm("What is the difference between a duck and a goose? And why there are so many Goose in Canada?")

def qlora_llm():
    prompt = "How many months are there in an year?"
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_quant_type="nf4",
        bnb_8bit_use_double_quant=True,
    )
    

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,trust_remote_code=True)
    stop_tokens = [prompt.split(" ")[:5]]
    print("Stop Tokens:",stop_tokens)
    stopping_criteria = StoppingCriteriaList(
        [StopGenerationCriteria(stop_tokens, tokenizer, model.device)]
    )

    generation_config = model.generation_config
    generation_config.temperature = 0
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = 256
    generation_config.use_cache = False
    generation_config.repetition_penalty = 1.7
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    print("Loaded Model", MODEL_NAME)
    pipe = pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        task="text-generation",
        stopping_criteria=stopping_criteria,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    print("Connected With Langchain!")
    result = local_llm(prompt)
    print(result)
    
    # print("Trainable Params:")
    # model.print_trainable_parameters()
    # lora_config = LoraConfig(
    #     r=8,
    #     lora_alpha=32,
    #     target_modules=[
    #         "query_key_value",
    #         "dense",
    #         "dense_h_to_4h",
    #         "dense_4h_to_h",
    #     ],
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM"
    # )
    # model = get_peft_model(model, lora_config)
    # print("Trainable Params (After PEFT):")
    # model.print_trainable_parameters()

def main():
    qlora_llm()

if __name__ == "__main__":
    main()









