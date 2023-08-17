from langchain.llms import OpenLLM

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


import torch

MODEL_NAME = "tiiuae/falcon-7b"


def langchain_llm():
    print("Loading LLM...")
    llm = OpenLLM(model_name=MODEL_NAME, model_id=MODEL_NAME)
    print("Loaded Open Source LLM: ",llm.model_name)
    with torch.cuda.amp.autocast():
        print("In Autocast, Inferencing")
        llm("What is the difference between a duck and a goose? And why there are so many Goose in Canada?")

def qlora_llm():
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, load_in_4bit=True)
    print("Loaded Model", MODEL_NAME)
    
    
    
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









