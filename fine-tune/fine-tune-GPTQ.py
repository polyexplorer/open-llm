# Misc Imports
import json
import argparse
import os

# LLM Imports
from transformers import GPTQConfig, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, pipeline,AutoTokenizer
from peft import prepare_model_for_kbit_training,LoraConfig, get_peft_model
from datasets import load_dataset


print("Current Directory:",os.getcwd())

current_file_path = os.path.abspath(__file__)

# Define a function to print the number of trainable parameters in the model
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    # Iterate through model parameters
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    # Print the number of trainable parameters, total parameters, and the percentage of trainable parameters
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# A wrapper function whcih will get completion by the model from a user's query
def get_completion(query: str, model, tokenizer) -> str:
  device = "cuda:0"

  prompt_template = """
  Below is an instruction that describes a task. Write a response that appropriately completes the request.
  ### Question:
  {query}

  ### Answer:
  """
  prompt = prompt_template.format(query=query)

  encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

  model_inputs = encodeds.to(device)


  generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
  decoded = tokenizer.batch_decode(generated_ids)
  return (decoded[0])



# Define a function to generate a prompt text based on a data point
def generate_prompt(data_point):
    """
    Generate input text based on a prompt, task instruction, (context info.), and answer

    :param data_point: dict: Data point
    :return: dict: tokenized prompt
    """
    # Check if the data point has additional context information
    instruction = "You are a good Question Answering Assistant. Given a Question and a relevant context, answer the question as truthfully as possible. If you don't know the answer, simply say 'I don't know'. Don't try to come up with an answer."
    if data_point['input']:
        # Create a text with instruction, input, and response
        text = 'Below is an instruction that describes a task, paired with an input that provides' \
               ' further context. Write a response that appropriately completes the request.\n\n'
        text += f'### Instruction:\n{instruction}\n\n'
        text += f'### Input:\n{data_point["input"]}\n\n'
        text += f'### Response:\n{data_point["output"]}'

    # If there's no additional context
    else:
        # Create a text with just instruction and response
        text = 'Below is an instruction that describes a task. Write a response that ' \
               'appropriately completes the request.\n\n'
        text += f'### Instruction:\n{instruction}\n\n'
        text += f'### Response:\n{data_point["output"]}'
    return text










def run_training(model,tokenizer,training_data, output_dir):
    
    # Split the Data in Train/Validation Set
    train_data = training_data["train"]
    test_data = training_data["test"]

    # Enable gradient checkpointing for the model
    model.gradient_checkpointing_enable()

    # Prepare the model for k-bit training using the "prepare_model_for_kbit_training" function
    model = prepare_model_for_kbit_training(model)
    print("Params Before LoRA")
    print_trainable_parameters(model)

    # Define a configuration for the LoRA (Learnable Requantization Activation) method
    lora_config = LoraConfig(
        r=8,                                   # Number of quantization levels
        lora_alpha=32,                         # Hyperparameter for LoRA
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Modules to apply LoRA to
        lora_dropout=0.05,                     # Dropout probability
        bias="none",                           # Type of bias
        task_type="CAUSAL_LM"                  # Task type (in this case, Causal Language Modeling)
    )

    # Get a model with LoRA applied to it using the defined configuration
    peft_model = get_peft_model(model, lora_config)

    # Print the number of trainable parameters in the model after applying LoRA
    print("Params After LoRA")
    print_trainable_parameters(peft_model)

    # Create a trainer for fine-tuning a model
    trainer = Trainer(
        model=peft_model,  # The model to be trained
        train_dataset=train_data,  # Training dataset
        eval_dataset=test_data,    # Evaluation dataset
        args=TrainingArguments(
            per_device_train_batch_size=1,        # Batch size per device during training
            gradient_accumulation_steps=4,        # Number of gradient accumulation steps
            warmup_steps=0.03,                    # Number of warm-up steps for learning rate
            max_steps=10,                        # Maximum number of training steps
            learning_rate=2e-4,                   # Learning rate
            fp16=True,                            # Enable mixed-precision training
            logging_steps=1,                      # Logging frequency during training
            output_dir=output_dir,  # Directory to save output files
            optim="paged_adamw_8bit",             # Optimizer type
            save_strategy="epoch",                # Strategy for saving checkpoints
            push_to_hub=False                      # Push to the Hugging Face model hub
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        # Data collator for language modeling task
    )

    peft_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    # Save the Trained Model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def prepare_resources_for_training(model_id, data_path):
    # Model

    quantization_config_loading = GPTQConfig(bits=4, disable_exllama=True)

    # Load the pre-trained model with the specified quantization configuration
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config_loading, device_map="auto")

    # Load the tokenizer for the same pre-trained model and add an end-of-sequence token
    tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)


    # Data
    
    data = load_dataset("json",data_files=data_path,split='train')
    text_column = [generate_prompt(data_point) for data_point in data]
    data = data.add_column("prompt", text_column)
    data = data.map(lambda samples: tokenizer(samples["prompt"]), batched=True)
    
    return model, tokenizer, data



def load_and_train(model_id, data_path,output_dir):
    
    model, tokenizer, data = prepare_resources_for_training(model_id, data_path)

    run_training(model, tokenizer, data, output_dir)


def run_training_from_arguments(training_args_path):
    with open(training_args_path,'r') as f:
        training_args = json.loads(f.read())
    model_id = training_args['model_id']
    data_path = training_args['data_path']
    output_dir = training_args['output_dir']
    load_and_train(model_id,data_path,output_dir)


def main():
    parser = argparse.ArgumentParser(description="Fine-Tune A HuggingFace GPTQ Model")
    parser.add_argument('training_config',type=str,help='Path the the Training Configuration JSON File.')
    args = parser.parse_args()

    training_args_path = args.training_config
    run_training_from_arguments(training_args_path)


if __name__ == "__main__":
    main()
