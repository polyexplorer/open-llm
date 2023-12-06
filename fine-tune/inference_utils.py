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