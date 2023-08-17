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

def local_langchain_llm(model_id):
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_quant_type="nf4",
        bnb_8bit_use_double_quant=True,
    )
    

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
    # stop_tokens = [prompt.split(" ")[:5]]
    # print("Stop Tokens:",stop_tokens)
    # stopping_criteria = StoppingCriteriaList(
    #     [StopGenerationCriteria(stop_tokens, tokenizer, model.device)]
    # )

    generation_config = model.generation_config
    generation_config.temperature = 0
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = 256
    generation_config.use_cache = False
    generation_config.repetition_penalty = 1.7
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    print("Loaded Model", model_id)

    pipe = pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        task="text-generation",
        # stopping_criteria=stopping_criteria,
        generation_config=generation_config,
    )

    return HuggingFacePipeline(pipeline=pipe)
def load_pdf(pdf_path):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
    chunks = text_splitter.split_text(text=text)

    # # embeddings
    store_name = pdf.name[:-4]
    st.write(f'{store_name}')
    # st.write(chunks)

    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            VectorStore = pickle.load(f)
        # st.write('Embeddings Loaded from the Disk')s
    else:
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)

def main():
    local_langchain_llm(MODEL_ID)

if __name__ == "__main__":
    main()









