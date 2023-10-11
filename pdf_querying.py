from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer, pipeline, logging, AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from huggingface_hub import snapshot_download
import torch
protocol_pdf_path = "/content/drive/MyDrive/sample_protocol.pdf"

# model_name = "TheBloke/falcon-7b-instruct-GPTQ"
model_name = "huggingface/falcon-40b-gptq"
# local_folder = "workspace/llama-2"
# snapshot_download(repo_id=model_name, local_dir=local_folder, local_dir_use_symlinks=False)
model_basename = "model"


class PDFQueryEngine:
  def __init__(self, llm_model_name,pdf_path):
    torch.cuda.empty_cache()
    self.llm_model_name = llm_model_name
    self.pdf_path = pdf_path

    self.model,self.tokenizer = self.create_model(self.llm_model_name)
    print(f"Model '{self.model_id}'s Attention Heads are ")
    for name, param in model.named_parameters():
      if 'self_attn' in name:
          print(name)

    self.create_langchain_llm(self.model,self.tokenizer)
    self.llm = self.create_langchain_llm(self.model,self.tokenizer)
    self.retriever = self.create_doc_retriever(self.pdf_path)

    self.qa_chain  = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.retriever, return_source_documents=True)


  def create_model(self, llm_model_name):
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

  def create_langchain_llm(self, huggingface_model, tokenizer, generation_config =None):
    if not generation_config:
      generation_config = huggingface_model.generation_config
      generation_config.temperature = 0.1
      generation_config.do_sample = True
      generation_config.num_return_sequences = 1
      generation_config.max_new_tokens = 256
      generation_config.use_cache = False
      generation_config.repetition_penalty = 2.5
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

  def create_doc_retriever(self,pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # # select which embeddings we want to use

    # Huggingfface Instruct

    embeddings = HuggingFaceInstructEmbeddings(
      query_instruction="How long is the medical trial duration? What symptoms are expected to be seen?",
      cache_folder = 'embeddings_cache/'
  )


    # Huggingface mini
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    # expose this index in a retriever interface and return
    return db.as_retriever(search_type="similarity", search_kwargs={"k":3})

  def ask_question(self,question):
    return self.qa_chain({"query": question})['result']
