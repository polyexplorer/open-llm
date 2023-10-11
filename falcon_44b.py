from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "tiiuae/falcon-40b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
sequences = pipeline(
   """Create question answer pairs from the given corpus:
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
OUTPUT Q/A:
""",
    max_length=300,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
