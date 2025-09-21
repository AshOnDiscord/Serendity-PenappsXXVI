import vec2text
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel

def get_gtr_embeddings(text_list,
                      encoder: PreTrainedModel,
                      tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    inputs = tokenizer(text_list,
                      return_tensors="pt",
                      max_length=128,
                      truncation=True,
                      padding="max_length")
    
    with torch.no_grad():
        # Use only the encoder part of the T5 model
        model_output = encoder.encoder(input_ids=inputs['input_ids'], 
                                      attention_mask=inputs['attention_mask'])
        hidden_state = model_output.last_hidden_state
        embeddings = vec2text.models.model_utils.mean_pool(hidden_state, inputs['attention_mask'])
    return embeddings

# Load model and tokenizer
encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
corrector = vec2text.load_pretrained_corrector("gtr-base")

# Get embeddings
embeddings = get_gtr_embeddings([
    "Blue",
    "Red",
    "Yellow",
], encoder, tokenizer)

# Invert embeddings
result = str(vec2text.invert_embeddings(
    embeddings=embeddings.mean(dim=0, keepdim=True),
    corrector=corrector,
    num_steps=10,
)[0]).strip()



print(result)