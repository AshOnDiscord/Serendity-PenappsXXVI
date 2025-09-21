from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import vec2text


model_name = "gpt2"  # or any Hugging Face LLM
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

corrector = vec2text.load_pretrained_corrector("gtr-base")  



def get_hf_embeddings(text_list, model, tokenizer):
    inputs = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.base_model(**inputs, output_hidden_states=True)
        hidden_state = outputs.hidden_states[-1]  # last layer
        # mean pooling
        attention_mask = inputs['attention_mask']
        embeddings = (hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1).unsqueeze(-1)
    return embeddings

texts = [
    "Jack Morris is a PhD student at Cornell Tech in New York City",
    "It was the best of times, it was the worst of times..."
]

embeddings = get_hf_embeddings(texts, model, tokenizer)


inverted_texts = vec2text.invert_embeddings(
    embeddings=embeddings.cuda(),
    corrector=corrector,
    num_steps=20,
    sequence_beam_width=4
)

for text in inverted_texts:
    print(text)
