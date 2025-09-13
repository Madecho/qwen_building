import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM
from prompts import system_prompt

# model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# model_name = "params/checkpoint-234000"
model_name = "params/checkpoint-1000"

model = Qwen2ForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


q_text = tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Please randomly create a 3D building model, the number of vertices n is 20."}
    ],
    tokenize=False,
    add_generation_prompt=True
)

print(q_text)

q_token_inds = tokenizer([q_text], return_tensors="pt")["input_ids"].to(model.device)
print(q_token_inds)
print(q_token_inds.shape)

text_embds = model.get_input_embeddings()(q_token_inds)
print(text_embds.shape, text_embds.device)

model.eval()
with torch.no_grad():
    output = model.generate(inputs_embeds=text_embds, max_length=4096, use_cache=True, top_k=3, temperature=0.7)
    print(output.shape)

    resp = tokenizer.batch_decode(output)
    print(resp[0])

