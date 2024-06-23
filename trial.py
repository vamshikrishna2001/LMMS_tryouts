import torch
import torch.nn as nn
import transformers

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

input_text = "Write a detailed review of the movie Interstellar, including its plot, characters, and overall impact."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids, max_length=512, num_beams=10, repetition_penalty=2.0, early_stopping=True)
print(tokenizer.decode(outputs[0]))
