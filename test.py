import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the Flan-T5 model and tokenizer
model_name = 'google/flan-t5-small'  # You can choose the specific Flan-T5 model variant
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Prepare the model for export
model.eval()

# Dummy input for tracing
input_text = "Translate English to French: The house is wonderful."
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])

# Export the model to ONNX
onnx_file_path = "flan-t5.onnx"
torch.onnx.export(
    model,
    (input_ids, attention_mask, decoder_input_ids),
    onnx_file_path,
    input_names=["input_ids", "attention_mask", "decoder_input_ids"],
    output_names=["logits"],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'decoder_input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size', 1: 'sequence_length'}
    },
    opset_version=11
)