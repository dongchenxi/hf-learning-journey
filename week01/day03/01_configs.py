from transformers import AutoConfig

models = {
    "bert": "google-bert/bert-base-uncased",
    "gpt2": "openai-community/gpt2",
    "t5": "google-t5/t5-small",
}

for name, model_id in models.items():
    print("=" * 80)
    print("Model:", name)
    print("Model ID:", model_id)

    config = AutoConfig.from_pretrained(model_id)

    print("Architecture:", config.architectures)
    print("Model type:", config.model_type)
    print("Is encoder-decoder:", config.is_encoder_decoder)