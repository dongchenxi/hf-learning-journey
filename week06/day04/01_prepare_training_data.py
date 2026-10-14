from transformers import AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

examples = [
    {
        "question": "如何查询订单？",
        "answer": "您可以在个人中心的订单页面查询订单状态。",
    },
    {
        "question": "如何申请退款？",
        "answer": "您可以进入订单详情页面，点击申请退款。",
    },
    {
        "question": "如何修改收货地址？",
        "answer": "您可以在订单发货前进入订单详情页面修改收货地址。",
    },
]


for index, example in enumerate(examples):

    messages = [
        {
            "role": "user",
            "content": example["question"],
        },
        {
            "role": "assistant",
            "content": example["answer"],
        },
    ]

    # 1. 应用 Chat Template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # 2. Tokenization
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    # 3. 创建 labels
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    labels = input_ids.clone()

    # 4. 输出检查
    print(f"\n=== Example {index + 1} ===")

    print("Formatted text:")
    print(text)

    print("input_ids shape:", tuple(input_ids.shape))
    print("attention_mask shape:", tuple(attention_mask.shape))
    print("labels shape:", tuple(labels.shape))

    print(
        "input_ids == labels:",
        (input_ids == labels).all().item(),
    )