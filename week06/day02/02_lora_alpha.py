def lora_forward(
    base_output,
    lora_output,
    r,
    alpha
):
    scaling = alpha / r

    output = base_output + scaling * lora_output

    return scaling, output


base_output = 10
lora_output = 2

configs = [
    (8, 8),
    (8, 16),
    (8, 32),
    (16, 16),
    (16, 32),
]

for r, alpha in configs:
    scaling, output = lora_forward(
        base_output,
        lora_output,
        r,
        alpha
    )

    print(
        f"r={r:<2} | "
        f"alpha={alpha:<2} | "
        f"scaling={scaling:.2f} | "
        f"output={output:.2f}"
    )