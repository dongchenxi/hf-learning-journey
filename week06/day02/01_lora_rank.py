def count_lora_params(d_in, d_out, r):
    return r * (d_in + d_out)


d_in = 512
d_out = 512

full_params = d_in * d_out

for r in [4, 8, 16, 32]:
    lora_params = count_lora_params(
        d_in,
        d_out,
        r
    )

    ratio = lora_params / full_params * 100

    print(
        f"r={r:<2} | "
        f"LoRA Params={lora_params:<6} | "
        f"Ratio={ratio:.2f}%"
    )