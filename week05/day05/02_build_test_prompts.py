import json
from pathlib import Path


# 1. 固定客服测试问题
questions = {
    "订单查询": [
        "我的订单什么时候发货？",
        "如何查看订单状态？",
        "下单后可以修改收货地址吗？",
        "如何取消尚未发货的订单？",
        "为什么我的订单显示待处理？",
    ],
    "物流配送": [
        "如何查询物流信息？",
        "快递显示已签收，但我没有收到怎么办？",
        "可以指定快递公司吗？",
        "物流信息三天没有更新怎么办？",
        "是否支持配送到偏远地区？",
    ],
    "退款退货": [
        "如何申请退款？",
        "商品支持七天无理由退货吗？",
        "退款一般多久到账？",
        "退货运费由谁承担？",
        "商品拆封后还能退货吗？",
    ],
    "商品咨询": [
        "如何查看商品的详细参数？",
        "商品有其他颜色可以选择吗？",
        "如何确认商品尺寸是否合适？",
        "商品页面显示缺货怎么办？",
        "如何判断商品是否为正品？",
    ],
    "支付问题": [
        "订单支付失败怎么办？",
        "支持哪些支付方式？",
        "支付成功后订单仍显示未付款怎么办？",
        "可以使用两张银行卡支付吗？",
        "如何申请开发票？",
    ],
    "优惠活动": [
        "优惠券在哪里领取？",
        "为什么我的优惠券无法使用？",
        "优惠券可以叠加使用吗？",
        "商品降价后可以申请补差价吗？",
        "满减活动的使用条件是什么？",
    ],
    "售后服务": [
        "收到损坏的商品怎么办？",
        "商品保修期是多久？",
        "如何申请换货？",
        "商品使用一周后出现故障怎么办？",
        "售后申请提交后多久会处理？",
    ],
    "账户问题": [
        "忘记登录密码怎么办？",
        "如何修改绑定的手机号？",
        "如何注销账户？",
        "为什么我的账户无法登录？",
        "如何修改账户中的个人信息？",
    ],
}


# 2. 转换成固定测试样本
dataset = []

for category, prompts in questions.items():
    for prompt in prompts:
        sample = {
            "id": f"qa_{len(dataset) + 1:03d}",
            "category": category,
            "prompt": prompt,
        }

        dataset.append(sample)


# 3. 检查样本数量与 ID 唯一性
assert len(dataset) == 40

ids = [item["id"] for item in dataset]
assert len(ids) == len(set(ids))


# 4. 固定输出路径
output_dir = Path(__file__).resolve().parent / "data"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "test_prompts.jsonl"


# 5. 保存 JSONL
with open(output_path, "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


# 6. 查看结果
print("Total prompts:", len(dataset))
print("Total categories:", len(questions))
print("First sample:", dataset[0])
print("Last sample:", dataset[-1])
print("Saved to:", output_path)