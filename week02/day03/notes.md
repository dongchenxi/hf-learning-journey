Raw Text
   │
   ▼
Tokenizer
   │
   ├── Tokenization
   │
   ├── Token → ID
   │
   ├── Special Tokens
   │
   ├── Padding
   │
   └── Attention Mask
   │
   ▼
input_ids
attention_mask
   │
   ▼
Model
   │
   ▼
Forward Pass
   │
   ▼
Outputs

1. Tokenizer 是什么？
分词器，将文本变成模型能处理的数字，也可以将数字变为文字
2. tokenization 是什么？
将文本切分成token的过程
3. token 是什么？
文本被切分后的基本单位
4. token_id 是什么？
token在词表中的数字
5. input_ids 是什么？
模型输入的token_ids
6. vocabulary 是什么？
此表，保存token和token ID的分类
7. token 和 token_id 有什么区别？
token是文本单位，token_id对应的数字编号
8. input_ids 和 token_ids 是什么关系？
input_ids是模型输入的token_ids，文本切割后的token ID
9. special token 是什么？
有特殊用途的 Token
10. tokenizer.tokenize() 做什么？
将文本切割成token
11. convert_tokens_to_ids() 做什么？
把 Tokens 转换成 Token IDs。
12. convert_ids_to_tokens() 做什么？
把 Token IDs 转换成 Tokens。
13. decode() 做什么？
将Token IDs转换回可读文本
14. decode() 可以接收什么？
一个 Token ID 或一串 Token IDs
15. convert_ids_to_tokens() 和 decode() 有什么区别？
convert_ids_to_tokens转换为tokens
将Token IDs转换回可读文本
16. skip_special_tokens=True 做什么？
解码时，去掉特殊token
17. 为什么 tokenizer.tokenize(text) 得到的 token
    数量可能和 tokenizer(text)["input_ids"]
    的数量不同？
因为tokenizer(text)["input_ids"]会加入特殊token
18. 为什么 decode 后的字符串不一定与原始字符串
    逐字符完全相同？
decode() 是 Token IDs → 可读文本，不是 Token IDs → 原始字符串的逐字符还原。