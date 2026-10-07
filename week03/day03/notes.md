                Dataset
                   │
      ┌────────────┼────────────┐
      ↓            ↓            ↓
    map()       filter()     select()
      │            │            │
修改/转换       条件过滤       index选择

                   +
                shuffle()
                   │
                打乱顺序

1. map() 用于对 Dataset 中的数据应用处理函数，修改字段或创建新字段。

2. filter() 根据条件决定一条数据是否保留。

3. select() 根据 row index 选择指定数据。

4. shuffle() 打乱 Dataset 的 row 顺序，seed 用于保证随机操作可复现。

5. map/filter/select/shuffle 都可以组成 Dataset processing pipeline，为 Day 4 的 Dataset Tokenization 做准备。