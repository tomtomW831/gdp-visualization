# GDP数据可视化项目

这个项目使用Python创建了一个包含10种不同可视化图表的HTML页面，展示全球人均GDP数据的多种视角。

## 文件说明

- `gdp_pcap.xlsx`: 原始GDP数据文件
- `final_visualizations.py`: 最终版本的可视化脚本
- `gdp_visualizations_final.html`: 最终生成的HTML页面，包含10种可视化图表
- `english_visualizations.py`: 英文版可视化脚本
- `gdp_visualizations_en.html`: 英文版HTML页面
- `simple_visualizations.py`: 简化版可视化脚本
- `gdp_visualizations.html`: 简化版HTML页面

## 可视化图表类型

1. **折线图** - 主要国家人均GDP趋势 (1990-2020)
2. **条形图** - 2020年人均GDP最高的10个国家
3. **热力图** - 关键国家在不同年份的人均GDP对比
4. **饼图** - 2020年全球GDP份额（按区域）
5. **散点图** - 1950年与2020年人均GDP对比
6. **面积图** - 主要国家GDP随时间变化
7. **堆叠柱状图** - 不同时期主要国家GDP对比
8. **箱形图** - 各大洲2020年人均GDP分布
9. **趋势图** - 全球平均人均GDP历史变化
10. **分组柱状图** - 不同时期主要国家GDP增长率

## 使用方法

1. 直接在浏览器中打开`gdp_visualizations_final.html`文件查看可视化结果
2. 点击每个图表下方的"Show Code"按钮可以查看生成该图表的Python代码
3. 如果要修改或生成新的可视化，可以编辑`final_visualizations.py`文件并运行：

```bash
python3 final_visualizations.py
```

## 依赖库

- pandas
- matplotlib
- numpy
- base64
- BytesIO

## 数据来源

数据来源于`gdp_pcap.xlsx`文件，包含从1800年到2100年的全球195个国家/地区的人均GDP数据。 