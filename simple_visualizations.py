import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
import os
import json

# 设置样式
plt.style.use('ggplot')

# 确保输出目录存在
os.makedirs('visualizations', exist_ok=True)

# 读取GDP数据
df = pd.read_excel('gdp_pcap.xlsx')

# 处理数据 - 将字符串形式的值(如 '25.6k')转换为数值
for col in df.columns:
    if col != 'country':
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('k', '000'), errors='coerce')

# 为可视化准备数据
# 选择关键年份进行分析
key_years = [1900, 1950, 2000, 2020]
key_countries = ['China', 'United States', 'India', 'Germany', 'Japan', 'United Kingdom', 'Brazil', 'Russia']

# 仅保留有完整数据的主要国家
main_countries_df = df[df['country'].isin(key_countries)]

# 函数：将matplotlib图转换为base64图像
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

# 1. 折线图 - 主要国家人均GDP随时间的变化
def create_line_chart():
    plt.figure(figsize=(12, 6))
    for country in key_countries:
        country_data = df[df['country'] == country]
        if not country_data.empty:
            years = range(1990, 2021)
            values = country_data.iloc[0][1990:2021].values
            plt.plot(years, values, label=country)
    
    plt.title('人均GDP趋势 (1990-2020)', fontsize=16)
    plt.ylabel('人均GDP（美元）', fontsize=14)
    plt.xlabel('年份', fontsize=14)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 2. 条形图 - 2020年人均GDP最高的10个国家
def create_bar_chart():
    top_countries_2020 = df.sort_values(by=2020, ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    plt.bar(top_countries_2020['country'], top_countries_2020[2020])
    plt.title('2020年人均GDP最高的10个国家', fontsize=16)
    plt.ylabel('人均GDP（美元）', fontsize=14)
    plt.xlabel('国家', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 3. 热力图 - 关键国家在不同年份的人均GDP对比
def create_heatmap():
    heatmap_data = df[df['country'].isin(key_countries)].set_index('country')[key_years]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heatmap_data.values, cmap='YlGnBu')
    
    # 设置坐标轴
    ax.set_xticks(np.arange(len(key_years)))
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_xticklabels(key_years)
    ax.set_yticklabels(heatmap_data.index)
    
    # 添加数值标签
    for i in range(len(heatmap_data.index)):
        for j in range(len(key_years)):
            text = ax.text(j, i, f"{heatmap_data.iloc[i, j]:.0f}",
                          ha="center", va="center", color="black")
    
    plt.title('主要国家在关键年份的人均GDP', fontsize=16)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 4. 饼图 - 2020年全球GDP份额（按区域）
def create_pie_chart():
    # 定义区域分组
    regions = {
        'North America': ['United States', 'Canada', 'Mexico'],
        'Europe': ['Germany', 'United Kingdom', 'France', 'Italy', 'Spain'],
        'Asia': ['China', 'Japan', 'India', 'South Korea'],
        'South America': ['Brazil', 'Argentina', 'Colombia'],
        'Oceania': ['Australia', 'New Zealand'],
        'Africa': ['South Africa', 'Nigeria', 'Egypt']
    }
    
    # 计算各区域2020年GDP总和
    region_gdp = {}
    for region, countries in regions.items():
        region_df = df[df['country'].isin(countries)]
        if not region_df.empty:
            region_gdp[region] = region_df[2020].sum()
    
    # 创建饼图
    plt.figure(figsize=(10, 10))
    plt.pie(list(region_gdp.values()), labels=list(region_gdp.keys()), autopct='%1.1f%%', startangle=90)
    plt.title('2020年全球GDP份额（按区域）', fontsize=16)
    plt.axis('equal')
    
    return fig_to_base64(plt.gcf())

# 5. 散点图 - 1950年与2020年人均GDP对比
def create_scatter_plot():
    scatter_df = df[['country', 1950, 2020]].dropna()
    
    plt.figure(figsize=(12, 8))
    plt.scatter(scatter_df[1950], scatter_df[2020], alpha=0.7)
    
    # 为主要国家添加标签
    for country in key_countries:
        country_data = scatter_df[scatter_df['country'] == country]
        if not country_data.empty:
            plt.annotate(country, 
                        (country_data[1950].values[0], country_data[2020].values[0]),
                        xytext=(5, 5), textcoords='offset points')
    
    plt.title('1950年与2020年人均GDP对比', fontsize=16)
    plt.xlabel('1950年人均GDP', fontsize=14)
    plt.ylabel('2020年人均GDP', fontsize=14)
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 6. 面积图 - 主要国家GDP随时间变化
def create_area_chart():
    plt.figure(figsize=(12, 6))
    
    years = range(1950, 2021, 10)
    for country in key_countries[:5]:  # 只取前5个国家，避免图表过于拥挤
        country_data = df[df['country'] == country]
        if not country_data.empty:
            values = [country_data[year].values[0] for year in years]
            plt.fill_between(years, values, alpha=0.4, label=country)
    
    plt.title('主要国家人均GDP变化 (1950-2020)', fontsize=16)
    plt.ylabel('人均GDP（美元）', fontsize=14)
    plt.xlabel('年份', fontsize=14)
    plt.legend()
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 7. 堆叠柱状图 - 不同时期主要国家GDP对比
def create_stacked_bar():
    comparison_years = [1950, 1980, 2000, 2020]
    countries = key_countries[:5]  # 只取前5个国家
    
    data = []
    for country in countries:
        country_data = df[df['country'] == country]
        if not country_data.empty:
            values = [country_data[year].values[0] for year in comparison_years]
            data.append(values)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(comparison_years))
    
    for i, d in enumerate(data):
        ax.bar(comparison_years, d, bottom=bottom, label=countries[i])
        bottom += d
    
    ax.set_title('主要国家GDP累计对比', fontsize=16)
    ax.set_ylabel('人均GDP（美元）', fontsize=14)
    ax.set_xlabel('年份', fontsize=14)
    ax.legend()
    
    return fig_to_base64(plt.gcf())

# 8. 箱形图 - 各大洲2020年人均GDP分布
def create_boxplot():
    # 定义洲别映射
    continents = {
        'Asia': ['China', 'Japan', 'India', 'South Korea', 'Indonesia', 'Saudi Arabia', 'Turkey'],
        'Europe': ['Germany', 'United Kingdom', 'France', 'Italy', 'Spain', 'Russia', 'Netherlands'],
        'North America': ['United States', 'Canada', 'Mexico'],
        'South America': ['Brazil', 'Argentina', 'Colombia', 'Chile'],
        'Africa': ['South Africa', 'Nigeria', 'Egypt', 'Algeria', 'Morocco'],
        'Oceania': ['Australia', 'New Zealand']
    }
    
    # 准备数据
    box_data = {}
    for continent, countries in continents.items():
        continent_data = []
        for country in countries:
            country_data = df[df['country'] == country]
            if not country_data.empty and not pd.isna(country_data[2020].values[0]):
                continent_data.append(country_data[2020].values[0])
        if continent_data:
            box_data[continent] = continent_data
    
    plt.figure(figsize=(12, 6))
    plt.boxplot(list(box_data.values()), labels=list(box_data.keys()))
    plt.title('2020年各大洲人均GDP分布', fontsize=16)
    plt.ylabel('人均GDP（美元）', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 9. 历史趋势图 - 全球平均人均GDP变化
def create_trend_chart():
    # 计算每年的全球平均人均GDP
    yearly_avg = {}
    for year in range(1900, 2021):
        if year in df.columns:
            yearly_avg[year] = df[year].mean()
    
    years = list(yearly_avg.keys())
    values = list(yearly_avg.values())
    
    plt.figure(figsize=(12, 6))
    plt.plot(years, values, marker='o', linestyle='-', alpha=0.7)
    plt.title('全球平均人均GDP历史趋势 (1900-2020)', fontsize=16)
    plt.ylabel('平均人均GDP（美元）', fontsize=14)
    plt.xlabel('年份', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 10. 分组柱状图 - 不同时期主要国家GDP增长率
def create_grouped_bar():
    periods = [(1950, 1980), (1980, 2000), (2000, 2020)]
    countries = key_countries[:5]  # 只取前5个国家
    
    growth_data = {}
    for country in countries:
        country_data = df[df['country'] == country]
        if not country_data.empty:
            growth_rates = []
            for start, end in periods:
                start_gdp = country_data[start].values[0]
                end_gdp = country_data[end].values[0]
                # 计算年均复合增长率
                years = end - start
                growth_rate = ((end_gdp / start_gdp) ** (1 / years) - 1) * 100
                growth_rates.append(growth_rate)
            growth_data[country] = growth_rates
    
    # 设置图表
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 设置x轴位置
    x = np.arange(len(periods))
    width = 0.15
    multiplier = 0
    
    # 绘制每个国家的柱状图
    for country, growth_rates in growth_data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, growth_rates, width, label=country)
        multiplier += 1
    
    # 添加标签和图例
    ax.set_title('主要国家在不同时期的GDP年均增长率(%)', fontsize=16)
    ax.set_ylabel('年均增长率(%)', fontsize=14)
    ax.set_xticks(x + width * (len(countries) - 1) / 2)
    ax.set_xticklabels([f'{start}-{end}' for start, end in periods])
    ax.legend(loc='best')
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 生成所有可视化并保存代码和图表数据
visualizations = [
    {
        'id': 'line-chart',
        'title': '折线图 - 主要国家人均GDP趋势 (1990-2020)',
        'description': '展示主要经济体人均GDP随时间的变化趋势',
        'function': create_line_chart,
        'code': create_line_chart.__code__
    },
    {
        'id': 'bar-chart',
        'title': '条形图 - 2020年人均GDP最高的10个国家',
        'description': '比较2020年全球人均GDP最高的10个国家',
        'function': create_bar_chart,
        'code': create_bar_chart.__code__
    },
    {
        'id': 'heatmap',
        'title': '热力图 - 关键国家在不同年份的人均GDP对比',
        'description': '通过热力图比较主要国家在关键历史时间点的人均GDP',
        'function': create_heatmap,
        'code': create_heatmap.__code__
    },
    {
        'id': 'pie-chart',
        'title': '饼图 - 2020年全球GDP份额（按区域）',
        'description': '展示2020年全球GDP在不同地理区域的分布情况',
        'function': create_pie_chart,
        'code': create_pie_chart.__code__
    },
    {
        'id': 'scatter-plot',
        'title': '散点图 - 1950年与2020年人均GDP对比',
        'description': '探索各国1950年与2020年人均GDP之间的关系',
        'function': create_scatter_plot,
        'code': create_scatter_plot.__code__
    },
    {
        'id': 'area-chart',
        'title': '面积图 - 主要国家GDP随时间变化',
        'description': '展示主要国家人均GDP随时间的累积变化',
        'function': create_area_chart,
        'code': create_area_chart.__code__
    },
    {
        'id': 'stacked-bar',
        'title': '堆叠柱状图 - 不同时期主要国家GDP对比',
        'description': '比较不同时期主要国家GDP的累积变化',
        'function': create_stacked_bar,
        'code': create_stacked_bar.__code__
    },
    {
        'id': 'boxplot',
        'title': '箱形图 - 各大洲2020年人均GDP分布',
        'description': '比较各大洲国家2020年人均GDP的分布特征',
        'function': create_boxplot,
        'code': create_boxplot.__code__
    },
    {
        'id': 'trend-chart',
        'title': '历史趋势图 - 全球平均人均GDP变化',
        'description': '展示1900-2020年全球平均人均GDP的历史变化趋势',
        'function': create_trend_chart,
        'code': create_trend_chart.__code__
    },
    {
        'id': 'grouped-bar',
        'title': '分组柱状图 - 不同时期主要国家GDP增长率',
        'description': '比较主要国家在不同历史时期的GDP年均增长率',
        'function': create_grouped_bar,
        'code': create_grouped_bar.__code__
    }
]

# 生成所有可视化
results = []
for viz in visualizations:
    try:
        print(f"生成: {viz['title']}")
        chart_data = viz['function']()
        viz['chart_data'] = chart_data
        results.append(viz)
    except Exception as e:
        print(f"生成 {viz['title']} 时出错: {e}")

# 创建HTML展示页面
html_output = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GDP人均数据可视化展示</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/github.min.css">
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        h1, h2, h3 {{
            color: #343a40;
        }}
        .chart-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            padding: 20px;
            transition: transform 0.3s ease;
        }}
        .chart-container:hover {{
            transform: translateY(-5px);
        }}
        .code-container {{
            background: #f5f5f5;
            border-radius: 4px;
            padding: 15px;
            margin-top: 15px;
            overflow-x: auto;
            display: none;
        }}
        .btn-toggle {{
            margin-top: 10px;
        }}
        pre {{
            white-space: pre-wrap;
            font-family: Consolas, monospace;
            font-size: 14px;
        }}
        .header {{
            background-color: #343a40;
            color: white;
            padding: 30px 0;
            margin-bottom: 40px;
            border-radius: 8px;
        }}
        .viz-description {{
            color: #6c757d;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header text-center">
            <h1>GDP人均数据可视化展示</h1>
            <p class="lead">基于历史GDP人均数据的10种多样化可视化图表</p>
        </div>
        
        <div class="row">
"""

# 添加每个可视化图表
for i, viz in enumerate(results):
    # 开始新的一行
    if i % 2 == 0 and i > 0:
        html_output += """
        </div>
        <div class="row">
        """
    
    # 获取函数源代码
    function_code = viz['function'].__code__
    function_name = viz['function'].__name__
    
    # 从文件中读取函数源代码
    import inspect
    function_source = inspect.getsource(viz['function'])
    
    html_output += f"""
            <div class="col-md-6">
                <div class="chart-container">
                    <h3>{viz['title']}</h3>
                    <p class="viz-description">{viz['description']}</p>
                    
                    <!-- 图表 -->
                    <div class="chart-image">
                        <img src="data:image/png;base64,{viz['chart_data']}" class="img-fluid" alt="{viz['title']}">
                    </div>
                    
                    <!-- 代码显示/隐藏按钮 -->
                    <button class="btn btn-sm btn-primary btn-toggle" onclick="toggleCode('{viz['id']}')">显示代码</button>
                    
                    <!-- 代码容器 -->
                    <div id="code-{viz['id']}" class="code-container">
                        <pre><code class="python">{function_source}</code></pre>
                    </div>
                </div>
            </div>
    """

# 完成HTML页面
html_output += """
        </div>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
    <script>
        function toggleCode(id) {
            const codeContainer = document.getElementById(`code-${id}`);
            const button = event.target;
            
            if (codeContainer.style.display === 'block') {
                codeContainer.style.display = 'none';
                button.textContent = '显示代码';
            } else {
                codeContainer.style.display = 'block';
                button.textContent = '隐藏代码';
                hljs.highlightAll();
            }
        }
        
        // 初始化代码高亮
        document.addEventListener('DOMContentLoaded', (event) => {
            hljs.highlightAll();
        });
    </script>
    
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# 保存HTML文件
with open('gdp_visualizations.html', 'w', encoding='utf-8') as f:
    f.write(html_output)

print("可视化生成完成！HTML页面已保存为 gdp_visualizations.html") 