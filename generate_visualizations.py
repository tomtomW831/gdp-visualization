import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import os
from wordcloud import WordCloud
import networkx as nx
import base64
from io import BytesIO

# 设置样式
plt.style.use('ggplot')
sns.set(style="whitegrid")

# 确保输出目录存在
os.makedirs('visualizations', exist_ok=True)

# 读取GDP数据
df = pd.read_excel('gdp_pcap.xlsx')

# 处理数据 - 将字符串形式的值(如 '25.6k')转换为数值
for col in df.columns:
    if col != 'country':
        df[col] = df[col].astype(str).str.replace('k', '000').astype(float)

# 为可视化准备数据
# 选择关键年份进行分析
key_years = [1900, 1950, 2000, 2020, 2050]
key_countries = ['China', 'United States', 'India', 'Germany', 'Japan', 'United Kingdom', 'Brazil', 'Russia']

# 仅保留有完整数据的主要国家
main_countries_df = df[df['country'].isin(key_countries)]

# 函数：将matplotlib/seaborn图转换为base64图像
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

# 函数：将plotly图转换为HTML
def plotly_to_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

# 1. 折线图 - 主要国家人均GDP随时间的变化
def create_line_chart():
    plt.figure(figsize=(12, 6))
    for country in key_countries:
        country_data = df[df['country'] == country]
        if not country_data.empty:
            plt.plot(country_data.iloc[0][1990:2023], label=country)
    
    plt.title('人均GDP趋势 (1990-2022)', fontsize=16)
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
    sns.barplot(x=top_countries_2020['country'], y=top_countries_2020[2020])
    plt.title('2020年人均GDP最高的10个国家', fontsize=16)
    plt.ylabel('人均GDP（美元）', fontsize=14)
    plt.xlabel('国家', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 3. 热力图 - 关键国家在不同年份的人均GDP对比
def create_heatmap():
    heatmap_data = df[df['country'].isin(key_countries)].set_index('country')[key_years]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlGnBu')
    plt.title('主要国家在关键年份的人均GDP', fontsize=16)
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
    
    fig = px.scatter(scatter_df, x=1950, y=2020, text='country',
                     title='1950年与2020年人均GDP对比',
                     labels={'1950': '1950年人均GDP', '2020': '2020年人均GDP'})
    
    fig.update_traces(marker=dict(size=10, opacity=0.7),
                     textposition='top center')
    fig.update_layout(height=600, width=800)
    
    return plotly_to_html(fig)

# 6. 地理图 - 2020年全球人均GDP地图
def create_choropleth():
    # 国家代码映射（需要ISO-3字母代码）
    # 简化版本，实际应用中应使用完整映射
    iso_mapping = {
        'United States': 'USA', 'China': 'CHN', 'Japan': 'JPN', 'Germany': 'DEU',
        'United Kingdom': 'GBR', 'France': 'FRA', 'India': 'IND', 'Italy': 'ITA',
        'Brazil': 'BRA', 'Canada': 'CAN', 'Russia': 'RUS', 'South Korea': 'KOR',
        'Australia': 'AUS', 'Spain': 'ESP', 'Mexico': 'MEX', 'Indonesia': 'IDN'
    }
    
    # 创建包含ISO代码的新数据框
    map_df = df[['country', 2020]].copy()
    map_df['iso_alpha'] = map_df['country'].map(iso_mapping)
    map_df = map_df.dropna()
    
    fig = px.choropleth(map_df, locations='iso_alpha', color=2020,
                       color_continuous_scale=px.colors.sequential.Plasma,
                       title='2020年全球人均GDP地图',
                       labels={2020: '人均GDP'})
    
    fig.update_layout(height=600, width=1000)
    
    return plotly_to_html(fig)

# 7. 雷达图 - 主要经济体在不同时期的GDP增长率对比
def create_radar_chart():
    # 计算主要国家在不同时期的GDP增长率
    periods = [(1950, 1970), (1970, 1990), (1990, 2010), (2010, 2020)]
    growth_data = {}
    
    for country in key_countries:
        country_data = df[df['country'] == country]
        if not country_data.empty:
            growth_rates = []
            for start, end in periods:
                start_gdp = country_data[start].values[0]
                end_gdp = country_data[end].values[0]
                annual_growth = ((end_gdp / start_gdp) ** (1 / (end - start)) - 1) * 100
                growth_rates.append(annual_growth)
            growth_data[country] = growth_rates
    
    # 创建雷达图
    categories = [f'{start}-{end}' for start, end in periods]
    
    fig = go.Figure()
    
    for country, growth_rates in growth_data.items():
        fig.add_trace(go.Scatterpolar(
            r=growth_rates,
            theta=categories,
            fill='toself',
            name=country
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
            )
        ),
        title='主要经济体在不同时期的GDP年均增长率(%)',
        height=600,
        width=800
    )
    
    return plotly_to_html(fig)

# 8. 堆叠面积图 - 不同收入组GDP总量变化
def create_stacked_area():
    # 根据2020年人均GDP对国家进行分组
    df['income_group'] = pd.cut(df[2020], bins=[0, 5000, 15000, 50000, float('inf')],
                               labels=['低收入', '中低收入', '中高收入', '高收入'])
    
    # 选择1950-2020年的数据进行可视化
    decades = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]
    income_data = {}
    
    for group in ['低收入', '中低收入', '中高收入', '高收入']:
        group_df = df[df['income_group'] == group]
        income_data[group] = [group_df[year].sum() for year in decades]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(decades, 
                 income_data['低收入'], 
                 income_data['中低收入'],
                 income_data['中高收入'],
                 income_data['高收入'],
                 labels=['低收入', '中低收入', '中高收入', '高收入'],
                 alpha=0.7)
    
    ax.set_title('1950-2020年不同收入组GDP总量变化', fontsize=16)
    ax.set_xlabel('年份', fontsize=14)
    ax.set_ylabel('GDP总量', fontsize=14)
    ax.legend(loc='upper left')
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 9. 箱形图 - 各大洲2020年人均GDP分布
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
    box_data = []
    for continent, countries in continents.items():
        for country in countries:
            country_data = df[df['country'] == country]
            if not country_data.empty:
                gdp_2020 = country_data[2020].values[0]
                box_data.append({'Continent': continent, 'GDP_2020': gdp_2020})
    
    box_df = pd.DataFrame(box_data)
    
    # 创建箱形图
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Continent', y='GDP_2020', data=box_df)
    plt.title('2020年各大洲人均GDP分布', fontsize=16)
    plt.ylabel('人均GDP（美元）', fontsize=14)
    plt.xlabel('洲别', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 10. 网络图 - 基于GDP相似性的国家关系网络
def create_network_graph():
    # 选择2020年数据，计算国家间GDP相似性
    countries = key_countries
    similarity_matrix = np.zeros((len(countries), len(countries)))
    
    # 计算简单的相似度矩阵（基于GDP差异的倒数）
    country_gdps = {}
    for i, country1 in enumerate(countries):
        country1_data = df[df['country'] == country1]
        if not country1_data.empty:
            country_gdps[country1] = country1_data[2020].values[0]
    
    for i, country1 in enumerate(countries):
        for j, country2 in enumerate(countries):
            if i != j:
                gdp1 = country_gdps.get(country1, 0)
                gdp2 = country_gdps.get(country2, 0)
                # 使用差异的倒数作为相似度，缩放以适应可视化
                similarity = 1 / (1 + abs(gdp1 - gdp2) / 10000)
                similarity_matrix[i, j] = similarity
    
    # 创建网络图
    G = nx.Graph()
    
    # 添加节点
    for country in countries:
        G.add_node(country)
    
    # 添加边（只包含相似度高于阈值的）
    threshold = 0.2
    for i, country1 in enumerate(countries):
        for j, country2 in enumerate(countries):
            if i < j and similarity_matrix[i, j] > threshold:
                G.add_edge(country1, country2, weight=similarity_matrix[i, j])
    
    # 绘制网络图
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    
    # 根据GDP设置节点大小
    node_sizes = [country_gdps.get(country, 0)/500 for country in G.nodes()]
    
    # 绘制网络
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
    
    # 根据相似度设置边的宽度
    edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)
    
    # 添加标签
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
    
    plt.title('基于2020年人均GDP相似性的国家关系网络', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 生成所有可视化并保存代码和图表数据
visualizations = [
    {
        'id': 'line-chart',
        'title': '折线图 - 主要国家人均GDP趋势 (1990-2022)',
        'description': '展示主要经济体人均GDP随时间的变化趋势',
        'function': create_line_chart,
        'code': inspect.getsource(create_line_chart)
    },
    {
        'id': 'bar-chart',
        'title': '条形图 - 2020年人均GDP最高的10个国家',
        'description': '比较2020年全球人均GDP最高的10个国家',
        'function': create_bar_chart,
        'code': inspect.getsource(create_bar_chart)
    },
    {
        'id': 'heatmap',
        'title': '热力图 - 关键国家在不同年份的人均GDP对比',
        'description': '通过热力图比较主要国家在关键历史时间点的人均GDP',
        'function': create_heatmap,
        'code': inspect.getsource(create_heatmap)
    },
    {
        'id': 'pie-chart',
        'title': '饼图 - 2020年全球GDP份额（按区域）',
        'description': '展示2020年全球GDP在不同地理区域的分布情况',
        'function': create_pie_chart,
        'code': inspect.getsource(create_pie_chart)
    },
    {
        'id': 'scatter-plot',
        'title': '散点图 - 1950年与2020年人均GDP对比',
        'description': '探索各国1950年与2020年人均GDP之间的关系',
        'function': create_scatter_plot,
        'code': inspect.getsource(create_scatter_plot)
    },
    {
        'id': 'choropleth',
        'title': '地理图 - 2020年全球人均GDP地图',
        'description': '在世界地图上直观展示2020年各国人均GDP水平',
        'function': create_choropleth,
        'code': inspect.getsource(create_choropleth)
    },
    {
        'id': 'radar-chart',
        'title': '雷达图 - 主要经济体在不同时期的GDP增长率对比',
        'description': '比较主要经济体在不同历史阶段的GDP增长表现',
        'function': create_radar_chart,
        'code': inspect.getsource(create_radar_chart)
    },
    {
        'id': 'stacked-area',
        'title': '堆叠面积图 - 不同收入组GDP总量变化',
        'description': '展示1950-2020年间不同收入组国家的GDP总量变化',
        'function': create_stacked_area,
        'code': inspect.getsource(create_stacked_area)
    },
    {
        'id': 'boxplot',
        'title': '箱形图 - 各大洲2020年人均GDP分布',
        'description': '比较各大洲国家2020年人均GDP的分布特征',
        'function': create_boxplot,
        'code': inspect.getsource(create_boxplot)
    },
    {
        'id': 'network-graph',
        'title': '网络图 - 基于GDP相似性的国家关系网络',
        'description': '基于2020年人均GDP相似性构建国家关系网络',
        'function': create_network_graph,
        'code': inspect.getsource(create_network_graph)
    }
]

# 检查导入了inspect库
import inspect

# 生成所有可视化
results = []
for viz in visualizations:
    try:
        print(f"生成: {viz['title']}")
        if 'plotly' in viz['function'].__name__:
            chart_data = viz['function']()
            viz['is_plotly'] = True
        else:
            chart_data = viz['function']()
            viz['is_plotly'] = False
        
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
    if i % 2 == 0:
        html_output += """
        </div>
        <div class="row">
        """
    
    html_output += f"""
            <div class="col-md-6">
                <div class="chart-container">
                    <h3>{viz['title']}</h3>
                    <p class="viz-description">{viz['description']}</p>
                    
                    <!-- 图表 -->
                    <div class="chart-image">
                        {f'<img src="data:image/png;base64,{viz["chart_data"]}" class="img-fluid" alt="{viz["title"]}">' if not viz['is_plotly'] else viz["chart_data"]}
                    </div>
                    
                    <!-- 代码显示/隐藏按钮 -->
                    <button class="btn btn-sm btn-primary btn-toggle" onclick="toggleCode('{viz['id']}')">显示代码</button>
                    
                    <!-- 代码容器 -->
                    <div id="code-{viz['id']}" class="code-container">
                        <pre><code>{viz['code']}</code></pre>
                    </div>
                </div>
            </div>
    """

# 完成HTML页面
html_output += """
        </div>
    </div>
    
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
            }
        }
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