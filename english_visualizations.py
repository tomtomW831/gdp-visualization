import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
import os
import json
import inspect

# Set style
plt.style.use('ggplot')

# Ensure output directory exists
os.makedirs('visualizations', exist_ok=True)

# Read GDP data
df = pd.read_excel('gdp_pcap.xlsx')

# Process data - convert string values (like '25.6k') to numeric
for col in df.columns:
    if col != 'country':
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('k', '000'), errors='coerce')

# Prepare data for visualization
# Select key years for analysis
key_years = [1900, 1950, 2000, 2020]
key_countries = ['China', 'United States', 'India', 'Germany', 'Japan', 'United Kingdom', 'Brazil', 'Russia']

# Function: Convert matplotlib figure to base64 image
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

# 1. Line Chart - GDP per capita trends for major countries
def create_line_chart():
    plt.figure(figsize=(12, 6))
    for country in key_countries:
        country_data = df[df['country'] == country]
        if not country_data.empty:
            years = range(1990, 2021)
            values = country_data.iloc[0][1990:2021].values
            plt.plot(years, values, label=country)
    
    plt.title('GDP per Capita Trends (1990-2020)', fontsize=16)
    plt.ylabel('GDP per Capita (USD)', fontsize=14)
    plt.xlabel('Year', fontsize=14)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 2. Bar Chart - Top 10 countries by GDP per capita in 2020
def create_bar_chart():
    top_countries_2020 = df.sort_values(by=2020, ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    plt.bar(top_countries_2020['country'], top_countries_2020[2020])
    plt.title('Top 10 Countries by GDP per Capita in 2020', fontsize=16)
    plt.ylabel('GDP per Capita (USD)', fontsize=14)
    plt.xlabel('Country', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 3. Heatmap - GDP per capita comparison for key countries across different years
def create_heatmap():
    heatmap_data = df[df['country'].isin(key_countries)].set_index('country')[key_years]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heatmap_data.values, cmap='YlGnBu')
    
    # Set axis labels
    ax.set_xticks(np.arange(len(key_years)))
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_xticklabels(key_years)
    ax.set_yticklabels(heatmap_data.index)
    
    # Add value labels
    for i in range(len(heatmap_data.index)):
        for j in range(len(key_years)):
            text = ax.text(j, i, f"{heatmap_data.iloc[i, j]:.0f}",
                          ha="center", va="center", color="black")
    
    plt.title('GDP per Capita of Major Countries in Key Years', fontsize=16)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 4. Pie Chart - Global GDP share by region in 2020
def create_pie_chart():
    # Define region groupings
    regions = {
        'North America': ['United States', 'Canada', 'Mexico'],
        'Europe': ['Germany', 'United Kingdom', 'France', 'Italy', 'Spain'],
        'Asia': ['China', 'Japan', 'India', 'South Korea'],
        'South America': ['Brazil', 'Argentina', 'Colombia'],
        'Oceania': ['Australia', 'New Zealand'],
        'Africa': ['South Africa', 'Nigeria', 'Egypt']
    }
    
    # Calculate total GDP for each region in 2020
    region_gdp = {}
    for region, countries in regions.items():
        region_df = df[df['country'].isin(countries)]
        if not region_df.empty:
            region_gdp[region] = region_df[2020].sum()
    
    # Create pie chart
    plt.figure(figsize=(10, 10))
    plt.pie(list(region_gdp.values()), labels=list(region_gdp.keys()), autopct='%1.1f%%', startangle=90)
    plt.title('Global GDP Share by Region in 2020', fontsize=16)
    plt.axis('equal')
    
    return fig_to_base64(plt.gcf())

# 5. Scatter Plot - Comparison of GDP per capita between 1950 and 2020
def create_scatter_plot():
    scatter_df = df[['country', 1950, 2020]].dropna()
    
    plt.figure(figsize=(12, 8))
    plt.scatter(scatter_df[1950], scatter_df[2020], alpha=0.7)
    
    # Add labels for key countries
    for country in key_countries:
        country_data = scatter_df[scatter_df['country'] == country]
        if not country_data.empty:
            plt.annotate(country, 
                        (country_data[1950].values[0], country_data[2020].values[0]),
                        xytext=(5, 5), textcoords='offset points')
    
    plt.title('GDP per Capita: 1950 vs 2020', fontsize=16)
    plt.xlabel('GDP per Capita in 1950 (USD)', fontsize=14)
    plt.ylabel('GDP per Capita in 2020 (USD)', fontsize=14)
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 6. Area Chart - GDP per capita changes over time for major countries
def create_area_chart():
    plt.figure(figsize=(12, 6))
    
    years = range(1950, 2021, 10)
    for country in key_countries[:5]:  # Only use first 5 countries to avoid overcrowding
        country_data = df[df['country'] == country]
        if not country_data.empty:
            values = [country_data[year].values[0] for year in years]
            plt.fill_between(years, values, alpha=0.4, label=country)
    
    plt.title('GDP per Capita Changes for Major Countries (1950-2020)', fontsize=16)
    plt.ylabel('GDP per Capita (USD)', fontsize=14)
    plt.xlabel('Year', fontsize=14)
    plt.legend()
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 7. Stacked Bar Chart - GDP comparison across different periods for major countries
def create_stacked_bar():
    comparison_years = [1950, 1980, 2000, 2020]
    countries = key_countries[:5]  # Only use first 5 countries
    
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
    
    ax.set_title('Cumulative GDP Comparison of Major Countries', fontsize=16)
    ax.set_ylabel('GDP per Capita (USD)', fontsize=14)
    ax.set_xlabel('Year', fontsize=14)
    ax.legend()
    
    return fig_to_base64(plt.gcf())

# 8. Box Plot - GDP per capita distribution by continent in 2020
def create_boxplot():
    # Define continent mappings
    continents = {
        'Asia': ['China', 'Japan', 'India', 'South Korea', 'Indonesia', 'Saudi Arabia', 'Turkey'],
        'Europe': ['Germany', 'United Kingdom', 'France', 'Italy', 'Spain', 'Russia', 'Netherlands'],
        'North America': ['United States', 'Canada', 'Mexico'],
        'South America': ['Brazil', 'Argentina', 'Colombia', 'Chile'],
        'Africa': ['South Africa', 'Nigeria', 'Egypt', 'Algeria', 'Morocco'],
        'Oceania': ['Australia', 'New Zealand']
    }
    
    # Prepare data
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
    plt.title('GDP per Capita Distribution by Continent in 2020', fontsize=16)
    plt.ylabel('GDP per Capita (USD)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 9. Trend Chart - Global average GDP per capita changes over history
def create_trend_chart():
    # Calculate global average GDP per capita for each year
    yearly_avg = {}
    for year in range(1900, 2021):
        if year in df.columns:
            yearly_avg[year] = df[year].mean()
    
    years = list(yearly_avg.keys())
    values = list(yearly_avg.values())
    
    plt.figure(figsize=(12, 6))
    plt.plot(years, values, marker='o', linestyle='-', alpha=0.7)
    plt.title('Global Average GDP per Capita Historical Trend (1900-2020)', fontsize=16)
    plt.ylabel('Average GDP per Capita (USD)', fontsize=14)
    plt.xlabel('Year', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 10. Grouped Bar Chart - GDP growth rates for major countries across different periods
def create_grouped_bar():
    periods = [(1950, 1980), (1980, 2000), (2000, 2020)]
    countries = key_countries[:5]  # Only use first 5 countries
    
    growth_data = {}
    for country in countries:
        country_data = df[df['country'] == country]
        if not country_data.empty:
            growth_rates = []
            for start, end in periods:
                start_gdp = country_data[start].values[0]
                end_gdp = country_data[end].values[0]
                # Calculate compound annual growth rate
                years = end - start
                growth_rate = ((end_gdp / start_gdp) ** (1 / years) - 1) * 100
                growth_rates.append(growth_rate)
            growth_data[country] = growth_rates
    
    # Set up chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set x-axis positions
    x = np.arange(len(periods))
    width = 0.15
    multiplier = 0
    
    # Plot bars for each country
    for country, growth_rates in growth_data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, growth_rates, width, label=country)
        multiplier += 1
    
    # Add labels and legend
    ax.set_title('Annual GDP Growth Rates of Major Countries in Different Periods (%)', fontsize=16)
    ax.set_ylabel('Annual Growth Rate (%)', fontsize=14)
    ax.set_xticks(x + width * (len(countries) - 1) / 2)
    ax.set_xticklabels([f'{start}-{end}' for start, end in periods])
    ax.legend(loc='best')
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# Generate all visualizations and save code and chart data
visualizations = [
    {
        'id': 'line-chart',
        'title': 'Line Chart - GDP per Capita Trends (1990-2020)',
        'description': 'Shows GDP per capita trends for major economies over time',
        'function': create_line_chart
    },
    {
        'id': 'bar-chart',
        'title': 'Bar Chart - Top 10 Countries by GDP per Capita in 2020',
        'description': 'Compares the top 10 countries globally by GDP per capita in 2020',
        'function': create_bar_chart
    },
    {
        'id': 'heatmap',
        'title': 'Heatmap - GDP per Capita Comparison Across Key Years',
        'description': 'Compares major countries\' GDP per capita at key historical points using a heatmap',
        'function': create_heatmap
    },
    {
        'id': 'pie-chart',
        'title': 'Pie Chart - Global GDP Share by Region in 2020',
        'description': 'Shows the distribution of global GDP across different geographic regions in 2020',
        'function': create_pie_chart
    },
    {
        'id': 'scatter-plot',
        'title': 'Scatter Plot - GDP per Capita: 1950 vs 2020',
        'description': 'Explores the relationship between countries\' GDP per capita in 1950 and 2020',
        'function': create_scatter_plot
    },
    {
        'id': 'area-chart',
        'title': 'Area Chart - GDP per Capita Changes Over Time',
        'description': 'Shows the cumulative changes in GDP per capita for major countries over time',
        'function': create_area_chart
    },
    {
        'id': 'stacked-bar',
        'title': 'Stacked Bar Chart - Cumulative GDP Comparison',
        'description': 'Compares the cumulative GDP of major countries across different time periods',
        'function': create_stacked_bar
    },
    {
        'id': 'boxplot',
        'title': 'Box Plot - GDP per Capita Distribution by Continent in 2020',
        'description': 'Compares the distribution characteristics of GDP per capita across continents in 2020',
        'function': create_boxplot
    },
    {
        'id': 'trend-chart',
        'title': 'Trend Chart - Global Average GDP per Capita (1900-2020)',
        'description': 'Shows the historical trend of global average GDP per capita from 1900 to 2020',
        'function': create_trend_chart
    },
    {
        'id': 'grouped-bar',
        'title': 'Grouped Bar Chart - GDP Growth Rates Across Different Periods',
        'description': 'Compares annual GDP growth rates of major countries across different historical periods',
        'function': create_grouped_bar
    }
]

# Generate all visualizations
results = []
for viz in visualizations:
    try:
        print(f"Generating: {viz['title']}")
        chart_data = viz['function']()
        viz['chart_data'] = chart_data
        # Get function source code
        viz['code'] = inspect.getsource(viz['function'])
        results.append(viz)
    except Exception as e:
        print(f"Error generating {viz['title']}: {e}")

# Create HTML display page
html_output = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GDP per Capita Data Visualization</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/github.min.css">
    <style>
        body {{
            font-family: 'Arial', sans-serif;
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
            <h1>GDP per Capita Data Visualization</h1>
            <p class="lead">10 Different Visualization Types Based on Historical GDP per Capita Data</p>
        </div>
        
        <div class="row">
"""

# Add each visualization chart
for i, viz in enumerate(results):
    # Start a new row
    if i % 2 == 0 and i > 0:
        html_output += """
        </div>
        <div class="row">
        """
    
    html_output += f"""
            <div class="col-md-6">
                <div class="chart-container">
                    <h3>{viz['title']}</h3>
                    <p class="viz-description">{viz['description']}</p>
                    
                    <!-- Chart -->
                    <div class="chart-image">
                        <img src="data:image/png;base64,{viz['chart_data']}" class="img-fluid" alt="{viz['title']}">
                    </div>
                    
                    <!-- Code show/hide button -->
                    <button class="btn btn-sm btn-primary btn-toggle" onclick="toggleCode('{viz['id']}')">Show Code</button>
                    
                    <!-- Code container -->
                    <div id="code-{viz['id']}" class="code-container">
                        <pre><code class="python">{viz['code']}</code></pre>
                    </div>
                </div>
            </div>
    """

# Complete HTML page
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
                button.textContent = 'Show Code';
            } else {
                codeContainer.style.display = 'block';
                button.textContent = 'Hide Code';
                hljs.highlightAll();
            }
        }
        
        // Initialize code highlighting
        document.addEventListener('DOMContentLoaded', (event) => {
            hljs.highlightAll();
        });
    </script>
    
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# Save HTML file
with open('gdp_visualizations_en.html', 'w', encoding='utf-8') as f:
    f.write(html_output)

print("Visualization generation complete! HTML page saved as gdp_visualizations_en.html") 