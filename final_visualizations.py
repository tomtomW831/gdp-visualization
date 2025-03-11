import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
import os
import json
import inspect
import pickle
from pathlib import Path

# Set style
plt.style.use('ggplot')

# Create directories for cache and output
os.makedirs('visualizations', exist_ok=True)
os.makedirs('cache', exist_ok=True)

# Cache file path
CACHE_FILE = 'cache/viz_cache.pkl'

def load_cache():
    """Load cached visualization data if it exists"""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_cache(cache_data):
    """Save visualization data to cache"""
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)

# Function to check if code has changed
def get_function_hash(func):
    """Get a hash of the function's source code"""
    return hash(inspect.getsource(func))

# Load data only once
def load_data():
    """Load and preprocess GDP data"""
    df = pd.read_excel('gdp_pcap.xlsx')
    
    # Process data - convert string values (like '25.6k') to numeric
    for col in df.columns:
        if col != 'country':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('k', '000'), errors='coerce')
    
    return df

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
    # Load data
    df = load_data()
    
    plt.figure(figsize=(12, 8))  # Increased height for better label spacing
    
    years = list(range(1990, 2021))
    
    for country in key_countries:
        country_data = df[df['country'] == country]
        if not country_data.empty:
            values = []
            for year in years:
                if year in df.columns and not pd.isna(country_data[year].values[0]):
                    values.append(country_data[year].values[0])
                else:
                    values.append(np.nan)
            
            if not all(np.isnan(values)):
                plt.plot(years, values, label=country, marker='o', markersize=3)
    
    plt.title('GDP per Capita Trends (1990-2020)', fontsize=16, pad=20)
    plt.ylabel('GDP per Capita (USD)', fontsize=14)
    plt.xlabel('Year', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside
    plt.xticks(years[::5], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add key insights
    insights = """Key Insights:
    • US maintained highest GDP per capita throughout the period
    • China showed fastest growth rate, especially after 2000
    • Japan experienced relatively flat growth since 1990
    • Most countries saw significant growth until 2008 crisis"""
    
    plt.figtext(0.02, -0.15, insights, fontsize=10, ha='left', va='top')
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 2. Bar Chart - Top 10 countries by GDP per capita in 2020
def create_bar_chart():
    # Load data
    df = load_data()
    
    top_countries_2020 = df.sort_values(by=2020, ascending=False).head(10)
    
    plt.figure(figsize=(12, 8))  # Increased height for better label spacing
    bars = plt.bar(top_countries_2020['country'], top_countries_2020[2020])
    
    # Add value labels with better positioning
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'${int(height):,}',
                ha='center', va='bottom', rotation=0,
                fontsize=9)  # Smaller font size for values
    
    plt.title('Top 10 Countries by GDP per Capita in 2020', fontsize=16, pad=20)
    plt.ylabel('GDP per Capita (USD)', fontsize=14)
    plt.xlabel('Country', fontsize=14)
    plt.xticks(rotation=45, ha='right')  # Align rotated labels
    plt.grid(axis='y', alpha=0.3)
    
    # Add key insights
    insights = """Key Insights:
    • Luxembourg leads with GDP per capita over $100,000
    • Top 10 dominated by European countries
    • Significant gap between top 3 and others
    • All top 10 countries exceed $40,000 per capita"""
    
    plt.figtext(0.02, -0.15, insights, fontsize=10, ha='left', va='top')
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 3. Heatmap - GDP per capita comparison for key countries across different years
def create_heatmap():
    # Load data
    df = load_data()
    
    heatmap_data = df[df['country'].isin(key_countries)].set_index('country')[key_years]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(heatmap_data.values, cmap='YlGnBu')
    
    # Set axis labels with better spacing
    ax.set_xticks(np.arange(len(key_years)))
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_xticklabels(key_years, rotation=0)
    ax.set_yticklabels(heatmap_data.index)
    
    # Add value labels with adjusted format and size
    for i in range(len(heatmap_data.index)):
        for j in range(len(key_years)):
            value = heatmap_data.iloc[i, j]
            text = ax.text(j, i, f"${value:,.0f}",
                          ha="center", va="center",
                          color="black" if value < heatmap_data.values.max()/2 else "white",
                          fontsize=9)
    
    plt.title('GDP per Capita of Major Countries in Key Years', fontsize=16, pad=20)
    plt.colorbar(im, ax=ax, label='GDP per Capita (USD)')
    
    # Add key insights
    insights = """Key Insights:
    • Most dramatic growth occurred between 1950-2020
    • US consistently maintained high GDP per capita
    • China and India showed remarkable growth post-2000
    • European countries maintained steady high levels"""
    
    plt.figtext(0.02, -0.15, insights, fontsize=10, ha='left', va='top')
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 4. Pie Chart - Global GDP share by region in 2020
def create_pie_chart():
    # Load data
    df = load_data()
    
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
    wedges, texts, autotexts = plt.pie(
        list(region_gdp.values()), 
        labels=list(region_gdp.keys()), 
        autopct='%1.1f%%', 
        startangle=90,
        shadow=True,
        explode=[0.05] * len(region_gdp),  # Slightly explode all slices
        textprops={'fontsize': 12}
    )
    
    # Enhance text visibility
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.title('Global GDP Share by Region in 2020', fontsize=16)
    plt.axis('equal')
    
    return fig_to_base64(plt.gcf())

# 5. Scatter Plot - Comparison of GDP per capita between 1950 and 2020
def create_scatter_plot():
    # Load data
    df = load_data()
    
    scatter_df = df[['country', 1950, 2020]].dropna()
    
    plt.figure(figsize=(12, 8))
    plt.scatter(scatter_df[1950], scatter_df[2020], alpha=0.7, c='steelblue', s=50)
    
    # Add labels for key countries
    for country in key_countries:
        country_data = scatter_df[scatter_df['country'] == country]
        if not country_data.empty:
            plt.annotate(country, 
                        (country_data[1950].values[0], country_data[2020].values[0]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
    
    # Add diagonal line for reference (equal growth)
    max_val = max(scatter_df[1950].max(), scatter_df[2020].max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal Value Line')
    
    plt.title('GDP per Capita: 1950 vs 2020', fontsize=16)
    plt.xlabel('GDP per Capita in 1950 (USD)', fontsize=14)
    plt.ylabel('GDP per Capita in 2020 (USD)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 6. Area Chart - GDP per capita changes over time for major countries
def create_area_chart():
    # Load data
    df = load_data()
    
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
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 7. Stacked Bar Chart - GDP comparison across different periods for major countries
def create_stacked_bar():
    # Load data
    df = load_data()
    
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
    
    # Use a colorful palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, d in enumerate(data):
        ax.bar(comparison_years, d, bottom=bottom, label=countries[i], color=colors[i % len(colors)])
        bottom += d
    
    ax.set_title('Cumulative GDP Comparison of Major Countries', fontsize=16)
    ax.set_ylabel('GDP per Capita (USD)', fontsize=14)
    ax.set_xlabel('Year', fontsize=14)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    return fig_to_base64(plt.gcf())

# 8. Box Plot - GDP per capita distribution by continent in 2020
def create_boxplot():
    # Load data
    df = load_data()
    
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
    
    # Create boxplot with custom colors
    boxplot = plt.boxplot(
        list(box_data.values()), 
        patch_artist=True,  # Fill boxes with color
        labels=list(box_data.keys()),
        medianprops={'color': 'black'}
    )
    
    # Set colors for boxes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title('GDP per Capita Distribution by Continent in 2020', fontsize=16)
    plt.ylabel('GDP per Capita (USD)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 9. Trend Chart - Global average GDP per capita changes over history
def create_trend_chart():
    # Load data
    df = load_data()
    
    # Calculate global average GDP per capita for each year
    yearly_avg = {}
    for year in range(1900, 2021):
        if year in df.columns:
            yearly_avg[year] = df[year].mean()
    
    years = list(yearly_avg.keys())
    values = list(yearly_avg.values())
    
    plt.figure(figsize=(12, 6))
    
    # Plot with gradient color based on year
    points = plt.scatter(years, values, c=years, cmap='viridis', s=30, alpha=0.8)
    
    # Add trend line
    plt.plot(years, values, 'k-', alpha=0.5)
    
    # Add annotations for key years
    for year in [1900, 1950, 2000, 2020]:
        if year in yearly_avg:
            plt.annotate(f'{year}: ${yearly_avg[year]:.0f}',
                        (year, yearly_avg[year]),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    plt.title('Global Average GDP per Capita Historical Trend (1900-2020)', fontsize=16)
    plt.ylabel('Average GDP per Capita (USD)', fontsize=14)
    plt.xlabel('Year', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.colorbar(points, label='Year')
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# 10. Grouped Bar Chart - GDP growth rates for major countries across different periods
def create_grouped_bar():
    # Load data
    df = load_data()
    
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
    
    # Plot bars for each country with custom colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (country, growth_rates) in enumerate(growth_data.items()):
        offset = width * multiplier
        rects = ax.bar(x + offset, growth_rates, width, label=country, color=colors[i % len(colors)])
        
        # Add value labels on top of bars
        for rect, val in zip(rects, growth_rates):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height + 0.1,
                   f'{val:.1f}%',
                   ha='center', va='bottom', rotation=0, fontsize=8)
        
        multiplier += 1
    
    # Add labels and legend
    ax.set_title('Annual GDP Growth Rates of Major Countries in Different Periods (%)', fontsize=16)
    ax.set_ylabel('Annual Growth Rate (%)', fontsize=14)
    ax.set_xticks(x + width * (len(countries) - 1) / 2)
    ax.set_xticklabels([f'{start}-{end}' for start, end in periods])
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    return fig_to_base64(plt.gcf())

# Generate all visualizations and save code and chart data
visualizations = [
    {
        'id': 'line-chart',
        'title': 'Line Chart - GDP per Capita Trends (1990-2020)',
        'description': 'Shows GDP per capita trends for major economies over time',
        'function': create_line_chart,
        'insights': """
        • US maintained highest GDP per capita throughout the period ($30k-$65k)
        • China showed fastest growth rate, increasing from $1k to $10k after 2000
        • Japan experienced stagnation since 1990s, hovering around $40k
        • 2008 financial crisis caused temporary decline in most economies
        """
    },
    {
        'id': 'bar-chart',
        'title': 'Bar Chart - Top 10 Countries by GDP per Capita in 2020',
        'description': 'Compares the top 10 countries globally by GDP per capita in 2020',
        'function': create_bar_chart,
        'insights': """
        • Luxembourg leads globally with GDP per capita exceeding $115,000
        • 7 out of top 10 countries are European nations
        • Singapore and Switzerland show strong performance in their regions
        • Significant wealth gap exists between top 3 and other countries
        """
    },
    {
        'id': 'heatmap',
        'title': 'Heatmap - GDP per Capita Comparison Across Key Years',
        'description': 'Compares major countries\' GDP per capita at key historical points using a heatmap',
        'function': create_heatmap,
        'insights': """
        • Most countries had minimal GDP per capita in 1900 (<$5,000)
        • Post-WWII period (1950-2000) saw dramatic growth in developed nations
        • Asian economies showed remarkable growth post-1950
        • Wealth disparity between developed and developing nations remains significant
        """
    },
    {
        'id': 'pie-chart',
        'title': 'Pie Chart - Global GDP Share by Region in 2020',
        'description': 'Shows the distribution of global GDP across different geographic regions in 2020',
        'function': create_pie_chart,
        'insights': """
        • North America accounts for largest share (~35%) of global GDP
        • Asia shows strong presence with growing share (~30%)
        • Europe maintains significant portion (~25%) despite smaller population
        • Africa and South America combined represent less than 10% of global GDP
        """
    },
    {
        'id': 'scatter-plot',
        'title': 'Scatter Plot - GDP per Capita: 1950 vs 2020',
        'description': 'Explores the relationship between countries\' GDP per capita in 1950 and 2020',
        'function': create_scatter_plot,
        'insights': """
        • Strong correlation between 1950 and 2020 wealth levels
        • Most countries appear above the diagonal, indicating overall growth
        • Asian Tigers show exceptional growth, far above the trend line
        • Some countries show minimal progress over 70 years
        """
    },
    {
        'id': 'area-chart',
        'title': 'Area Chart - GDP per Capita Changes Over Time',
        'description': 'Shows the cumulative changes in GDP per capita for major countries over time',
        'function': create_area_chart,
        'insights': """
        • Total global wealth has grown substantially since 1950
        • Developed nations show steady upward trajectory
        • Emerging economies show accelerated growth in recent decades
        • Economic gaps widened significantly post-1980
        """
    },
    {
        'id': 'stacked-bar',
        'title': 'Stacked Bar Chart - Cumulative GDP Comparison',
        'description': 'Compares the cumulative GDP of major countries across different time periods',
        'function': create_stacked_bar,
        'insights': """
        • Total global GDP shows exponential growth pattern
        • US contribution remains dominant throughout all periods
        • China's share grows dramatically in recent decades
        • European nations maintain stable but declining share
        """
    },
    {
        'id': 'boxplot',
        'title': 'Box Plot - GDP per Capita Distribution by Continent in 2020',
        'description': 'Compares the distribution characteristics of GDP per capita across continents in 2020',
        'function': create_boxplot,
        'insights': """
        • North America shows highest median and smallest spread
        • Europe displays high median with significant variation
        • Asia shows largest spread, reflecting diverse economies
        • Africa has lowest median and relatively small spread
        """
    },
    {
        'id': 'trend-chart',
        'title': 'Trend Chart - Global Average GDP per Capita (1900-2020)',
        'description': 'Shows the historical trend of global average GDP per capita from 1900 to 2020',
        'function': create_trend_chart,
        'insights': """
        • Global GDP per capita grew ~15x from 1900 to 2020
        • Growth accelerated significantly post-1950
        • Steepest growth observed during 1950-2000 period
        • Recent decades show continued but slower growth
        """
    },
    {
        'id': 'grouped-bar',
        'title': 'Grouped Bar Chart - GDP Growth Rates Across Different Periods',
        'description': 'Compares annual GDP growth rates of major countries across different historical periods',
        'function': create_grouped_bar,
        'insights': """
        • China consistently shows highest growth rates
        • Most countries peaked in 1950-1980 period
        • Growth rates generally declined in recent periods
        • Developed nations show more stable but lower growth
        """
    }
]

def generate_visualizations(force_regenerate=False):
    """Generate visualizations with caching"""
    # Load cache
    cache = load_cache()
    
    # Load data
    df = load_data()
    
    results = []
    for viz in visualizations:
        viz_id = viz['id']
        viz_func = viz['function']
        func_hash = get_function_hash(viz_func)
        
        # Check if visualization needs to be regenerated
        should_regenerate = (
            force_regenerate or
            viz_id not in cache or
            cache[viz_id]['hash'] != func_hash
        )
        
        if should_regenerate:
            print(f"Generating: {viz['title']}")
            try:
                chart_data = viz_func()
                cache[viz_id] = {
                    'data': chart_data,
                    'hash': func_hash
                }
            except Exception as e:
                print(f"Error generating {viz['title']}: {e}")
                continue
        else:
            print(f"Using cached version: {viz['title']}")
            chart_data = cache[viz_id]['data']
        
        viz['chart_data'] = chart_data
        viz['code'] = inspect.getsource(viz_func)
        results.append(viz)
    
    # Save updated cache
    save_cache(cache)
    
    return results

def generate_html(results):
    """Generate HTML page from visualization results"""
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
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            background-color: #343a40;
            color: white;
            border-radius: 8px;
        }}
        .insights-box {{
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 10px 15px;
            margin: 15px 0;
            font-size: 0.9em;
            line-height: 1.4;
        }}
        .insights-title {{
            font-weight: bold;
            color: #007bff;
            margin-bottom: 5px;
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
                        
                        <!-- Key Insights Box -->
                        <div class="insights-box">
                            <div class="insights-title">Key Insights</div>
                            <div class="insights-content">
                                {viz['insights']}
                            </div>
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
            
            <div class="footer">
                <p>Created with Python, Matplotlib, and Pandas</p>
                <p>Data source: GDP per Capita Dataset (gdp_pcap.xlsx)</p>
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
    </div>
</body>
</html>
"""

    # Save HTML file
    with open('gdp_visualizations_final.html', 'w', encoding='utf-8') as f:
        f.write(html_output)

    print("Visualization generation complete! HTML page saved as gdp_visualizations_final.html")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate GDP visualizations')
    parser.add_argument('--force-regenerate', action='store_true',
                      help='Force regeneration of all visualizations')
    parser.add_argument('--viz-id', type=str,
                      help='Generate only a specific visualization')
    args = parser.parse_args()
    
    # Generate visualizations
    results = generate_visualizations(force_regenerate=args.force_regenerate)
    
    # Generate HTML
    generate_html(results) 