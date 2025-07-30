import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import logging

logger = logging.getLogger(__name__)

class Utils:
    """Utility functions for the supply chain platform"""
    
    def __init__(self):
        """Initialize utilities"""
        pass
    
    def dataframe_to_excel(self, df):
        """Convert DataFrame to Excel buffer for download"""
        try:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Report')
            output.seek(0)
            return output.getvalue()
        except Exception as e:
            logger.error(f"Error converting DataFrame to Excel: {str(e)}")
            return None
    
    def format_currency(self, amount):
        """Format amount as currency"""
        try:
            return f"${amount:,.2f}"
        except:
            return "$0.00"
    
    def format_percentage(self, value):
        """Format value as percentage"""
        try:
            return f"{value:.1f}%"
        except:
            return "0.0%"
    
    def calculate_growth_rate(self, current, previous):
        """Calculate growth rate between two values"""
        try:
            if previous == 0:
                return 0
            return ((current - previous) / previous) * 100
        except:
            return 0
    
    def get_date_range_options(self):
        """Get predefined date range options"""
        today = datetime.now().date()
        return {
            "Today": (today, today),
            "Yesterday": (today - timedelta(days=1), today - timedelta(days=1)),
            "Last 7 Days": (today - timedelta(days=7), today),
            "Last 30 Days": (today - timedelta(days=30), today),
            "Last 90 Days": (today - timedelta(days=90), today),
            "This Month": (today.replace(day=1), today),
            "Last Month": (
                (today.replace(day=1) - timedelta(days=1)).replace(day=1),
                today.replace(day=1) - timedelta(days=1)
            ),
            "This Year": (today.replace(month=1, day=1), today),
            "Last Year": (
                today.replace(year=today.year-1, month=1, day=1),
                today.replace(year=today.year-1, month=12, day=31)
            )
        }
    
    def validate_email(self, email):
        """Validate email format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def generate_product_id(self, category, sequence):
        """Generate standardized product ID"""
        category_codes = {
            'Electronics': 'ELEC',
            'Clothing': 'CLTH',
            'Home & Garden': 'HOME',
            'Sports': 'SPRT',
            'Books': 'BOOK',
            'Health': 'HLTH',
            'Automotive': 'AUTO'
        }
        
        code = category_codes.get(category, 'MISC')
        return f"{code}_{str(sequence).zfill(4)}"
    
    def calculate_safety_stock(self, avg_demand, lead_time, service_level=0.95):
        """Calculate safety stock using statistical method"""
        try:
            # Z-score for different service levels
            z_scores = {0.90: 1.28, 0.95: 1.645, 0.99: 2.33}
            z_score = z_scores.get(service_level, 1.645)
            
            # Assume demand variability is 20% of average demand
            demand_std = avg_demand * 0.2
            
            safety_stock = z_score * demand_std * np.sqrt(lead_time)
            return max(0, int(safety_stock))
        except:
            return 0
    
    def calculate_reorder_point(self, avg_demand, lead_time, safety_stock):
        """Calculate reorder point"""
        try:
            return int((avg_demand * lead_time) + safety_stock)
        except:
            return 0
    
    def calculate_economic_order_quantity(self, annual_demand, ordering_cost, holding_cost):
        """Calculate Economic Order Quantity (EOQ)"""
        try:
            eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
            return max(1, int(eoq))
        except:
            return 1
    
    def categorize_abc_analysis(self, df, value_column):
        """Perform ABC analysis on items"""
        try:
            df = df.copy()
            df = df.sort_values(value_column, ascending=False)
            df['cumulative_value'] = df[value_column].cumsum()
            df['cumulative_percentage'] = (df['cumulative_value'] / df[value_column].sum()) * 100
            
            df['ABC_Category'] = 'C'  # Default to C
            df.loc[df['cumulative_percentage'] <= 80, 'ABC_Category'] = 'A'
            df.loc[(df['cumulative_percentage'] > 80) & (df['cumulative_percentage'] <= 95), 'ABC_Category'] = 'B'
            
            return df
        except Exception as e:
            logger.error(f"Error in ABC analysis: {str(e)}")
            return df
    
    def calculate_inventory_turnover(self, cost_of_goods_sold, avg_inventory_value):
        """Calculate inventory turnover ratio"""
        try:
            if avg_inventory_value == 0:
                return 0
            return cost_of_goods_sold / avg_inventory_value
        except:
            return 0
    
    def calculate_days_sales_inventory(self, avg_inventory_value, daily_cost_of_goods_sold):
        """Calculate days sales in inventory"""
        try:
            if daily_cost_of_goods_sold == 0:
                return 0
            return avg_inventory_value / daily_cost_of_goods_sold
        except:
            return 0
    
    def get_business_days_between(self, start_date, end_date):
        """Calculate business days between two dates"""
        try:
            return np.busday_count(start_date, end_date)
        except:
            return 0
    
    def format_large_number(self, number):
        """Format large numbers with appropriate suffixes"""
        try:
            if number >= 1_000_000_000:
                return f"{number/1_000_000_000:.1f}B"
            elif number >= 1_000_000:
                return f"{number/1_000_000:.1f}M"
            elif number >= 1_000:
                return f"{number/1_000:.1f}K"
            else:
                return f"{number:.0f}"
        except:
            return "0"
    
    def create_color_scale(self, values, colorscale='RdYlGn'):
        """Create color scale for visualization"""
        try:
            import plotly.colors as pc
            
            if not values:
                return []
            
            min_val, max_val = min(values), max(values)
            if min_val == max_val:
                return ['rgb(128,128,128)'] * len(values)
            
            normalized = [(v - min_val) / (max_val - min_val) for v in values]
            
            if colorscale == 'RdYlGn':
                colors = ['rgb(255,0,0)', 'rgb(255,255,0)', 'rgb(0,255,0)']
            else:
                colors = ['rgb(0,0,255)', 'rgb(255,255,255)', 'rgb(255,0,0)']
            
            result_colors = []
            for norm_val in normalized:
                if norm_val <= 0.5:
                    # Interpolate between first two colors
                    ratio = norm_val * 2
                    result_colors.append(f'rgb({int(255 * (1-ratio) + 255 * ratio)},{int(255 * ratio)},{int(255 * (1-ratio))})')
                else:
                    # Interpolate between last two colors
                    ratio = (norm_val - 0.5) * 2
                    result_colors.append(f'rgb({int(255 * (1-ratio))},{int(255)},{int(255 * ratio)})')
            
            return result_colors
        except:
            return ['rgb(128,128,128)'] * len(values)
    
    def export_data_summary(self, df, title="Data Summary"):
        """Create a summary of DataFrame for export"""
        try:
            summary = {
                'Title': title,
                'Generated On': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Total Records': len(df),
                'Columns': list(df.columns),
                'Data Types': df.dtypes.to_dict(),
                'Missing Values': df.isnull().sum().to_dict(),
                'Summary Statistics': df.describe().to_dict() if len(df) > 0 else {}
            }
            return summary
        except Exception as e:
            logger.error(f"Error creating data summary: {str(e)}")
            return {'Error': str(e)}
