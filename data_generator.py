import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random

class DataGenerator:
    """Generates realistic mock data for supply chain tables"""
    
    def __init__(self, seed=42):
        """Initialize data generator with seed for reproducibility"""
        self.fake = Faker()
        Faker.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_customers(self, num_customers=50):
        """Generate customers data"""
        regions = ['North', 'South', 'East', 'West', 'Central']
        customer_types = ['Retail', 'Wholesale', 'Enterprise', 'SMB']
        shipping_modes = ['Standard', 'Express', 'Overnight', 'Ground']
        
        customers = []
        for i in range(num_customers):
            customers.append({
                'Customer_ID': f'CUST_{str(i+1).zfill(4)}',
                'Region': random.choice(regions),
                'Customer_Type': random.choice(customer_types),
                'Preferred_Shipping_Mode': random.choice(shipping_modes)
            })
        
        return pd.DataFrame(customers)
    
    def generate_products(self, num_products=100):
        """Generate products data"""
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Health', 'Automotive']
        
        products = []
        for i in range(num_products):
            unit_cost = round(random.uniform(10, 500), 2)
            unit_price = round(unit_cost * random.uniform(1.2, 3.0), 2)
            
            products.append({
                'Product_ID': f'PROD_{str(i+1).zfill(4)}',
                'Product_Name': f'{random.choice(categories)} Item {i+1}',
                'Category': random.choice(categories),
                'Unit_Price': unit_price,
                'Unit_Cost': unit_cost,
                'Shelf_Life_Days': str(random.choice([30, 60, 90, 180, 365, 'N/A']))
            })
        
        return pd.DataFrame(products)
    
    def generate_suppliers(self, products_df, suppliers_per_product=2):
        """Generate suppliers data"""
        suppliers = []
        supplier_counter = 1
        
        for _, product in products_df.iterrows():
            for j in range(random.randint(1, suppliers_per_product)):
                suppliers.append({
                    'Supplier_ID': f'SUPP_{str(supplier_counter).zfill(4)}',
                    'Product_ID': product['Product_ID'],
                    'Lead_Time_Days': random.randint(1, 30),
                    'Supply_Cost': round(product['Unit_Cost'] * random.uniform(0.8, 1.2), 2),
                    'Reliability_Score': round(random.uniform(0.7, 1.0), 2)
                })
                supplier_counter += 1
        
        return pd.DataFrame(suppliers)
    
    def generate_inventory(self, products_df):
        """Generate inventory data"""
        warehouses = ['WH001', 'WH002', 'WH003', 'WH004', 'WH005']
        
        inventory = []
        for _, product in products_df.iterrows():
            warehouse = random.choice(warehouses)
            reorder_point = random.randint(50, 500)
            safety_stock = int(reorder_point * 0.2)
            current_stock = random.randint(0, reorder_point * 2)
            
            inventory.append({
                'Product_ID': product['Product_ID'],
                'Current_Stock': current_stock,
                'Reorder_Point': reorder_point,
                'Safety_Stock': safety_stock,
                'Warehouse_ID': warehouse
            })
        
        return pd.DataFrame(inventory)
    
    def generate_orders(self, products_df, customers_df, num_orders=500):
        """Generate orders data"""
        order_statuses = ['Pending', 'Processing', 'Shipped', 'Delivered', 'Cancelled']
        
        orders = []
        start_date = datetime.now() - timedelta(days=365)
        
        for i in range(num_orders):
            order_date = start_date + timedelta(days=random.randint(0, 365))
            
            orders.append({
                'Order_ID': f'ORD_{str(i+1).zfill(6)}',
                'Product_ID': random.choice(products_df['Product_ID'].tolist()),
                'Order_Date': order_date.strftime('%Y-%m-%d'),
                'Quantity': random.randint(1, 100),
                'Customer_ID': random.choice(customers_df['Customer_ID'].tolist()),
                'Order_Status': random.choice(order_statuses)
            })
        
        return pd.DataFrame(orders)
    
    def generate_shipments(self, orders_df):
        """Generate shipments data"""
        shipping_modes = ['Standard', 'Express', 'Overnight', 'Ground']
        carriers = ['FedEx', 'UPS', 'DHL', 'USPS', 'Local Courier']
        
        shipments = []
        shipment_counter = 1
        
        # Only create shipments for shipped/delivered orders
        shipped_orders = orders_df[orders_df['Order_Status'].isin(['Shipped', 'Delivered'])]
        
        for _, order in shipped_orders.iterrows():
            dispatch_date = datetime.strptime(order['Order_Date'], '%Y-%m-%d') + timedelta(days=random.randint(1, 5))
            delivery_date = dispatch_date + timedelta(days=random.randint(1, 10))
            
            shipments.append({
                'Shipment_ID': f'SHIP_{str(shipment_counter).zfill(6)}',
                'Order_ID': order['Order_ID'],
                'Dispatch_Date': dispatch_date.strftime('%Y-%m-%d'),
                'Delivery_Date': delivery_date.strftime('%Y-%m-%d') if order['Order_Status'] == 'Delivered' else None,
                'Shipping_Mode': random.choice(shipping_modes),
                'Shipping_Cost': round(random.uniform(5, 50), 2),
                'Carrier': random.choice(carriers)
            })
            shipment_counter += 1
        
        return pd.DataFrame(shipments)
    
    def generate_demand_history(self, products_df, days_back=365):
        """Generate demand history data"""
        demand_history = []
        start_date = datetime.now() - timedelta(days=days_back)
        
        for _, product in products_df.iterrows():
            # Generate demand for each day
            for day in range(days_back):
                date = start_date + timedelta(days=day)
                
                # Create seasonal patterns and trends
                base_demand = random.randint(10, 100)
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day / 365)  # Yearly seasonality
                weekly_factor = 1 + 0.1 * np.sin(2 * np.pi * day / 7)     # Weekly seasonality
                noise = random.uniform(0.8, 1.2)
                
                demand = int(base_demand * seasonal_factor * weekly_factor * noise)
                
                # Skip some days randomly to create realistic gaps
                if random.random() > 0.1:  # 90% chance of having demand data
                    demand_history.append({
                        'Product_ID': product['Product_ID'],
                        'Date': date.strftime('%Y-%m-%d'),
                        'Demand_Units': max(0, demand)
                    })
        
        return pd.DataFrame(demand_history)
