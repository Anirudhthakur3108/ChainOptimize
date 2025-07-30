import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from data_generator import DataGenerator
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database operations and mock data"""
    
    def __init__(self):
        """Initialize database manager with mock data"""
        self.data_generator = DataGenerator()
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize all mock data tables"""
        try:
            self.customers = self.data_generator.generate_customers()
            self.products = self.data_generator.generate_products()
            self.suppliers = self.data_generator.generate_suppliers(self.products)
            self.inventory = self.data_generator.generate_inventory(self.products)
            self.orders = self.data_generator.generate_orders(self.products, self.customers)
            self.shipments = self.data_generator.generate_shipments(self.orders)
            self.demand_history = self.data_generator.generate_demand_history(self.products)
            
            logger.info("Database initialized with mock data")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def get_customers(self):
        """Get customers data"""
        return self.customers.copy()
    
    def get_products(self):
        """Get products data"""
        return self.products.copy()
    
    def get_suppliers(self):
        """Get suppliers data"""
        return self.suppliers.copy()
    
    def get_inventory(self):
        """Get inventory data"""
        return self.inventory.copy()
    
    def get_orders(self):
        """Get orders data"""
        return self.orders.copy()
    
    def get_shipments(self):
        """Get shipments data"""
        return self.shipments.copy()
    
    def get_demand_history(self):
        """Get demand history data"""
        return self.demand_history.copy()
    
    def get_low_stock_items(self, threshold_percentage=20):
        """Get items with low stock levels"""
        inventory = self.get_inventory()
        inventory['stock_percentage'] = (inventory['Current_Stock'] / inventory['Reorder_Point']) * 100
        return inventory[inventory['stock_percentage'] <= threshold_percentage]
    
    def get_orders_by_status(self, status=None):
        """Get orders filtered by status"""
        orders = self.get_orders()
        if status:
            return orders[orders['Order_Status'] == status]
        return orders
    
    def get_supplier_performance(self):
        """Calculate supplier performance metrics"""
        suppliers = self.get_suppliers()
        shipments = self.get_shipments()
        orders = self.get_orders()
        
        # Merge data for analysis
        performance_data = suppliers.merge(
            orders[['Order_ID', 'Product_ID']], on='Product_ID', how='left'
        ).merge(
            shipments[['Order_ID', 'Shipping_Cost', 'Dispatch_Date', 'Delivery_Date']], 
            on='Order_ID', how='left'
        )
        
        # Calculate performance metrics
        performance_summary = performance_data.groupby('Supplier_ID').agg({
            'Reliability_Score': 'mean',
            'Lead_Time_Days': 'mean',
            'Supply_Cost': 'mean',
            'Shipping_Cost': 'mean'
        }).reset_index()
        
        return performance_summary
    
    def get_demand_forecast_data(self, product_id=None):
        """Get demand history data for forecasting"""
        demand = self.get_demand_history()
        if product_id:
            return demand[demand['Product_ID'] == product_id]
        return demand
    
    def update_inventory(self, product_id, new_stock):
        """Update inventory levels"""
        try:
            self.inventory.loc[self.inventory['Product_ID'] == product_id, 'Current_Stock'] = new_stock
            logger.info(f"Updated inventory for product {product_id} to {new_stock}")
            return True
        except Exception as e:
            logger.error(f"Error updating inventory: {str(e)}")
            return False
    
    def get_kpi_data(self):
        """Calculate key performance indicators"""
        try:
            orders = self.get_orders()
            inventory = self.get_inventory()
            products = self.get_products()
            shipments = self.get_shipments()
            
            # Calculate KPIs
            total_orders = len(orders)
            total_revenue = orders.merge(products, on='Product_ID')['Unit_Price'].sum()
            low_stock_items = len(self.get_low_stock_items())
            avg_delivery_time = shipments['Delivery_Date'].apply(
                lambda x: (datetime.strptime(x, '%Y-%m-%d') - datetime.strptime(
                    shipments.loc[shipments['Delivery_Date'] == x, 'Dispatch_Date'].iloc[0], '%Y-%m-%d'
                )).days if pd.notna(x) else 0
            ).mean()
            
            return {
                'total_orders': total_orders,
                'total_revenue': total_revenue,
                'low_stock_items': low_stock_items,
                'avg_delivery_time': avg_delivery_time,
                'total_products': len(products),
                'total_suppliers': len(self.get_suppliers()),
                'total_customers': len(self.get_customers())
            }
        except Exception as e:
            logger.error(f"Error calculating KPIs: {str(e)}")
            return {}
