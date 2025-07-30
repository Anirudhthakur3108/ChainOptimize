import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import os
from utils import Utils
import logging

logger = logging.getLogger(__name__)

class DashboardManager:
    """Manages all dashboard views and visualizations"""
    
    def __init__(self, db_manager):
        """Initialize dashboard manager with database connection"""
        self.db_manager = db_manager
        self.utils = Utils()
    
    def show_overview_dashboard(self):
        """Display main overview dashboard"""
        st.header("📊 Supply Chain Overview Dashboard")
        
        try:
            # Get KPI data
            kpis = self.db_manager.get_kpi_data()
            
            # Display KPIs
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Orders",
                    f"{kpis.get('total_orders', 0):,}",
                    delta="12 this week"
                )
            
            with col2:
                st.metric(
                    "Total Revenue",
                    f"${kpis.get('total_revenue', 0):,.2f}",
                    delta="5.2%"
                )
            
            with col3:
                st.metric(
                    "Low Stock Items",
                    f"{kpis.get('low_stock_items', 0)}",
                    delta="-2 from yesterday",
                    delta_color="inverse"
                )
            
            with col4:
                st.metric(
                    "Avg Delivery Time",
                    f"{kpis.get('avg_delivery_time', 0):.1f} days",
                    delta="-0.5 days"
                )
            
            st.markdown("---")
            
            # Charts row 1
            col1, col2 = st.columns(2)
            
            with col1:
                self._show_orders_trend_chart()
            
            with col2:
                self._show_inventory_status_chart()
            
            # Charts row 2
            col1, col2 = st.columns(2)
            
            with col1:
                self._show_supplier_performance_chart()
            
            with col2:
                self._show_category_sales_chart()
            
        except Exception as e:
            logger.error(f"Error in overview dashboard: {str(e)}")
            st.error("Error loading dashboard data")
    
    def show_inventory_dashboard(self):
        """Display inventory management dashboard"""
        st.header("📦 Inventory Management")
        
        try:
            inventory = self.db_manager.get_inventory()
            products = self.db_manager.get_products()
            
            # Merge inventory with product details
            inventory_details = inventory.merge(products, on='Product_ID')
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Products", len(inventory))
            
            with col2:
                total_value = (inventory_details['Current_Stock'] * inventory_details['Unit_Cost']).sum()
                st.metric("Total Inventory Value", f"${total_value:,.2f}")
            
            with col3:
                low_stock = len(self.db_manager.get_low_stock_items())
                st.metric("Low Stock Items", low_stock)
            
            with col4:
                avg_stock_ratio = (inventory['Current_Stock'] / inventory['Reorder_Point']).mean()
                st.metric("Avg Stock Ratio", f"{avg_stock_ratio:.1f}x")
            
            st.markdown("---")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_warehouse = st.selectbox(
                    "Filter by Warehouse:",
                    ["All"] + list(inventory['Warehouse_ID'].unique())
                )
            
            with col2:
                selected_category = st.selectbox(
                    "Filter by Category:",
                    ["All"] + list(products['Category'].unique())
                )
            
            with col3:
                stock_filter = st.selectbox(
                    "Stock Status:",
                    ["All", "Low Stock", "Normal Stock", "Overstock"]
                )
            
            # Apply filters
            filtered_data = inventory_details.copy()
            
            if selected_warehouse != "All":
                filtered_data = filtered_data[filtered_data['Warehouse_ID'] == selected_warehouse]
            
            if selected_category != "All":
                filtered_data = filtered_data[filtered_data['Category'] == selected_category]
            
            if stock_filter == "Low Stock":
                filtered_data = filtered_data[filtered_data['Current_Stock'] <= filtered_data['Reorder_Point']]
            elif stock_filter == "Overstock":
                filtered_data = filtered_data[filtered_data['Current_Stock'] > filtered_data['Reorder_Point'] * 2]
            elif stock_filter == "Normal Stock":
                filtered_data = filtered_data[
                    (filtered_data['Current_Stock'] > filtered_data['Reorder_Point']) &
                    (filtered_data['Current_Stock'] <= filtered_data['Reorder_Point'] * 2)
                ]
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Stock levels by category
                category_stock = filtered_data.groupby('Category')['Current_Stock'].sum().reset_index()
                fig = px.bar(
                    category_stock,
                    x='Category',
                    y='Current_Stock',
                    title="Stock Levels by Category"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Stock ratio distribution
                filtered_data['stock_ratio'] = filtered_data['Current_Stock'] / filtered_data['Reorder_Point']
                fig = px.histogram(
                    filtered_data,
                    x='stock_ratio',
                    title="Stock Ratio Distribution",
                    nbins=20
                )
                fig.add_vline(x=1, line_dash="dash", line_color="red", annotation_text="Reorder Point")
                st.plotly_chart(fig, use_container_width=True)
            
            # Inventory table
            st.subheader("Inventory Details")
            
            # Add stock status column
            filtered_data['Stock_Status'] = filtered_data.apply(
                lambda row: 'Low Stock' if row['Current_Stock'] <= row['Reorder_Point']
                else 'Overstock' if row['Current_Stock'] > row['Reorder_Point'] * 2
                else 'Normal', axis=1
            )
            
            # Display table
            display_columns = ['Product_ID', 'Product_Name', 'Category', 'Current_Stock', 
                             'Reorder_Point', 'Safety_Stock', 'Warehouse_ID', 'Stock_Status']
            st.dataframe(filtered_data[display_columns], use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error in inventory dashboard: {str(e)}")
            st.error("Error loading inventory data")
    
    def show_order_analytics(self):
        """Display order analytics dashboard"""
        st.header("📋 Order Analytics")
        
        try:
            orders = self.db_manager.get_orders()
            products = self.db_manager.get_products()
            customers = self.db_manager.get_customers()
            
            # Merge for detailed analysis
            order_details = orders.merge(products, on='Product_ID').merge(customers, on='Customer_ID')
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Orders", len(orders))
            
            with col2:
                total_revenue = (order_details['Quantity'] * order_details['Unit_Price']).sum()
                st.metric("Total Revenue", f"${total_revenue:,.2f}")
            
            with col3:
                avg_order_value = total_revenue / len(orders) if len(orders) > 0 else 0
                st.metric("Avg Order Value", f"${avg_order_value:.2f}")
            
            with col4:
                pending_orders = len(orders[orders['Order_Status'] == 'Pending'])
                st.metric("Pending Orders", pending_orders)
            
            st.markdown("---")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Orders by status
                status_counts = orders['Order_Status'].value_counts()
                fig = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title="Orders by Status"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Orders by customer type
                customer_type_orders = order_details['Customer_Type'].value_counts()
                fig = px.bar(
                    x=customer_type_orders.index,
                    y=customer_type_orders.values,
                    title="Orders by Customer Type"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Time series analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # Orders over time
                order_details['Order_Date'] = pd.to_datetime(order_details['Order_Date'])
                daily_orders = order_details.groupby(order_details['Order_Date'].dt.date).size().reset_index()
                daily_orders.columns = ['Date', 'Order_Count']
                
                fig = px.line(
                    daily_orders,
                    x='Date',
                    y='Order_Count',
                    title="Daily Order Trend"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Revenue over time
                order_details['Revenue'] = order_details['Quantity'] * order_details['Unit_Price']
                daily_revenue = order_details.groupby(order_details['Order_Date'].dt.date)['Revenue'].sum().reset_index()
                daily_revenue.columns = ['Date', 'Revenue']
                
                fig = px.line(
                    daily_revenue,
                    x='Date',
                    y='Revenue',
                    title="Daily Revenue Trend"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Recent orders table
            st.subheader("Recent Orders")
            recent_orders = order_details.sort_values('Order_Date', ascending=False).head(20)
            display_columns = ['Order_ID', 'Product_Name', 'Customer_Type', 'Quantity', 
                             'Unit_Price', 'Order_Date', 'Order_Status']
            st.dataframe(recent_orders[display_columns], use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error in order analytics: {str(e)}")
            st.error("Error loading order data")
    
    def show_supplier_analytics(self):
        """Display supplier analytics dashboard"""
        st.header("🏭 Supplier Analytics")
        
        try:
            suppliers = self.db_manager.get_suppliers()
            products = self.db_manager.get_products()
            
            # Merge supplier data with products
            supplier_details = suppliers.merge(products, on='Product_ID')
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Suppliers", len(suppliers['Supplier_ID'].unique()))
            
            with col2:
                avg_reliability = suppliers['Reliability_Score'].mean()
                st.metric("Avg Reliability", f"{avg_reliability:.2f}")
            
            with col3:
                avg_lead_time = suppliers['Lead_Time_Days'].mean()
                st.metric("Avg Lead Time", f"{avg_lead_time:.1f} days")
            
            with col4:
                avg_cost = suppliers['Supply_Cost'].mean()
                st.metric("Avg Supply Cost", f"${avg_cost:.2f}")
            
            st.markdown("---")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Reliability score distribution
                fig = px.histogram(
                    suppliers,
                    x='Reliability_Score',
                    title="Supplier Reliability Distribution",
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Lead time vs reliability scatter
                fig = px.scatter(
                    suppliers,
                    x='Lead_Time_Days',
                    y='Reliability_Score',
                    size='Supply_Cost',
                    title="Lead Time vs Reliability",
                    hover_data=['Supplier_ID']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Top/Bottom performers
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Top Performers")
                top_suppliers = suppliers.nlargest(5, 'Reliability_Score')[
                    ['Supplier_ID', 'Reliability_Score', 'Lead_Time_Days', 'Supply_Cost']
                ]
                st.dataframe(top_suppliers, use_container_width=True)
            
            with col2:
                st.subheader("⚠️ Needs Improvement")
                bottom_suppliers = suppliers.nsmallest(5, 'Reliability_Score')[
                    ['Supplier_ID', 'Reliability_Score', 'Lead_Time_Days', 'Supply_Cost']
                ]
                st.dataframe(bottom_suppliers, use_container_width=True)
            
            # Supplier comparison tool
            st.subheader("🔍 Supplier Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                supplier1 = st.selectbox("Select Supplier 1:", suppliers['Supplier_ID'].unique())
            
            with col2:
                supplier2 = st.selectbox("Select Supplier 2:", suppliers['Supplier_ID'].unique())
            
            if supplier1 != supplier2:
                comp_data = suppliers[suppliers['Supplier_ID'].isin([supplier1, supplier2])]
                
                # Comparison metrics
                metrics = ['Reliability_Score', 'Lead_Time_Days', 'Supply_Cost']
                comparison_df = comp_data.groupby('Supplier_ID')[metrics].mean().round(2)
                
                fig = go.Figure()
                
                for metric in metrics:
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=comparison_df.index,
                        y=comparison_df[metric],
                        text=comparison_df[metric],
                        textposition='auto'
                    ))
                
                fig.update_layout(
                    title="Supplier Comparison",
                    barmode='group',
                    yaxis_title="Value"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error in supplier analytics: {str(e)}")
            st.error("Error loading supplier data")
    
    def show_demand_forecasting(self, ml_manager):
        """Display demand forecasting dashboard"""
        st.header("📈 Demand Forecasting")
        
        # Check if universal model exists
        model_path = os.path.join(ml_manager.models_dir, "demand_forecast_universal_random_forest.pkl")
        model_available = os.path.exists(model_path)
        
        if model_available:
            st.success("✅ Universal demand forecasting model is ready for use")
        else:
            st.warning("⚠️ No trained model found. Please train the universal model first in ML Model Training section.")
            st.info("Go to ML Model Training → Demand Forecasting tab to train a model")
        
        try:
            products = self.db_manager.get_products()
            
            # Product selection
            selected_product = st.selectbox(
                "Select Product for Forecasting:",
                products['Product_ID'].tolist()
            )
            
            # Forecast period input
            forecast_days = st.slider("Forecast Period (days):", 7, 90, 30)
            
            # Generate forecast button
            if st.button("Generate Forecast", type="primary"):
                with st.spinner("Generating forecast..."):
                    predictions_df, message = ml_manager.predict_demand(selected_product, forecast_days)
                    
                    if predictions_df is not None:
                        st.success(message)
                        
                        # Get historical data for context
                        historical_data = self.db_manager.get_demand_forecast_data(selected_product)
                        historical_data['Date'] = pd.to_datetime(historical_data['Date'])
                        recent_history = historical_data.tail(60)  # Last 60 days
                        
                        # Create forecast visualization with proper layout
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=recent_history['Date'],
                            y=recent_history['Demand_Units'],
                            mode='lines+markers',
                            name='Historical Demand',
                            line=dict(color='blue'),
                            marker=dict(size=4)
                        ))
                        
                        # Forecasted data
                        fig.add_trace(go.Scatter(
                            x=predictions_df['Date'],
                            y=predictions_df['Predicted_Demand'],
                            mode='lines+markers',
                            name='Forecasted Demand',
                            line=dict(color='red', dash='dash'),
                            marker=dict(size=4)
                        ))
                        
                        fig.update_layout(
                            title=f"Demand Forecast for {selected_product}",
                            xaxis_title="Date",
                            yaxis_title="Demand Units",
                            hovermode='x unified',
                            height=500,
                            margin=dict(l=50, r=50, t=80, b=50)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast statistics
                        st.subheader("📊 Forecast Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Avg Daily Demand", f"{predictions_df['Predicted_Demand'].mean():.1f}")
                        
                        with col2:
                            st.metric("Max Daily Demand", f"{predictions_df['Predicted_Demand'].max():.1f}")
                        
                        with col3:
                            st.metric("Total Forecast", f"{predictions_df['Predicted_Demand'].sum():.0f}")
                        
                        with col4:
                            trend = "Increasing" if predictions_df['Predicted_Demand'].iloc[-1] > predictions_df['Predicted_Demand'].iloc[0] else "Decreasing"
                            st.metric("Trend", trend)
                        
                        # Forecast table
                        st.subheader("📋 Forecast Details")
                        display_df = predictions_df.copy()
                        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                        display_df['Predicted_Demand'] = display_df['Predicted_Demand'].round(1)
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                    else:
                        st.error(message)
            
            # Historical demand analysis
            st.subheader("📊 Historical Demand Analysis")
            
            historical_data = self.db_manager.get_demand_forecast_data(selected_product)
            if not historical_data.empty:
                historical_data['Date'] = pd.to_datetime(historical_data['Date'])
                
                # Time period selection
                period = st.selectbox("Analysis Period:", ["Last 30 Days", "Last 90 Days", "Last Year"])
                
                if period == "Last 30 Days":
                    cutoff_date = datetime.now() - timedelta(days=30)
                elif period == "Last 90 Days":
                    cutoff_date = datetime.now() - timedelta(days=90)
                else:
                    cutoff_date = datetime.now() - timedelta(days=365)
                
                filtered_history = historical_data[historical_data['Date'] >= cutoff_date]
                
                if not filtered_history.empty:
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Avg Demand", f"{filtered_history['Demand_Units'].mean():.1f}")
                    
                    with col2:
                        st.metric("Max Demand", f"{filtered_history['Demand_Units'].max()}")
                    
                    with col3:
                        st.metric("Min Demand", f"{filtered_history['Demand_Units'].min()}")
                    
                    with col4:
                        st.metric("Std Deviation", f"{filtered_history['Demand_Units'].std():.1f}")
                    
                    # Historical trend chart
                    fig = px.line(
                        filtered_history,
                        x='Date',
                        y='Demand_Units',
                        title=f"Historical Demand - {period}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No historical data available for the selected period")
            else:
                st.warning("No historical data available for this product")
            
        except Exception as e:
            logger.error(f"Error in demand forecasting: {str(e)}")
            st.error("Error loading forecasting data")
    
    def show_reports_export(self):
        """Display reports and export functionality"""
        st.header("📄 Reports & Export")
        
        try:
            # Report type selection
            report_type = st.selectbox(
                "Select Report Type:",
                ["Inventory Report", "Sales Report", "Supplier Performance", "Demand Analysis"]
            )
            
            # Date range selection
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
            with col2:
                end_date = st.date_input("End Date", datetime.now())
            
            if st.button("Generate Report", type="primary"):
                if report_type == "Inventory Report":
                    report_data = self._generate_inventory_report()
                elif report_type == "Sales Report":
                    report_data = self._generate_sales_report(start_date, end_date)
                elif report_type == "Supplier Performance":
                    report_data = self._generate_supplier_report()
                else:
                    report_data = self._generate_demand_report(start_date, end_date)
                
                if report_data is not None:
                    st.success("Report generated successfully!")
                    
                    # Display report
                    st.subheader(f"{report_type} Summary")
                    st.dataframe(report_data, use_container_width=True)
                    
                    # Download options
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        csv = report_data.to_csv(index=False)
                        st.download_button(
                            "Download CSV",
                            csv,
                            file_name=f"{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        excel_buffer = self.utils.dataframe_to_excel(report_data)
                        st.download_button(
                            "Download Excel",
                            excel_buffer,
                            file_name=f"{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    with col3:
                        json_data = report_data.to_json(orient='records', indent=2)
                        st.download_button(
                            "Download JSON",
                            json_data,
                            file_name=f"{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.json",
                            mime="application/json"
                        )
                else:
                    st.error("Error generating report")
            
        except Exception as e:
            logger.error(f"Error in reports dashboard: {str(e)}")
            st.error("Error loading reports interface")
    
    def _show_orders_trend_chart(self):
        """Show orders trend chart"""
        try:
            orders = self.db_manager.get_orders()
            orders['Order_Date'] = pd.to_datetime(orders['Order_Date'])
            
            daily_orders = orders.groupby(orders['Order_Date'].dt.date).size().reset_index()
            daily_orders.columns = ['Date', 'Order_Count']
            
            fig = px.line(
                daily_orders.tail(30),
                x='Date',
                y='Order_Count',
                title="Orders Trend (Last 30 Days)"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error("Error loading orders trend")
    
    def _show_inventory_status_chart(self):
        """Show inventory status chart"""
        try:
            inventory = self.db_manager.get_inventory()
            
            # Calculate stock status
            inventory['Stock_Status'] = inventory.apply(
                lambda row: 'Low Stock' if row['Current_Stock'] <= row['Reorder_Point']
                else 'Overstock' if row['Current_Stock'] > row['Reorder_Point'] * 2
                else 'Normal', axis=1
            )
            
            status_counts = inventory['Stock_Status'].value_counts()
            
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Inventory Status Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error("Error loading inventory status")
    
    def _show_supplier_performance_chart(self):
        """Show supplier performance chart"""
        try:
            suppliers = self.db_manager.get_suppliers()
            
            # Group by reliability score ranges
            suppliers['Reliability_Range'] = pd.cut(
                suppliers['Reliability_Score'],
                bins=[0, 0.7, 0.8, 0.9, 1.0],
                labels=['Poor', 'Fair', 'Good', 'Excellent']
            )
            
            reliability_counts = suppliers['Reliability_Range'].value_counts()
            
            fig = px.bar(
                x=reliability_counts.index,
                y=reliability_counts.values,
                title="Supplier Reliability Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error("Error loading supplier performance")
    
    def _show_category_sales_chart(self):
        """Show category sales chart"""
        try:
            orders = self.db_manager.get_orders()
            products = self.db_manager.get_products()
            
            order_details = orders.merge(products, on='Product_ID')
            order_details['Revenue'] = order_details['Quantity'] * order_details['Unit_Price']
            
            category_revenue = order_details.groupby('Category')['Revenue'].sum().reset_index()
            
            fig = px.bar(
                category_revenue,
                x='Category',
                y='Revenue',
                title="Revenue by Product Category"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error("Error loading category sales")
    
    def _generate_inventory_report(self):
        """Generate inventory report"""
        try:
            inventory = self.db_manager.get_inventory()
            products = self.db_manager.get_products()
            
            report = inventory.merge(products, on='Product_ID')
            report['Stock_Value'] = report['Current_Stock'] * report['Unit_Cost']
            report['Stock_Status'] = report.apply(
                lambda row: 'Low Stock' if row['Current_Stock'] <= row['Reorder_Point']
                else 'Overstock' if row['Current_Stock'] > row['Reorder_Point'] * 2
                else 'Normal', axis=1
            )
            
            return report[['Product_ID', 'Product_Name', 'Category', 'Current_Stock', 
                          'Reorder_Point', 'Stock_Value', 'Stock_Status']]
        except Exception as e:
            logger.error(f"Error generating inventory report: {str(e)}")
            return None
    
    def _generate_sales_report(self, start_date, end_date):
        """Generate sales report"""
        try:
            orders = self.db_manager.get_orders()
            products = self.db_manager.get_products()
            
            orders['Order_Date'] = pd.to_datetime(orders['Order_Date'])
            filtered_orders = orders[
                (orders['Order_Date'] >= pd.to_datetime(start_date)) &
                (orders['Order_Date'] <= pd.to_datetime(end_date))
            ]
            
            report = filtered_orders.merge(products, on='Product_ID')
            report['Revenue'] = report['Quantity'] * report['Unit_Price']
            
            return report[['Order_ID', 'Product_Name', 'Quantity', 'Unit_Price', 
                          'Revenue', 'Order_Date', 'Order_Status']]
        except Exception as e:
            logger.error(f"Error generating sales report: {str(e)}")
            return None
    
    def _generate_supplier_report(self):
        """Generate supplier performance report"""
        try:
            suppliers = self.db_manager.get_suppliers()
            products = self.db_manager.get_products()
            
            report = suppliers.merge(products, on='Product_ID')
            
            return report[['Supplier_ID', 'Product_Name', 'Lead_Time_Days', 
                          'Supply_Cost', 'Reliability_Score']]
        except Exception as e:
            logger.error(f"Error generating supplier report: {str(e)}")
            return None
    
    def _generate_demand_report(self, start_date, end_date):
        """Generate demand analysis report"""
        try:
            demand = self.db_manager.get_demand_history()
            products = self.db_manager.get_products()
            
            demand['Date'] = pd.to_datetime(demand['Date'])
            filtered_demand = demand[
                (demand['Date'] >= pd.to_datetime(start_date)) &
                (demand['Date'] <= pd.to_datetime(end_date))
            ]
            
            report = filtered_demand.merge(products, on='Product_ID')
            
            return report[['Product_ID', 'Product_Name', 'Date', 'Demand_Units']]
        except Exception as e:
            logger.error(f"Error generating demand report: {str(e)}")
            return None
