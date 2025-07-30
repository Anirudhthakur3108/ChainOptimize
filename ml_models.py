import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import os
import logging

logger = logging.getLogger(__name__)

class MLManager:
    """Manages machine learning models for supply chain optimization"""
    
    def __init__(self, db_manager):
        """Initialize ML manager with database connection"""
        self.db_manager = db_manager
        self.models_dir = "models"
        self._ensure_models_directory()
    
    def _ensure_models_directory(self):
        """Create models directory if it doesn't exist"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def prepare_demand_forecast_data(self, product_id=None):
        """Prepare data for demand forecasting - all products by default"""
        try:
            demand_data = self.db_manager.get_demand_forecast_data(product_id)
            
            if demand_data.empty:
                return None, "No demand data available"
            
            # Convert date to datetime
            demand_data['Date'] = pd.to_datetime(demand_data['Date'])
            demand_data = demand_data.sort_values(['Product_ID', 'Date'])
            
            # Create features for each product
            features_list = []
            
            for product in demand_data['Product_ID'].unique():
                product_data = demand_data[demand_data['Product_ID'] == product].copy()
                
                # Create time-based features
                product_data['day_of_week'] = product_data['Date'].dt.dayofweek
                product_data['month'] = product_data['Date'].dt.month
                product_data['day_of_month'] = product_data['Date'].dt.day
                product_data['quarter'] = product_data['Date'].dt.quarter
                product_data['day_of_year'] = product_data['Date'].dt.dayofyear
                
                # Create lag features
                product_data['demand_lag_1'] = product_data['Demand_Units'].shift(1)
                product_data['demand_lag_7'] = product_data['Demand_Units'].shift(7)
                product_data['demand_lag_30'] = product_data['Demand_Units'].shift(30)
                
                # Create rolling averages
                product_data['demand_ma_7'] = product_data['Demand_Units'].rolling(window=7).mean()
                product_data['demand_ma_30'] = product_data['Demand_Units'].rolling(window=30).mean()
                product_data['demand_std_7'] = product_data['Demand_Units'].rolling(window=7).std()
                
                # Product encoding (numerical representation of product)
                product_data['product_encoded'] = hash(product) % 1000
                
                features_list.append(product_data)
            
            # Combine all products
            combined_data = pd.concat(features_list, ignore_index=True)
            
            # Drop rows with NaN values
            combined_data = combined_data.dropna()
            
            return combined_data, "Data prepared successfully"
            
        except Exception as e:
            logger.error(f"Error preparing demand forecast data: {str(e)}")
            return None, f"Error preparing data: {str(e)}"
    
    def train_demand_forecast_model(self, product_id=None, model_type='random_forest'):
        """Train demand forecasting model for all products"""
        try:
            # Prepare data for all products
            data, message = self.prepare_demand_forecast_data(None)  # Always train on all products
            if data is None:
                return None, message
            
            # Enhanced feature set including product information
            feature_columns = ['day_of_week', 'month', 'day_of_month', 'quarter', 'day_of_year',
                             'demand_lag_1', 'demand_lag_7', 'demand_lag_30',
                             'demand_ma_7', 'demand_ma_30', 'demand_std_7', 'product_encoded']
            
            X = data[feature_columns]
            y = data['Demand_Units']
            
            # Split data ensuring temporal order
            # Use last 20% of data chronologically for testing
            split_index = int(len(data) * 0.8)
            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
            
            # Train enhanced model
            if model_type == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == 'gradient_boosting':
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=8,
                    random_state=42
                )
            else:
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Feature importance
            feature_importance = dict(zip(feature_columns, model.feature_importances_))
            
            # Save model
            model_name = f"demand_forecast_universal_{model_type}"
            model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
            joblib.dump({
                'model': model,
                'features': feature_columns,
                'model_type': model_type,
                'metrics': {'mae': mae, 'mse': mse, 'r2': r2},
                'feature_importance': feature_importance,
                'trained_on': 'all_products'
            }, model_path)
            
            return {
                'model': model,
                'metrics': {'mae': mae, 'mse': mse, 'r2': r2},
                'predictions': y_pred,
                'actual': y_test,
                'model_path': model_path,
                'feature_importance': feature_importance
            }, "Universal demand forecasting model trained successfully"
            
        except Exception as e:
            logger.error(f"Error training demand forecast model: {str(e)}")
            return None, f"Error training model: {str(e)}"
    
    def predict_demand(self, product_id, days_ahead=30):
        """Predict future demand for a specific product using universal model"""
        try:
            # Load the universal model
            model_name = "demand_forecast_universal_random_forest"
            model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
            
            if not os.path.exists(model_path):
                return None, "Universal model not found. Please train the model first."
            
            model_data = joblib.load(model_path)
            model = model_data['model']
            features = model_data['features']
            
            # Get recent data for prediction
            demand_data = self.db_manager.get_demand_forecast_data(product_id)
            demand_data['Date'] = pd.to_datetime(demand_data['Date'])
            demand_data = demand_data.sort_values('Date').tail(60)  # Last 60 days
            
            if demand_data.empty:
                return None, f"No historical data found for product {product_id}"
            
            # Get product encoding
            product_encoded = hash(product_id) % 1000
            
            # Prepare recent data with features
            recent_data = demand_data.copy()
            recent_data['day_of_week'] = recent_data['Date'].dt.dayofweek
            recent_data['month'] = recent_data['Date'].dt.month
            recent_data['day_of_month'] = recent_data['Date'].dt.day
            recent_data['quarter'] = recent_data['Date'].dt.quarter
            recent_data['day_of_year'] = recent_data['Date'].dt.dayofyear
            
            # Create lag and rolling features
            recent_data['demand_lag_1'] = recent_data['Demand_Units'].shift(1)
            recent_data['demand_lag_7'] = recent_data['Demand_Units'].shift(7)
            recent_data['demand_lag_30'] = recent_data['Demand_Units'].shift(30)
            recent_data['demand_ma_7'] = recent_data['Demand_Units'].rolling(window=7).mean()
            recent_data['demand_ma_30'] = recent_data['Demand_Units'].rolling(window=30).mean()
            recent_data['demand_std_7'] = recent_data['Demand_Units'].rolling(window=7).std()
            recent_data['product_encoded'] = product_encoded
            
            # Get the last valid row for prediction base
            last_valid_data = recent_data.dropna().iloc[-1] if not recent_data.dropna().empty else recent_data.iloc[-1]
            
            predictions = []
            prediction_data = recent_data['Demand_Units'].tolist()
            
            for i in range(days_ahead):
                pred_date = pd.to_datetime(last_valid_data['Date']) + timedelta(days=i+1)
                
                # Create features for prediction
                pred_features = {
                    'day_of_week': pred_date.dayofweek,
                    'month': pred_date.month,
                    'day_of_month': pred_date.day,
                    'quarter': pred_date.quarter,
                    'day_of_year': pred_date.dayofyear,
                    'demand_lag_1': prediction_data[-1] if prediction_data else 0,
                    'demand_lag_7': prediction_data[-7] if len(prediction_data) >= 7 else prediction_data[0] if prediction_data else 0,
                    'demand_lag_30': prediction_data[-30] if len(prediction_data) >= 30 else prediction_data[0] if prediction_data else 0,
                    'demand_ma_7': np.mean(prediction_data[-7:]) if len(prediction_data) >= 7 else np.mean(prediction_data) if prediction_data else 0,
                    'demand_ma_30': np.mean(prediction_data[-30:]) if len(prediction_data) >= 30 else np.mean(prediction_data) if prediction_data else 0,
                    'demand_std_7': np.std(prediction_data[-7:]) if len(prediction_data) >= 7 else 0,
                    'product_encoded': product_encoded
                }
                
                # Make prediction
                X_pred = pd.DataFrame([pred_features])[features]
                pred = model.predict(X_pred)[0]
                pred = max(0, pred)  # Ensure non-negative demand
                predictions.append(pred)
                prediction_data.append(pred)
            
            # Create prediction DataFrame
            pred_dates = [pd.to_datetime(last_valid_data['Date']) + timedelta(days=i+1) for i in range(days_ahead)]
            predictions_df = pd.DataFrame({
                'Date': pred_dates,
                'Predicted_Demand': predictions,
                'Product_ID': product_id
            })
            
            return predictions_df, "Predictions generated successfully"
            
        except Exception as e:
            logger.error(f"Error predicting demand: {str(e)}")
            return None, f"Error making predictions: {str(e)}"
    
    def optimize_inventory(self, product_id):
        """Optimize inventory levels using demand predictions"""
        try:
            # Get current inventory
            inventory = self.db_manager.get_inventory()
            product_inventory = inventory[inventory['Product_ID'] == product_id]
            
            if product_inventory.empty:
                return None, "Product not found in inventory"
            
            # Get demand predictions
            predictions_df, message = self.predict_demand(product_id, days_ahead=60)
            if predictions_df is None:
                return None, message
            
            # Calculate optimization metrics
            avg_daily_demand = predictions_df['Predicted_Demand'].mean()
            max_daily_demand = predictions_df['Predicted_Demand'].max()
            demand_std = predictions_df['Predicted_Demand'].std()
            
            # Get supplier lead time
            suppliers = self.db_manager.get_suppliers()
            product_suppliers = suppliers[suppliers['Product_ID'] == product_id]
            avg_lead_time = product_suppliers['Lead_Time_Days'].mean() if not product_suppliers.empty else 7
            
            # Calculate optimal parameters
            service_level = 0.95  # 95% service level
            z_score = 1.645  # For 95% service level
            
            optimal_reorder_point = (avg_daily_demand * avg_lead_time) + (z_score * demand_std * np.sqrt(avg_lead_time))
            optimal_safety_stock = z_score * demand_std * np.sqrt(avg_lead_time)
            optimal_order_quantity = avg_daily_demand * 30  # 30-day supply
            
            current_stock = product_inventory.iloc[0]['Current_Stock']
            current_reorder_point = product_inventory.iloc[0]['Reorder_Point']
            current_safety_stock = product_inventory.iloc[0]['Safety_Stock']
            
            recommendations = {
                'current_stock': current_stock,
                'current_reorder_point': current_reorder_point,
                'current_safety_stock': current_safety_stock,
                'optimal_reorder_point': int(optimal_reorder_point),
                'optimal_safety_stock': int(optimal_safety_stock),
                'optimal_order_quantity': int(optimal_order_quantity),
                'avg_daily_demand': avg_daily_demand,
                'max_daily_demand': max_daily_demand,
                'avg_lead_time': avg_lead_time,
                'stockout_risk': 'High' if current_stock < optimal_reorder_point else 'Low'
            }
            
            return recommendations, "Inventory optimization completed"
            
        except Exception as e:
            logger.error(f"Error optimizing inventory: {str(e)}")
            return None, f"Error optimizing inventory: {str(e)}"
    
    def show_model_training_interface(self):
        """Display model training interface"""
        st.header("🤖 Machine Learning Model Training")
        
        # Model training tabs
        tab1, tab2, tab3 = st.tabs(["Demand Forecasting", "Inventory Optimization", "Model Management"])
        
        with tab1:
            st.subheader("Train Universal Demand Forecasting Model")
            st.info("This model will be trained on all products and can predict demand for any product in the system.")
            
            model_type = st.selectbox(
                "Model Type:",
                ["random_forest", "gradient_boosting"],
                help="Random Forest: Robust, handles non-linear patterns well | Gradient Boosting: Better accuracy, slower training"
            )
            
            if st.button("Train Universal Model", type="primary"):
                with st.spinner("Training universal model on all products... This may take a moment."):
                    result, message = self.train_demand_forecast_model(None, model_type)
                    
                    if result:
                        st.success(message)
                        st.info("✅ Model saved successfully and is now available in the Model Management tab for viewing and use in demand forecasting.")
                        
                        # Display metrics
                        metrics = result['metrics']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean Absolute Error", f"{metrics['mae']:.2f}")
                        with col2:
                            st.metric("R² Score", f"{metrics['r2']:.3f}")
                        with col3:
                            st.metric("Root MSE", f"{np.sqrt(metrics['mse']):.2f}")
                        
                        # Feature importance
                        if 'feature_importance' in result:
                            st.subheader("📊 Feature Importance")
                            importance_df = pd.DataFrame(
                                list(result['feature_importance'].items()),
                                columns=['Feature', 'Importance']
                            ).sort_values('Importance', ascending=False)
                            
                            fig = px.bar(
                                importance_df,
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title="Feature Importance in Demand Prediction"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Plot predictions vs actual
                        fig = go.Figure()
                        sample_size = min(1000, len(result['actual']))  # Limit points for performance
                        indices = np.random.choice(len(result['actual']), sample_size, replace=False)
                        
                        fig.add_trace(go.Scatter(
                            x=result['actual'].iloc[indices],
                            y=result['predictions'][indices],
                            mode='markers',
                            name='Predictions',
                            marker=dict(color='blue', opacity=0.6)
                        ))
                        
                        # Perfect prediction line
                        min_val = min(result['actual'].min(), result['predictions'].min())
                        max_val = max(result['actual'].max(), result['predictions'].max())
                        fig.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        fig.update_layout(
                            title="Model Performance: Actual vs Predicted Demand",
                            xaxis_title="Actual Demand",
                            yaxis_title="Predicted Demand"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(message)
        
        with tab2:
            st.subheader("Inventory Optimization")
            st.info("Uses demand forecasting model to optimize inventory levels for each product.")
            
            products = self.db_manager.get_products()
            
            # Option to optimize single product or all products
            optimization_scope = st.radio(
                "Optimization Scope:",
                ["Single Product", "All Products"],
                help="Single Product: Quick optimization for one item | All Products: Comprehensive optimization"
            )
            
            if optimization_scope == "Single Product":
                selected_product = st.selectbox(
                    "Select Product for Optimization:",
                    products['Product_ID'].tolist()
                )
                
                if st.button("Optimize Inventory", type="primary"):
                    with st.spinner("Optimizing inventory..."):
                        result, message = self.optimize_inventory(selected_product)
                        
                        if result:
                            self._display_optimization_results(result, message, selected_product)
                        else:
                            st.error(message)
            else:
                if st.button("Optimize All Inventories", type="primary"):
                    with st.spinner("Optimizing inventory for all products... This may take a moment."):
                        optimization_results = []
                        progress_bar = st.progress(0)
                        
                        for i, product_id in enumerate(products['Product_ID'].tolist()[:10]):  # Limit to first 10 for demo
                            result, message = self.optimize_inventory(product_id)
                            if result:
                                result['Product_ID'] = product_id
                                optimization_results.append(result)
                            progress_bar.progress((i + 1) / 10)
                        
                        if optimization_results:
                            st.success(f"Optimized inventory for {len(optimization_results)} products")
                            self._display_bulk_optimization_results(optimization_results)
                        else:
                            st.error("No products could be optimized")
        
        with tab3:
            st.subheader("Model Management")
            
            # Refresh button
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🔄 Refresh", help="Refresh the model list"):
                    st.rerun()
            
            # List existing models
            if os.path.exists(self.models_dir):
                try:
                    all_files = os.listdir(self.models_dir)
                    model_files = [f for f in all_files if f.endswith('.pkl')]
                    st.write(f"Debug: Found {len(all_files)} total files, {len(model_files)} model files")
                    if model_files:
                        st.write(f"Model files: {model_files}")
                except Exception as e:
                    st.error(f"Error reading models directory: {str(e)}")
                    model_files = []
                
                if model_files:
                    st.markdown(f"**Found {len(model_files)} trained model(s):**")
                    
                    # Create detailed model information
                    for model_file in model_files:
                        with st.expander(f"📁 {model_file}", expanded=True):
                            try:
                                model_path = os.path.join(self.models_dir, model_file)
                                model_data = joblib.load(model_path)
                                
                                # Model info
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Model Information:**")
                                    st.write(f"• Type: {model_data.get('model_type', 'Unknown')}")
                                    st.write(f"• Training Scope: {model_data.get('trained_on', 'Unknown')}")
                                    st.write(f"• Features: {len(model_data.get('features', []))}")
                                    
                                    # File info
                                    file_stats = os.stat(model_path)
                                    file_size = file_stats.st_size / (1024 * 1024)  # MB
                                    st.write(f"• File Size: {file_size:.2f} MB")
                                
                                with col2:
                                    st.markdown("**Performance Metrics:**")
                                    metrics = model_data.get('metrics', {})
                                    if metrics:
                                        st.metric("Mean Absolute Error", f"{metrics.get('mae', 0):.2f}")
                                        st.metric("R² Score", f"{metrics.get('r2', 0):.3f}")
                                        st.metric("Root MSE", f"{np.sqrt(metrics.get('mse', 0)):.2f}")
                                    else:
                                        st.write("No metrics available")
                                
                                # Model features
                                features = model_data.get('features', [])
                                if features:
                                    st.markdown("**Model Features:**")
                                    st.write(", ".join(features))
                                
                                # Action buttons
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    if st.button("🔍 Test Model", key=f"test_{model_file}"):
                                        st.info("Model testing functionality would be implemented here")
                                
                                with col2:
                                    if st.button("📊 View Details", key=f"details_{model_file}"):
                                        st.json(model_data)
                                
                                with col3:
                                    if st.button("🗑️ Delete", key=f"delete_{model_file}", type="secondary"):
                                        os.remove(model_path)
                                        st.success(f"Deleted {model_file}")
                                        st.rerun()
                                        
                            except Exception as e:
                                st.error(f"Error loading model: {str(e)}")
                else:
                    st.info("No trained models found. Train a model first in the 'Demand Forecasting' tab.")
            else:
                st.info("Models directory not found. Train a model first in the 'Demand Forecasting' tab.")
                
            # Model usage instructions
            st.markdown("---")
            st.markdown("**💡 Model Usage:**")
            st.write("• Train models in the 'Demand Forecasting' tab")
            st.write("• Use trained models in the 'Demand Forecasting' section of the main dashboard")
            st.write("• Models are automatically used for inventory optimization")
            st.write("• Universal models work with all products in the system")
    
    def _display_optimization_results(self, result, message, product_id):
        """Display optimization results for a single product"""
        st.success(message)
        
        # Display current vs optimal
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Inventory**")
            st.metric("Current Stock", result['current_stock'])
            st.metric("Reorder Point", result['current_reorder_point'])
            st.metric("Safety Stock", result['current_safety_stock'])
        
        with col2:
            st.markdown("**Optimized Inventory**")
            st.metric("Optimal Reorder Point", result['optimal_reorder_point'])
            st.metric("Optimal Safety Stock", result['optimal_safety_stock'])
            st.metric("Optimal Order Quantity", result['optimal_order_quantity'])
        
        # Risk assessment
        risk_color = "red" if result['stockout_risk'] == 'High' else "green"
        st.markdown(f"**Stockout Risk:** :{risk_color}[{result['stockout_risk']}]")
        
        # Demand insights
        st.subheader("📊 Demand Insights")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Daily Demand", f"{result['avg_daily_demand']:.1f}")
        with col2:
            st.metric("Max Daily Demand", f"{result['max_daily_demand']:.1f}")
        with col3:
            st.metric("Avg Lead Time", f"{result['avg_lead_time']:.1f} days")
        
        # Recommendations
        st.markdown("**📋 Recommendations:**")
        if result['current_stock'] < result['optimal_reorder_point']:
            st.warning(f"🚨 Immediate reorder recommended. Current stock ({result['current_stock']}) is below optimal reorder point ({result['optimal_reorder_point']})")
        else:
            st.success("✅ Current stock levels are adequate")
    
    def _display_bulk_optimization_results(self, results):
        """Display optimization results for multiple products"""
        # Convert to DataFrame for easy display
        results_df = pd.DataFrame(results)
        
        # Summary metrics
        high_risk_count = len(results_df[results_df['stockout_risk'] == 'High'])
        total_products = len(results_df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Products Analyzed", total_products)
        with col2:
            st.metric("High Risk Products", high_risk_count)
        with col3:
            st.metric("Risk Percentage", f"{(high_risk_count/total_products)*100:.1f}%")
        
        # Risk distribution chart
        risk_counts = results_df['stockout_risk'].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Stockout Risk Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Results table
        st.subheader("📋 Optimization Results")
        display_columns = ['Product_ID', 'current_stock', 'optimal_reorder_point', 
                          'optimal_safety_stock', 'stockout_risk', 'avg_daily_demand']
        st.dataframe(results_df[display_columns], use_container_width=True, hide_index=True)
