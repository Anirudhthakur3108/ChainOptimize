# Supply Chain Optimization Platform

## Overview

This is a Python-based AI-driven Supply Chain Optimization Platform built with Streamlit. The application provides comprehensive supply chain management capabilities including demand forecasting, inventory optimization, and real-time dashboard analytics. The platform uses machine learning models for predictive analytics and optimization recommendations.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes (January 30, 2025)

✓ Implemented hierarchical user registration system - only logged-in users can register others
✓ Enhanced ML model training to use universal model for all products instead of individual product models
✓ Improved demand forecasting with better feature engineering and model architecture
✓ Fixed forecast visualization layout and positioning issues
✓ Added bulk inventory optimization capabilities
✓ Moved user registration to dedicated "User Management" section for better access control
✓ Fixed Model Management tab errors and added comprehensive model viewing interface
✓ Enhanced user management with user count statistics and removal functionality
✓ Removed demo login button and implemented proper hierarchy permissions
✓ Created comprehensive GitHub repository documentation and deployment guides
✓ Project ready for production deployment on Streamlit Cloud

## System Architecture

The application follows a modular architecture with clear separation of concerns:

**Frontend**: Streamlit-based web interface with interactive dashboards and visualizations
**Backend**: Python classes managing different aspects of the system (auth, database, ML, etc.)
**Data Layer**: Mock data generation with planned MySQL integration
**ML Pipeline**: Scikit-learn based models for demand forecasting and optimization
**Visualization**: Plotly for interactive charts and graphs

The architecture is designed to be scalable and maintainable, with each component handling specific responsibilities.

## Key Components

### Core Modules

1. **app.py** - Main application entry point and Streamlit configuration
2. **auth.py** - User authentication and session management with role-based access
3. **database.py** - Data management layer currently using mock data
4. **dashboard.py** - Dashboard views and KPI visualizations
5. **ml_models.py** - Machine learning models for forecasting and optimization
6. **utils.py** - Utility functions for data processing and formatting
7. **data_generator.py** - Mock data generation for testing and development

### Authentication System

- Role-based access control (admin, manager, viewer)
- BCrypt password hashing for security
- Session state management through Streamlit
- Mock users for development with planned database integration

### Machine Learning Components

- Demand forecasting using Random Forest and Linear Regression
- Model persistence using joblib
- Feature engineering for time series data
- Model evaluation metrics and performance tracking

## Data Flow

1. **Data Generation**: Mock data created for customers, products, suppliers, inventory, orders, and shipments
2. **Data Processing**: Raw data transformed into features for ML models and dashboard displays
3. **ML Processing**: Models trained on historical data to generate forecasts and recommendations
4. **Visualization**: Processed data and ML outputs displayed through interactive Plotly charts
5. **User Interaction**: Role-based access to different views and functionalities

## External Dependencies

### Python Packages
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **plotly**: Interactive visualizations
- **scikit-learn**: Machine learning algorithms
- **bcrypt**: Password hashing
- **faker**: Mock data generation
- **numpy**: Numerical computations
- **joblib**: Model serialization

### Planned Integrations
- **MySQL**: Database for production data storage
- **SQLAlchemy**: Database ORM layer
- **GitHub**: Model storage and version control

## Deployment Strategy

**Current State**: Development setup with mock data
**Target Deployment**: Streamlit Cloud with the following architecture:

1. **Application Hosting**: Streamlit Cloud for web interface
2. **Database**: MySQL for persistent data storage
3. **Model Storage**: GitHub repository for trained model files
4. **Model Loading**: Dynamic model loading from GitHub raw URLs
5. **Metadata Management**: MySQL models_metadata table for model versioning

### Database Schema Planning
- Supply chain data tables (customers, products, inventory, orders, etc.)
- User management tables with role-based permissions
- Models metadata table for ML model tracking and versioning

The application is designed to easily transition from mock data to live MySQL database integration while maintaining the same interface and functionality.