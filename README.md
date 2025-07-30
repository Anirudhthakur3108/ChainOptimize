# ChainOptimize - AI-Driven Supply Chain Optimization Platform

## Overview

ChainOptimize is a comprehensive AI-driven Supply Chain Optimization Platform built with Streamlit and Python. The platform provides advanced supply chain management capabilities including demand forecasting, inventory optimization, supplier analytics, and real-time dashboard insights using machine learning.

## Features

### 🔐 Authentication & User Management
- Role-based access control (Admin, Manager, Viewer)
- Hierarchical user registration system
- Secure password hashing with BCrypt
- User statistics and management interface

### 📊 Dashboard Analytics
- Real-time KPI monitoring
- Interactive visualizations with Plotly
- Inventory tracking and alerts
- Order analytics and trends
- Supplier performance metrics

### 🤖 Machine Learning Capabilities
- Universal demand forecasting models
- Inventory optimization algorithms
- Feature engineering for time series data
- Model performance tracking and management
- Support for Random Forest and Gradient Boosting

### 📈 Advanced Analytics
- Demand pattern analysis
- Stockout risk assessment
- Lead time optimization
- Supplier reliability scoring
- Cost analysis and reporting

## Architecture

The application follows a modular architecture with clear separation of concerns:

- **Frontend**: Streamlit-based web interface
- **Backend**: Python classes for business logic
- **Data Layer**: Mock data with MySQL schema compatibility
- **ML Pipeline**: Scikit-learn based predictive models
- **Visualization**: Interactive Plotly charts and graphs

## File Structure

```
ChainOptimize/
├── app.py                 # Main application entry point
├── auth.py               # Authentication and user management
├── database.py           # Data management layer
├── dashboard.py          # Dashboard views and visualizations
├── ml_models.py          # Machine learning models and training
├── utils.py              # Utility functions
├── data_generator.py     # Mock data generation
├── pyproject.toml        # Project dependencies
├── replit.md            # Project documentation
├── .streamlit/
│   └── config.toml      # Streamlit configuration
└── models/              # Trained ML models storage
```

## Installation & Setup

### Prerequisites
- Python 3.11+
- Git

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/Anirudhthakur3108/ChainOptimize.git
cd ChainOptimize
```

2. Install dependencies:
```bash
pip install streamlit pandas plotly scikit-learn bcrypt faker numpy joblib openpyxl
```

3. Run the application:
```bash
streamlit run app.py --server.port 5000
```

### Production Deployment

The application is configured for Streamlit Cloud deployment:

1. **Streamlit Cloud**: Connect your GitHub repository to Streamlit Cloud
2. **Configuration**: Uses `.streamlit/config.toml` for server settings
3. **Dependencies**: All packages listed in `pyproject.toml`

## Usage

### Default User Accounts

- **Admin**: username=`admin`, password=`admin123`
- **Manager**: username=`manager1`, password=`manager123`
- **Viewer**: username=`viewer1`, password=`viewer123`

### Key Workflows

1. **Train ML Models**: Admin → ML Model Training → Train Universal Model
2. **Generate Forecasts**: Any Role → Demand Forecasting → Select Product
3. **Optimize Inventory**: Manager/Admin → Inventory Optimization
4. **Manage Users**: Admin/Manager → User Management

## Machine Learning Models

### Demand Forecasting
- **Universal Model**: Trained on all products for cross-product learning
- **Algorithms**: Random Forest, Gradient Boosting
- **Features**: Historical demand, seasonality, product attributes
- **Performance Metrics**: MAE, R², RMSE

### Inventory Optimization
- **Safety Stock Calculation**: Statistical analysis with service levels
- **Reorder Point Optimization**: Lead time and demand variability
- **Risk Assessment**: Stockout probability scoring

## Database Schema

The platform uses a mock data structure compatible with MySQL:

- **customers**: Customer information and segmentation
- **products**: Product catalog and attributes
- **inventory**: Stock levels and warehouse data
- **orders**: Order history and tracking
- **suppliers**: Supplier information and performance
- **shipments**: Logistics and delivery tracking
- **demand_history**: Historical demand patterns
- **users**: User accounts and permissions

## Configuration

### Streamlit Configuration (`.streamlit/config.toml`)
```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
```

### Environment Variables
- No external API keys required for basic functionality
- All data currently uses mock generation
- Designed for easy MySQL integration

## Development Guidelines

### Code Style
- PEP 8 compliant Python code
- Modular architecture with clear separation
- Comprehensive error handling and logging
- Type hints and documentation

### Adding New Features
1. Follow existing module patterns
2. Update role-based permissions
3. Add appropriate visualizations
4. Include error handling
5. Update documentation

## Roadmap

### Planned Features
- [ ] MySQL database integration
- [ ] Advanced forecasting algorithms
- [ ] Real-time data streaming
- [ ] Mobile responsive design
- [ ] API endpoints for external integration
- [ ] Advanced reporting and exports

### Database Migration
- [ ] SQLAlchemy ORM integration
- [ ] Database migration scripts
- [ ] Production data connectors
- [ ] Backup and recovery systems

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. Please check the LICENSE file for details.

## Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation in `replit.md`
- Review the code comments for implementation details

## Acknowledgments

Built with modern Python tools:
- Streamlit for web interface
- Plotly for visualizations
- Scikit-learn for machine learning
- Pandas for data manipulation
- BCrypt for security

---

**ChainOptimize** - Optimizing supply chains with AI-driven insights.