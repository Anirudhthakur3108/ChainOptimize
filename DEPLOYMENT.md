# Deployment Guide for ChainOptimize

## Quick Start

### 1. Local Development
```bash
# Clone the repository
git clone https://github.com/Anirudhthakur3108/ChainOptimize.git
cd ChainOptimize

# Install dependencies
pip install streamlit pandas plotly scikit-learn bcrypt faker numpy joblib openpyxl

# Run the application
streamlit run app.py --server.port 5000
```

### 2. Streamlit Cloud Deployment

#### Step 1: Prepare Repository
1. Ensure all files are committed to your GitHub repository
2. Make sure `.streamlit/config.toml` is present with proper configuration
3. Dependencies are listed in `pyproject.toml`

#### Step 2: Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select repository: `Anirudhthakur3108/ChainOptimize`
4. Set main file path: `app.py`
5. Click "Deploy"

#### Step 3: Configuration
- The app will automatically use the configuration in `.streamlit/config.toml`
- No additional environment variables required for basic functionality
- All dependencies will be installed automatically

## File Upload Instructions

Since there are git lock issues in the current environment, please manually upload these files to your GitHub repository:

### Core Application Files
1. **app.py** - Main application entry point
2. **auth.py** - Authentication and user management system
3. **database.py** - Data management layer
4. **dashboard.py** - Dashboard views and visualizations
5. **ml_models.py** - Machine learning models and training
6. **utils.py** - Utility functions and helpers
7. **data_generator.py** - Mock data generation

### Configuration Files
8. **pyproject.toml** - Project dependencies
9. **replit.md** - Project documentation and preferences
10. **.streamlit/config.toml** - Streamlit server configuration

### Documentation
11. **README.md** - Comprehensive project documentation
12. **.gitignore** - Git ignore patterns
13. **DEPLOYMENT.md** - This deployment guide

### Manual Upload Steps

1. **Go to your GitHub repository**: https://github.com/Anirudhthakur3108/ChainOptimize

2. **Upload core files**:
   - Click "Add file" → "Upload files"
   - Drag and drop or select: `app.py`, `auth.py`, `database.py`, `dashboard.py`, `ml_models.py`, `utils.py`, `data_generator.py`

3. **Upload configuration**:
   - Upload `pyproject.toml`, `replit.md`
   - Create `.streamlit` folder and upload `config.toml` inside it

4. **Upload documentation**:
   - Upload `README.md`, `.gitignore`, `DEPLOYMENT.md`

5. **Commit changes**:
   - Add commit message: "Initial commit: Complete Supply Chain Optimization Platform"
   - Click "Commit changes"

## Key Features Ready for Production

### ✅ Authentication System
- Role-based access control (Admin, Manager, Viewer)
- Secure password hashing
- User management interface
- Hierarchical permissions

### ✅ Machine Learning Pipeline
- Universal demand forecasting models
- Model training and management
- Inventory optimization algorithms
- Performance metrics tracking

### ✅ Dashboard Analytics
- Real-time KPI monitoring
- Interactive visualizations
- Multi-role access control
- Comprehensive reporting

### ✅ Data Management
- Mock data compatible with MySQL schema
- Scalable architecture for database integration
- Error handling and logging

## Production Considerations

### Database Integration
- Current version uses mock data
- Designed for easy MySQL integration
- Schema compatible with production databases

### Security
- BCrypt password hashing implemented
- Role-based access control
- Session management through Streamlit

### Performance
- Optimized data loading
- Efficient visualization rendering
- Model caching for faster predictions

### Monitoring
- Comprehensive logging system
- Error tracking and handling
- User action monitoring

## Default Login Credentials

For testing the deployed application:

- **Admin**: username=`admin`, password=`admin123`
- **Manager**: username=`manager1`, password=`manager123`
- **Viewer**: username=`viewer1`, password=`viewer123`

## Support

After deployment, if you encounter any issues:

1. Check Streamlit Cloud logs for errors
2. Verify all files were uploaded correctly
3. Ensure `.streamlit/config.toml` is present
4. Check that dependencies in `pyproject.toml` are correct

The application is fully functional and ready for production use with the mock data system, and designed for seamless database integration when ready.