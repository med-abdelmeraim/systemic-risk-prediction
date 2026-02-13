# 🚀 Streamlit Deployment Guide
## Systemic Risk Prediction System

---

## 📦 Complete Package Contents

### Required Files (All Included!)

✅ **Application Files:**
- `app.py` - Main Streamlit dashboard (800+ lines)
- `requirements.txt` - Python dependencies

✅ **Trained Models:**
- `model_random_forest.pkl` - Random Forest classifier
- `model_gradient_boosting.pkl` - Gradient Boosting classifier  
- `model_logistic_regression.pkl` - Logistic Regression classifier
- `scaler.pkl` - Feature scaler
- `feature_names.txt` - List of feature names
- `best_model.txt` - Best model identifier

✅ **Data Files:**
- `final_120_rows_complete.csv` - Complete 120-row dataset
- `all_predictions.csv` - Model predictions
- `feature_importance.csv` - Feature importance scores
- `model_comparison.csv` - Model performance metrics

✅ **Training Scripts:**
- `train_final_model.py` - Model training pipeline
- `prepare_120_rows.py` - Data preparation

---

## 🎯 Quick Start (3 Steps!)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the App Locally

```bash
streamlit run app.py
```

### Step 3: Open in Browser

The app will automatically open at: `http://localhost:8501`

**That's it! 🎉**

---

## 📱 Dashboard Features

### 1. 📊 Main Dashboard
- **Real-time crisis probability**
- **Key risk metrics** (Leverage, Liquidity, VIX)
- **Alert system** (High Risk / Caution / Stable)
- **Crisis probability timeline**
- **Risk indicator trends**
- **Network & transaction metrics**

### 2. 🔮 Crisis Prediction Tool
- **Manual input** for custom predictions
- **CSV upload** for batch predictions
- **Risk score calculation**
- **Confidence intervals**

### 3. 📈 Historical Analysis
- **Time range filtering**
- **Multi-metric visualization**
- **Correlation heatmap**
- **Performance statistics**

### 4. 🎯 Feature Importance
- **Top N features ranking**
- **Interactive bar charts**
- **Category breakdown**
- **Detailed feature table**

### 5. ℹ️ About Page
- **System overview**
- **Model descriptions**
- **Technical details**
- **Disclaimer**

---

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Recommended - FREE!)

**Steps:**

1. **Create GitHub Repository**
   ```bash
   git init
   git add app.py requirements.txt *.pkl *.csv *.txt
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Click "New app"
   - Connect your GitHub repository
   - Select `app.py` as main file
   - Click "Deploy"!

3. **Access Your App**
   - URL: `https://YOUR_APP_NAME.streamlit.app`
   - Share with anyone!

**Pros:**
- ✅ FREE hosting
- ✅ Automatic HTTPS
- ✅ Easy updates (just push to GitHub)
- ✅ No server management

**Cons:**
- ⚠️ Public by default (can make private on Teams plan)
- ⚠️ Resource limits on free tier

---

### Option 2: Heroku

**Steps:**

1. **Create Procfile**
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Create runtime.txt**
   ```
   python-3.11
   ```

3. **Deploy**
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   heroku open
   ```

**Pros:**
- ✅ Scalable
- ✅ Custom domains
- ✅ Database integration

**Cons:**
- ⚠️ No longer free (starts at $5/month)

---

### Option 3: AWS EC2

**Steps:**

1. **Launch EC2 Instance**
   - Ubuntu 22.04 LTS
   - t2.micro (free tier eligible)

2. **SSH into Instance**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

3. **Install Dependencies**
   ```bash
   sudo apt update
   sudo apt install python3-pip -y
   pip3 install -r requirements.txt
   ```

4. **Run with Screen**
   ```bash
   screen -S streamlit
   streamlit run app.py --server.port=8501 --server.address=0.0.0.0
   # Press Ctrl+A then D to detach
   ```

5. **Configure Security Group**
   - Allow inbound traffic on port 8501

**Pros:**
- ✅ Full control
- ✅ Can run 24/7
- ✅ Free tier available

**Cons:**
- ⚠️ Requires server management
- ⚠️ Manual HTTPS setup

---

### Option 4: Docker (Advanced)

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build and Run:**
```bash
docker build -t systemic-risk-app .
docker run -p 8501:8501 systemic-risk-app
```

**Deploy to:**
- Google Cloud Run
- AWS ECS
- Azure Container Instances

---

## 🔧 Configuration

### Streamlit Config (.streamlit/config.toml)

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f6f8fb"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = true
```

### Performance Optimization

**For Large Datasets:**

```python
# Add to app.py
import streamlit as st

st.set_page_config(
    page_title="Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache data loading
@st.cache_data
def load_data():
    return pd.read_csv('data.csv')
```

---

## 📊 Monitoring & Analytics

### Add Google Analytics

```python
# Add to app.py
st.markdown("""
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
""", unsafe_allow_html=True)
```

### Monitor Performance

```bash
# Check app logs
streamlit run app.py --logger.level=debug
```

---

## 🐛 Troubleshooting

### Issue: Models not loading

**Solution:**
```python
# Check file paths
import os
print(os.listdir('.'))

# Use absolute paths if needed
model_path = os.path.join(os.path.dirname(__file__), 'model_random_forest.pkl')
```

### Issue: Memory error

**Solution:**
```python
# Reduce data in memory
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv('data.csv')
    return df[['essential', 'columns', 'only']]
```

### Issue: Slow performance

**Solutions:**
- Use `@st.cache_data` for data loading
- Use `@st.cache_resource` for models
- Reduce chart complexity
- Implement pagination for large tables

---

## 🔒 Security Best Practices

### 1. Environment Variables

```bash
# Create .env file (DON'T commit to git!)
API_KEY=your_secret_key
DATABASE_URL=your_db_url
```

```python
# Load in app.py
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('API_KEY')
```

### 2. Input Validation

```python
# Validate user inputs
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        # Validate schema
        required_cols = ['col1', 'col2', 'col3']
        if not all(col in df.columns for col in required_cols):
            st.error("Invalid file format!")
    except Exception as e:
        st.error(f"Error: {e}")
```

### 3. Rate Limiting

```python
import time

if 'last_request' not in st.session_state:
    st.session_state.last_request = 0

if time.time() - st.session_state.last_request < 1:
    st.warning("Please wait before making another request")
else:
    st.session_state.last_request = time.time()
    # Process request
```

---

## 📈 Scaling

### Horizontal Scaling

**Use Load Balancer:**
- Deploy multiple instances
- Use AWS ELB or nginx
- Session state in Redis

### Vertical Scaling

**Increase Resources:**
- Larger EC2 instance (t2.medium, t2.large)
- More memory for caching
- SSD for faster I/O

---

## 🎨 Customization

### Change Theme

```python
# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)
```

### Add Logo

```python
st.sidebar.image("logo.png", width=200)
```

### Custom Metrics

```python
st.metric(
    label="Custom KPI",
    value="$1.2M",
    delta="12%",
    delta_color="normal"
)
```

---

## 📚 Resources

### Official Documentation
- [Streamlit Docs](https://docs.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Streamlit Community](https://discuss.streamlit.io/)

### Tutorials
- [Streamlit 101](https://docs.streamlit.io/library/get-started)
- [Deploy to Cloud](https://docs.streamlit.io/streamlit-community-cloud/get-started)

### Examples
- [30 Days of Streamlit](https://30days.streamlit.app/)
- [Example Apps](https://github.com/streamlit/example-app-cv-risk-assessment)

---

## ✅ Pre-Launch Checklist

- [ ] All dependencies in requirements.txt
- [ ] Models and data files included
- [ ] App runs locally without errors
- [ ] Test all features (Dashboard, Prediction, Analysis)
- [ ] Mobile-responsive layout verified
- [ ] Error handling implemented
- [ ] Loading states for long operations
- [ ] Documentation updated
- [ ] Security review completed
- [ ] Performance optimized

---

## 🚀 Launch Commands

### Local Development
```bash
streamlit run app.py
```

### Production with Config
```bash
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
```

### With Custom Config
```bash
streamlit run app.py --config=.streamlit/config.toml
```

---

## 📞 Support

### Getting Help

1. **Check logs:**
   ```bash
   streamlit run app.py --logger.level=debug
   ```

2. **Community forum:**
   https://discuss.streamlit.io/

3. **GitHub issues:**
   https://github.com/streamlit/streamlit/issues

---

## 🎉 You're Ready to Deploy!

Your complete Streamlit app is ready to go. Just run:

```bash
streamlit run app.py
```

And share your amazing systemic risk prediction dashboard with the world! 🌍

**Good luck with your hackathon presentation! 🚀**
