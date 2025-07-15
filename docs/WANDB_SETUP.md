# W&B (Weights & Biases) Setup Guide

StreetUnveiler now uses W&B for experiment tracking instead of TensorBoard, providing better experiment management, visualization, and collaboration features.

## 🚀 Quick Setup

### 1. Install W&B
```bash
pip install wandb
```

### 2. Get Your API Key
1. Go to [https://wandb.ai/authorize](https://wandb.ai/authorize)
2. Sign up or log in
3. Copy your API key

### 3. Configure API Key

#### Option A: Environment Variable (Recommended)
```bash
export WANDB_API_KEY=your_api_key_here
```

#### Option B: Create Secret File
```bash
# Create secrets directory
mkdir -p secrets

# Add your API key
echo "your_api_key_here" > secrets/wandb_api_key.txt
```

#### Option C: Home Directory
```bash
echo "your_api_key_here" > ~/.wandb_api_key
```

### 4. Start Training
```bash
python train.py --config configs/training_config.yaml
```

## ✨ Features

### 📊 Automatic Logging
- **Loss metrics**: All training losses with individual components
- **Validation metrics**: L1 loss, PSNR for test/train sets  
- **Scene statistics**: Gaussian count, opacity histograms
- **Timing**: Iteration time, total training time

### 🖼️ Rich Visualizations
- **Rendered images**: Final renders, sky, disparity
- **Depth maps**: Rendered depth with automatic colormap
- **Normal maps**: Surface and rendered normals
- **Semantic maps**: Segmentation results and ground truth
- **Uncertainty maps**: Uncertainty visualizations
- **Alpha channels**: Rendering opacity

### 🔬 Experiment Organization
- Automatic project organization: "StreetUnveiler"
- Experiment names based on output directory
- Full configuration logging
- Run URLs for easy sharing

## 📈 W&B Dashboard

Once training starts, you'll see:
```
✅ W&B initialized: https://wandb.ai/your-username/StreetUnveiler/runs/run-id
```

Click the URL to view your experiment dashboard with:
- Real-time loss curves
- Image galleries updated during validation
- System metrics (GPU, CPU, memory)
- Configuration comparison across runs
- Notes and tags for experiment organization

## 🛠️ Troubleshooting

### API Key Not Found
```
⚠️  W&B API key not found. Logging will be disabled.
   Please create secrets/wandb_api_key.txt with your API key
   or set WANDB_API_KEY environment variable
```

**Solution**: Follow step 3 above to configure your API key.

### W&B Not Installed
```
Warning: wandb not installed. Install with: pip install wandb
```

**Solution**: 
```bash
pip install wandb
```

### Network Issues
If you're behind a firewall or proxy, see [W&B proxy documentation](https://docs.wandb.ai/guides/track/public-api-guide#using-the-api-behind-a-proxy).

## 🔒 Security Notes

- **Never commit API keys** to version control
- The `secrets/` directory is automatically gitignored
- Use environment variables in production/CI environments
- API keys provide full access to your W&B account

## 📚 Advanced Usage

### Custom Project Names
Modify `utils/wandb_utils.py`:
```python
init_wandb(project_name="MyCustomProject", ...)
```

### Custom Experiment Names
```python
init_wandb(experiment_name="my_experiment_v2", ...)
```

### Offline Mode
```bash
export WANDB_MODE=offline
python train.py --config configs/training_config.yaml
```

### Disable W&B
Simply don't install wandb or don't provide an API key - the system will automatically fall back to local-only logging.

## 🆚 Migration from TensorBoard

### What Changed
- ✅ **Removed**: TensorBoard dependencies
- ✅ **Added**: W&B integration with enhanced features
- ✅ **Improved**: Better image handling and visualization
- ✅ **Enhanced**: Experiment organization and sharing

### Benefits Over TensorBoard
- 🌐 **Cloud-based**: Access experiments from anywhere
- 🤝 **Collaboration**: Easy sharing and team access
- 📱 **Mobile**: View experiments on your phone
- 🔄 **Comparison**: Side-by-side experiment comparison
- 📝 **Notes**: Rich experiment documentation
- 🎯 **Hyperparameter tracking**: Automatic config logging

---

For more W&B features, see the [official documentation](https://docs.wandb.ai/). 