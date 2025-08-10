# Development Environment Setup Guide

**Complete setup for Dynamic Diffusion Model Optimization Development**

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Base Environment Setup](#base-environment-setup)
3. [ComfyUI Development Environment](#comfyui-development-environment)
4. [Custom Node Development Setup](#custom-node-development-setup)
5. [Profiling and Analysis Tools](#profiling-and-analysis-tools)
6. [Testing Framework](#testing-framework)
7. [Development Workflow](#development-workflow)
8. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware Verification
Before starting, verify your system meets the requirements:

```powershell
# Check GPU information
nvidia-smi

# Check system memory
Get-ComputerInfo | Select-Object TotalPhysicalMemory

# Check available disk space
Get-Volume
```

### Required Specifications
- **GPU**: NVIDIA RTX 3090 TI (24GB VRAM) ✓
- **System RAM**: 64GB minimum ✓
- **Storage**: 500GB+ free space for models and fragments
- **OS**: Windows 11 (with WSL2 recommended) or Linux
- **Python**: 3.10 or 3.11 (avoid 3.12 for compatibility)

---

## Base Environment Setup

### Step 1: Python Environment Management

```powershell
# Install Python 3.11 if not already installed
# Download from python.org or use winget
winget install Python.Python.3.11

# Verify Python installation
python --version
# Should output: Python 3.11.x

# Install pip and upgrade
python -m pip install --upgrade pip
```

### Step 2: Create Project Virtual Environment

```powershell
# Navigate to your development directory
cd c:\development

# Create project directory
mkdir diffusion-optimizer
cd diffusion-optimizer

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Verify activation (should show (venv) in prompt)
```

### Step 3: CUDA and PyTorch Setup

```powershell
# Install PyTorch with CUDA support
# Check https://pytorch.org/ for latest CUDA-compatible version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}')"
```

Expected output:
```
CUDA available: True
CUDA version: 12.1
Device count: 1
```

---

## ComfyUI Development Environment

### Step 1: ComfyUI Installation

```powershell
# Clone ComfyUI repository
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install ComfyUI dependencies
pip install -r requirements.txt

# Install additional development dependencies
pip install xformers  # For memory optimization
pip install accelerate  # For model loading optimization
```

### Step 2: ComfyUI Configuration

Create `extra_model_paths.yaml` for model organization:
```yaml
# extra_model_paths.yaml
checkpoints: "C:/development/models/checkpoints"
vae: "C:/development/models/vae"
loras: "C:/development/models/loras"
embeddings: "C:/development/models/embeddings"
custom_nodes: "C:/development/ComfyUI/custom_nodes"
```

### Step 3: Test ComfyUI Installation

```powershell
# Start ComfyUI
python main.py

# Should see output like:
# Total VRAM 24575 MB, total RAM 65536 MB
# Starting server
# To see the GUI go to: http://127.0.0.1:8188
```

Test in browser:
1. Go to `http://127.0.0.1:8188`
2. Load default workflow
3. Generate a test image to verify everything works

### Step 4: Download Test Models

```powershell
# Create models directory structure
mkdir C:\development\models\checkpoints
mkdir C:\development\models\vae

# Download SDXL base model for testing
# Place in C:\development\models\checkpoints\
# Example: sd_xl_base_1.0.safetensors

# Download SDXL VAE
# Place in C:\development\models\vae\
# Example: sdxl_vae.safetensors
```

---

## Custom Node Development Setup

### Step 1: Development Directory Structure

```powershell
# Create custom node directory
mkdir C:\development\ComfyUI\custom_nodes\diffusion-optimizer
cd C:\development\ComfyUI\custom_nodes\diffusion-optimizer

# Create development structure
mkdir nodes
mkdir utils
mkdir tests
mkdir docs

# Create initial files
New-Item __init__.py
New-Item nodes\__init__.py
New-Item utils\__init__.py
New-Item tests\__init__.py
```

### Step 2: Node Development Template

Create `nodes\profiler_model_loader.py`:
```python
import torch
import comfy.model_management as model_management
from comfy.model_base import BaseModel
import folder_paths
import time
import json
from typing import Dict, Any, Optional, Tuple

class ProfilerModelLoader:
    """
    Custom node for loading models with profiling capabilities
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "profiling_enabled": ("BOOLEAN", {"default": False}),
                "profiling_detail": (["basic", "detailed", "comprehensive"], {"default": "basic"}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "PROFILING_DATA")
    RETURN_NAMES = ("model", "profiling_data")
    FUNCTION = "load_checkpoint"
    CATEGORY = "optimization"
    
    def __init__(self):
        self.profiling_hooks = {}
        self.profiling_data = {}
        self.memory_tracker = MemoryTracker()
    
    def load_checkpoint(self, ckpt_name: str, profiling_enabled: bool = False, 
                       profiling_detail: str = "basic") -> Tuple[BaseModel, Dict[str, Any]]:
        """
        Load checkpoint with optional profiling hooks
        """
        print(f"Loading checkpoint: {ckpt_name}")
        print(f"Profiling enabled: {profiling_enabled}")
        
        # Load model normally first
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        model = comfy.sd.load_checkpoint_guess_config(
            ckpt_path, 
            output_vae=True, 
            output_clip=True, 
            embedding_directory=folder_paths.get_folder_paths("embeddings")
        )[0]
        
        if profiling_enabled:
            print(f"Installing profiling hooks (detail: {profiling_detail})")
            self.install_profiling_hooks(model.model, profiling_detail)
            
            profiling_context = {
                "enabled": True,
                "detail_level": profiling_detail,
                "model_name": ckpt_name,
                "hooks_installed": len(self.profiling_hooks),
                "timestamp": time.time()
            }
        else:
            profiling_context = {
                "enabled": False,
                "detail_level": "none",
                "model_name": ckpt_name,
                "hooks_installed": 0,
                "timestamp": time.time()
            }
        
        return (model, profiling_context)
    
    def install_profiling_hooks(self, model: torch.nn.Module, detail_level: str):
        """
        Install profiling hooks based on detail level
        """
        hook_count = 0
        
        for name, module in model.named_modules():
            if self.should_profile_module(module, detail_level):
                hook = self.create_profiling_hook(name, module)
                handle = module.register_forward_hook(hook)
                self.profiling_hooks[name] = handle
                hook_count += 1
        
        print(f"Installed {hook_count} profiling hooks")
    
    def should_profile_module(self, module: torch.nn.Module, detail_level: str) -> bool:
        """
        Determine if a module should be profiled based on detail level
        """
        if detail_level == "basic":
            # Profile only key components
            return isinstance(module, (
                torch.nn.Linear,
                torch.nn.Conv2d,
                torch.nn.MultiheadAttention
            ))
        elif detail_level == "detailed":
            # Profile more components
            return isinstance(module, (
                torch.nn.Linear,
                torch.nn.Conv2d,
                torch.nn.MultiheadAttention,
                torch.nn.LayerNorm,
                torch.nn.GroupNorm
            ))
        elif detail_level == "comprehensive":
            # Profile almost everything
            return len(list(module.children())) == 0  # Leaf modules only
        
        return False
    
    def create_profiling_hook(self, name: str, module: torch.nn.Module):
        """
        Create a profiling hook for a specific module
        """
        def hook(module, input, output):
            # Record profiling data
            if name not in self.profiling_data:
                self.profiling_data[name] = {
                    "activation_count": 0,
                    "total_magnitude": 0.0,
                    "memory_usage": [],
                    "execution_times": []
                }
            
            # Time execution
            start_time = time.perf_counter()
            
            # Calculate activation statistics
            if isinstance(output, torch.Tensor):
                magnitude = torch.norm(output).item()
                memory_usage = output.numel() * output.element_size()
                
                self.profiling_data[name]["activation_count"] += 1
                self.profiling_data[name]["total_magnitude"] += magnitude
                self.profiling_data[name]["memory_usage"].append(memory_usage)
            
            end_time = time.perf_counter()
            self.profiling_data[name]["execution_times"].append(end_time - start_time)
        
        return hook


class MemoryTracker:
    """
    Track GPU memory usage during model operations
    """
    
    def __init__(self):
        self.memory_snapshots = []
        self.baseline_memory = None
    
    def take_snapshot(self, label: str = ""):
        """
        Take a GPU memory snapshot
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            
            snapshot = {
                "label": label,
                "timestamp": time.time(),
                "allocated_mb": allocated / (1024 * 1024),
                "reserved_mb": reserved / (1024 * 1024),
                "free_mb": (torch.cuda.get_device_properties(0).total_memory - allocated) / (1024 * 1024)
            }
            
            self.memory_snapshots.append(snapshot)
            return snapshot
        
        return None
    
    def get_memory_usage_summary(self) -> Dict[str, Any]:
        """
        Get summary of memory usage over time
        """
        if not self.memory_snapshots:
            return {}
        
        allocated_values = [s["allocated_mb"] for s in self.memory_snapshots]
        
        return {
            "min_allocated_mb": min(allocated_values),
            "max_allocated_mb": max(allocated_values),
            "avg_allocated_mb": sum(allocated_values) / len(allocated_values),
            "total_snapshots": len(self.memory_snapshots),
            "snapshots": self.memory_snapshots
        }


# Node registration
NODE_CLASS_MAPPINGS = {
    "ProfilerModelLoader": ProfilerModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProfilerModelLoader": "Profiler Model Loader",
}
```

### Step 3: Node Registration

Create `__init__.py` in the custom node directory:
```python
from .nodes.profiler_model_loader import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
```

### Step 4: Development Testing

```powershell
# Restart ComfyUI to load custom nodes
cd C:\development\ComfyUI
python main.py

# Check console output for:
# Loading: custom_nodes\diffusion-optimizer
# Imported custom_nodes.diffusion-optimizer
```

---

## Profiling and Analysis Tools

### Step 1: Development Dependencies

```powershell
# Install profiling and analysis tools
pip install psutil  # System resource monitoring
pip install nvitop  # GPU monitoring
pip install tensorboard  # Logging and visualization
pip install matplotlib seaborn  # Data visualization
pip install pandas numpy  # Data analysis
pip install memory-profiler  # Memory profiling
pip install line-profiler  # Line-by-line profiling
```

### Step 2: GPU Monitoring Setup

Create `utils\gpu_monitor.py`:
```python
import psutil
import torch
import nvidia_ml_py3 as nvml
from typing import Dict, List
import time
import threading
from dataclasses import dataclass

@dataclass
class GPUSnapshot:
    timestamp: float
    gpu_utilization: float
    memory_used: int
    memory_total: int
    temperature: float
    power_draw: float

class GPUMonitor:
    """
    Real-time GPU monitoring for development and profiling
    """
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.snapshots: List[GPUSnapshot] = []
        self.monitor_thread = None
        
        # Initialize NVIDIA ML
        nvml.nvmlInit()
        self.device_count = nvml.nvmlDeviceGetCount()
        self.device_handle = nvml.nvmlDeviceGetHandleByIndex(0)  # Use first GPU
    
    def start_monitoring(self):
        """Start background GPU monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("GPU monitoring started")
    
    def stop_monitoring(self):
        """Stop GPU monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("GPU monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                time.sleep(self.sampling_interval)
            except Exception as e:
                print(f"GPU monitoring error: {e}")
                break
    
    def _take_snapshot(self) -> GPUSnapshot:
        """Take a single GPU measurement snapshot"""
        # GPU utilization
        utilization = nvml.nvmlDeviceGetUtilizationRates(self.device_handle)
        
        # Memory information
        memory_info = nvml.nvmlDeviceGetMemoryInfo(self.device_handle)
        
        # Temperature
        temperature = nvml.nvmlDeviceGetTemperature(self.device_handle, nvml.NVML_TEMPERATURE_GPU)
        
        # Power draw
        power_draw = nvml.nvmlDeviceGetPowerUsage(self.device_handle) / 1000.0  # Convert to watts
        
        return GPUSnapshot(
            timestamp=time.time(),
            gpu_utilization=utilization.gpu,
            memory_used=memory_info.used,
            memory_total=memory_info.total,
            temperature=temperature,
            power_draw=power_draw
        )
    
    def get_summary(self) -> Dict:
        """Get monitoring summary statistics"""
        if not self.snapshots:
            return {}
        
        gpu_utils = [s.gpu_utilization for s in self.snapshots]
        memory_uses = [s.memory_used / (1024**3) for s in self.snapshots]  # GB
        temperatures = [s.temperature for s in self.snapshots]
        power_draws = [s.power_draw for s in self.snapshots]
        
        return {
            "duration_minutes": (self.snapshots[-1].timestamp - self.snapshots[0].timestamp) / 60,
            "gpu_utilization": {
                "avg": sum(gpu_utils) / len(gpu_utils),
                "max": max(gpu_utils),
                "min": min(gpu_utils)
            },
            "memory_usage_gb": {
                "avg": sum(memory_uses) / len(memory_uses),
                "max": max(memory_uses),
                "min": min(memory_uses)
            },
            "temperature_c": {
                "avg": sum(temperatures) / len(temperatures),
                "max": max(temperatures)
            },
            "power_draw_w": {
                "avg": sum(power_draws) / len(power_draws),
                "max": max(power_draws)
            },
            "total_snapshots": len(self.snapshots)
        }
```

### Step 3: Profiling Utilities

Create `utils\profiling_utils.py`:
```python
import torch
import time
import functools
import cProfile
import pstats
import io
from contextlib import contextmanager
from typing import Dict, Any, Callable

class ProfilingContext:
    """
    Context manager for profiling code sections
    """
    
    def __init__(self, name: str, enable_cprofile: bool = False):
        self.name = name
        self.enable_cprofile = enable_cprofile
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.profiler = None
    
    def __enter__(self):
        print(f"Starting profiling: {self.name}")
        
        # Record start time
        self.start_time = time.perf_counter()
        
        # Record start memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_memory = torch.cuda.memory_allocated()
        
        # Start cProfile if enabled
        if self.enable_cprofile:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop cProfile
        if self.profiler:
            self.profiler.disable()
        
        # Record end memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.end_memory = torch.cuda.memory_allocated()
        
        # Record end time
        self.end_time = time.perf_counter()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print profiling summary"""
        duration = self.end_time - self.start_time
        print(f"Profiling complete: {self.name}")
        print(f"  Duration: {duration:.4f} seconds")
        
        if self.start_memory is not None and self.end_memory is not None:
            memory_delta = (self.end_memory - self.start_memory) / (1024 * 1024)  # MB
            print(f"  Memory delta: {memory_delta:.2f} MB")
            print(f"  Peak memory: {torch.cuda.max_memory_allocated() / (1024 * 1024):.2f} MB")
        
        if self.profiler:
            self.print_cprofile_stats()
    
    def print_cprofile_stats(self):
        """Print cProfile statistics"""
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions
        print("Top functions by cumulative time:")
        print(s.getvalue())

def profile_function(name: str = None, enable_cprofile: bool = False):
    """
    Decorator for profiling individual functions
    """
    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with ProfilingContext(func_name, enable_cprofile):
                return func(*args, **kwargs)
        return wrapper
    return decorator

@contextmanager
def memory_cleanup():
    """
    Context manager for memory cleanup
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
```

---

## Testing Framework

### Step 1: Test Structure

```powershell
# Create test directories
mkdir tests\unit
mkdir tests\integration
mkdir tests\fixtures

# Create test files
New-Item tests\test_profiler_node.py
New-Item tests\test_memory_tracking.py
New-Item tests\test_gpu_monitoring.py
New-Item tests\conftest.py  # pytest configuration
```

### Step 2: Test Dependencies

```powershell
pip install pytest pytest-cov pytest-mock pytest-asyncio
pip install hypothesis  # Property-based testing
pip install responses  # HTTP mocking
```

### Step 3: Basic Test Framework

Create `tests\conftest.py`:
```python
import pytest
import torch
import tempfile
import shutil
from pathlib import Path

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)

@pytest.fixture
def gpu_available():
    """Check if GPU is available for testing"""
    return torch.cuda.is_available()

@pytest.fixture
def small_test_model():
    """Create a small model for testing"""
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 1)
    )
    return model

@pytest.fixture(scope="session")
def test_data_dir():
    """Directory for test data and fixtures"""
    return Path(__file__).parent / "fixtures"
```

Create `tests\test_profiler_node.py`:
```python
import pytest
import torch
from unittest.mock import Mock, patch
import sys
import os

# Add custom node path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'custom_nodes', 'diffusion-optimizer'))

from nodes.profiler_model_loader import ProfilerModelLoader, MemoryTracker

class TestProfilerModelLoader:
    
    def test_initialization(self):
        """Test ProfilerModelLoader initialization"""
        loader = ProfilerModelLoader()
        assert loader.profiling_hooks == {}
        assert loader.profiling_data == {}
        assert isinstance(loader.memory_tracker, MemoryTracker)
    
    def test_should_profile_module_basic(self):
        """Test module profiling decision for basic level"""
        loader = ProfilerModelLoader()
        
        # Should profile these modules
        linear_module = torch.nn.Linear(10, 5)
        conv_module = torch.nn.Conv2d(3, 16, 3)
        
        assert loader.should_profile_module(linear_module, "basic") == True
        assert loader.should_profile_module(conv_module, "basic") == True
        
        # Should not profile these modules
        relu_module = torch.nn.ReLU()
        assert loader.should_profile_module(relu_module, "basic") == False
    
    def test_profiling_hook_creation(self, small_test_model):
        """Test profiling hook functionality"""
        loader = ProfilerModelLoader()
        
        # Create hook for first layer
        first_layer = small_test_model[0]
        hook = loader.create_profiling_hook("test_layer", first_layer)
        
        # Register hook
        handle = first_layer.register_forward_hook(hook)
        
        # Run forward pass
        test_input = torch.randn(5, 10)
        output = small_test_model(test_input)
        
        # Check profiling data was collected
        assert "test_layer" in loader.profiling_data
        assert loader.profiling_data["test_layer"]["activation_count"] == 1
        assert loader.profiling_data["test_layer"]["total_magnitude"] > 0
        
        # Cleanup
        handle.remove()

class TestMemoryTracker:
    
    def test_memory_snapshot(self, gpu_available):
        """Test memory snapshot functionality"""
        if not gpu_available:
            pytest.skip("GPU not available")
        
        tracker = MemoryTracker()
        snapshot = tracker.take_snapshot("test")
        
        assert snapshot is not None
        assert "allocated_mb" in snapshot
        assert "reserved_mb" in snapshot
        assert "free_mb" in snapshot
        assert snapshot["label"] == "test"
    
    def test_memory_usage_summary(self, gpu_available):
        """Test memory usage summary"""
        if not gpu_available:
            pytest.skip("GPU not available")
        
        tracker = MemoryTracker()
        
        # Take multiple snapshots
        tracker.take_snapshot("snapshot1")
        # Allocate some memory
        test_tensor = torch.randn(1000, 1000).cuda()
        tracker.take_snapshot("snapshot2")
        del test_tensor
        torch.cuda.empty_cache()
        tracker.take_snapshot("snapshot3")
        
        summary = tracker.get_memory_usage_summary()
        
        assert "min_allocated_mb" in summary
        assert "max_allocated_mb" in summary
        assert "avg_allocated_mb" in summary
        assert summary["total_snapshots"] == 3
```

### Step 4: Run Tests

```powershell
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=nodes --cov-report=html

# Run specific test
pytest tests/test_profiler_node.py::TestProfilerModelLoader::test_initialization -v
```

---

## Development Workflow

### Step 1: Development Cycle

```powershell
# Daily development workflow
cd C:\development\ComfyUI

# 1. Activate environment
.\venv\Scripts\Activate.ps1

# 2. Pull latest changes (if working with git)
git pull origin main

# 3. Run tests
pytest custom_nodes\diffusion-optimizer\tests\ -v

# 4. Start ComfyUI for development
python main.py

# 5. Development work in separate terminal
# Edit custom nodes, test in ComfyUI interface

# 6. After changes, restart ComfyUI and retest
```

### Step 2: Git Setup for Development

```powershell
# Initialize git repository for custom node
cd C:\development\ComfyUI\custom_nodes\diffusion-optimizer

git init
git add .
git commit -m "Initial custom node structure"

# Create .gitignore
@"
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.env
.pytest_cache/
.coverage
htmlcov/
.DS_Store
Thumbs.db
"@ | Out-File -FilePath .gitignore -Encoding utf8
```

### Step 3: Code Quality Tools

```powershell
# Install code quality tools
pip install black isort flake8 mypy

# Format code
black nodes/ utils/ tests/

# Sort imports
isort nodes/ utils/ tests/

# Check style
flake8 nodes/ utils/ tests/

# Type checking
mypy nodes/ utils/
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory
```python
# Add to your profiling code
try:
    # Your GPU operation
    result = model(input_tensor)
except torch.cuda.OutOfMemoryError:
    print("CUDA OOM - Current memory usage:")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    torch.cuda.empty_cache()
    raise
```

#### 2. ComfyUI Custom Node Not Loading
```powershell
# Check ComfyUI console for errors
# Common issues:
# - Missing __init__.py files
# - Import errors
# - Node registration issues

# Debug by adding print statements to __init__.py
print("Loading diffusion-optimizer custom node")
```

#### 3. Virtual Environment Issues
```powershell
# If venv activation fails
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# If packages not found
pip list  # Verify packages installed
which python  # Verify python path
```

#### 4. GPU Monitoring Errors
```powershell
# Install NVIDIA ML Python bindings
pip install nvidia-ml-py3

# If still failing, verify NVIDIA drivers
nvidia-smi
```

### Performance Optimization Tips

1. **Memory Management**
   - Use `torch.cuda.empty_cache()` regularly
   - Monitor memory usage with custom tracker
   - Use smaller batch sizes during development

2. **Development Speed**
   - Use smaller models during development
   - Cache profiling results
   - Use async operations where possible

3. **Debugging**
   - Add extensive logging
   - Use context managers for resource cleanup
   - Profile memory usage continuously

---

This development environment setup provides everything you need to start implementing the profiling system. The environment is designed to be:

- **Production-ready**: Uses best practices for Python development
- **Debuggable**: Comprehensive logging and monitoring
- **Testable**: Full testing framework setup
- **Maintainable**: Code quality tools and git integration

You're now ready to begin Phase 1 implementation with a solid foundation for development, testing, and debugging!
