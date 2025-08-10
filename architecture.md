# Technical Architecture: Dynamic Diffusion Model Optimization

**Detailed System Design and Component Architecture**

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [Data Flow Architecture](#data-flow-architecture)
4. [Memory Management](#memory-management)
5. [ComfyUI Integration](#comfyui-integration)
6. [Profiling Engine](#profiling-engine)
7. [Fragment Extraction](#fragment-extraction)
8. [Caching Strategy](#caching-strategy)
9. [Performance Optimization](#performance-optimization)
10. [Error Handling](#error-handling)
11. [Fragment Manifest & Packaging](#fragment-manifest--packaging)
12. [Metrics Pipeline & Reporting](#metrics-pipeline--reporting)

---

## System Overview

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     SYSTEM MEMORY (64GB)                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Full Models   │  │ Fragment Cache  │  │ Analysis Data   │  │
│  │   (20-100GB)    │  │   (10-50GB)     │  │    (1-5GB)      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GPU MEMORY (24GB)                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Active Fragment │  │ Working Tensors │  │ Output Buffers  │  │
│  │   (8-15GB)      │  │    (4-8GB)      │  │    (2-4GB)      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### System States

```
State 1: PROFILING MODE
┌────────────────┐    ┌─────────────┐    ┌──────────────┐
│  Full Model    │───▶│  Profiler   │───▶│ Usage Data   │
│   (Loaded)     │    │  (Active)   │    │ (Collecting) │
└────────────────┘    └─────────────┘    └──────────────┘

State 2: EXTRACTION MODE
┌────────────────┐    ┌─────────────┐    ┌──────────────┐
│  Usage Data    │───▶│ Extractor   │───▶│  Fragment    │
│  (Analysis)    │    │ (Processing)│    │  (Created)   │
└────────────────┘    └─────────────┘    └──────────────┘

State 3: OPTIMIZED MODE
┌────────────────┐    ┌─────────────┐    ┌──────────────┐
│   Fragment     │───▶│  Generator  │───▶│   Images     │
│   (Loaded)     │    │   (Fast)    │    │ (Multiple)   │
└────────────────┘    └─────────────┘    └──────────────┘
```

---

## Core Components

### 1. Model Manager
**Responsibility**: Handle loading, unloading, and switching between full models and fragments

```
ModelManager
├── FullModelLoader
│   ├── load_complete_model()
│   ├── validate_model_integrity()
│   └── memory_usage_estimation()
├── FragmentLoader
│   ├── load_fragment()
│   ├── validate_fragment()
│   └── fragment_compatibility_check()
└── MemoryOptimizer
    ├── garbage_collection()
    ├── memory_defragmentation()
    └── preload_strategies()
```

### 2. Profiling Engine
**Responsibility**: Monitor model usage during inference and collect activation data

```
ProfilingEngine
├── ActivationTracker
│   ├── hook_manager()
│   ├── activation_logger()
│   └── memory_access_monitor()
├── AttentionAnalyzer
│   ├── attention_weight_tracker()
│   ├── head_usage_analysis()
│   └── pattern_recognition()
└── LayerProfiler
    ├── layer_timing()
    ├── computational_load()
    └── skip_detection()
```

### 3. Fragment Extractor
**Responsibility**: Analyze profiling data and create optimized model fragments

```
FragmentExtractor
├── UsageAnalyzer
│   ├── statistical_analysis()
│   ├── threshold_determination()
│   └── importance_scoring()
├── WeightExtractor
│   ├── selective_copying()
│   ├── dependency_resolution()
│   └── architecture_preservation()
└── Optimizer
    ├── fragment_compression()
    ├── redundancy_elimination()
    └── validation_testing()
```

### 4. Cache Manager
**Responsibility**: Store, retrieve, and manage fragment libraries

```
CacheManager
├── FragmentStorage
│   ├── disk_serialization()
│   ├── compression_algorithms()
│   └── integrity_verification()
├── CachingStrategy
│   ├── LRU_eviction()
│   ├── usage_prediction()
│   └── preloading_hints()
└── MetadataManager
    ├── prompt_fingerprinting()
    ├── similarity_matching()
    └── version_control()
```

---

## Data Flow Architecture

### Phase 1: Profiling Workflow

```
┌─────────────┐
│   Prompt    │
│ "Dog on     │
│  jet ski"   │
└─────┬───────┘
      │
      ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Text Encode │───▶│ Full Model  │───▶│ Denoising   │
│   (CLIP)    │    │  Loading    │    │  Process    │
└─────────────┘    └─────────────┘    └─────┬───────┘
                         │                  │
                         ▼                  │
                   ┌─────────────┐          │
                   │  Profiler   │◀─────────┘
                   │  Hooks      │
                   └─────┬───────┘
                         │
                         ▼
                   ┌─────────────┐    ┌─────────────┐
                   │ Usage Data  │───▶│ Generated   │
                   │ Collection  │    │   Image     │
                   └─────────────┘    └─────────────┘
```

### Phase 2: Fragment Creation Workflow

```
┌─────────────┐
│ Usage Data  │
│ (Raw Logs)  │
└─────┬───────┘
      │
      ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Analysis   │───▶│ Threshold   │───▶│ Component   │
│  Engine     │    │ Calculation │    │ Selection   │
└─────────────┘    └─────────────┘    └─────┬───────┘
                                           │
                                           ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Fragment   │◀───│ Validation  │◀───│ Weight      │
│  Storage    │    │  Testing    │    │ Extraction  │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Phase 3: Optimized Generation Workflow

```
┌─────────────┐
│Same Prompt  │
│Different    │
│   Seed      │
└─────┬───────┘
      │
      ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Cache Lookup│───▶│ Fragment    │───▶│ Fast        │
│ (Fragment   │    │ Loading     │    │ Generation  │
│  Exists?)   │    │             │    │             │
└─────┬───────┘    └─────────────┘    └─────┬───────┘
      │                                     │
      │ (Cache Miss)                        ▼
      ▼                               ┌─────────────┐
┌─────────────┐                       │ Multiple    │
│ Fallback to │                       │ Images      │
│ Full Model  │                       │ (Batch)     │
└─────────────┘                       └─────────────┘
```

---

## Memory Management

### Memory Layout Strategy

```
System Memory (64GB) Allocation:
├── Operating System           (8GB)
├── Full Model Storage        (20-40GB)
├── Fragment Library          (10-20GB)
├── Analysis Workspace        (5GB)
├── Application Code          (2GB)
└── Free Buffer              (9-19GB)

GPU Memory (24GB) Allocation:
├── Model Weights            (8-15GB)
├── Activation Tensors       (4-6GB)
├── Working Buffers          (2-4GB)
├── Output Tensors           (1-2GB)
└── CUDA Runtime             (1-2GB)
```

### Dynamic Memory Management

```python
class MemoryManager:
    def __init__(self):
        self.gpu_budget = 24 * 1024**3  # 24GB
        self.fragment_cache = {}
        self.memory_tracker = GPUMemoryTracker()
    
    def allocate_for_fragment(self, fragment_size):
        # Estimate memory requirements
        required_memory = fragment_size + self.estimate_runtime_overhead()
        
        if required_memory > self.gpu_budget:
            return self.handle_oversized_fragment(fragment_size)
        
        # Ensure sufficient free memory
        self.ensure_free_memory(required_memory)
        return self.load_fragment_to_gpu()
    
    def memory_pressure_handling(self):
        # Progressive memory management strategies
        strategies = [
            self.clear_unused_tensors,
            self.compress_activations,
            self.offload_to_cpu,
            self.fragment_swapping
        ]
        
        for strategy in strategies:
            if self.check_memory_available():
                break
            strategy()
```

### Fragment Size Optimization

```
Fragment Size Calculation:
Base Model Size: M_base
Usage Threshold: T (0.1 - 0.9)
Dependency Overhead: D (10-30%)

Fragment Size = (M_base × T) + (M_base × D)

Example:
- Base Model: 20GB
- Threshold: 0.3 (30% of weights used)
- Dependency: 0.2 (20% overhead)
- Fragment: (20GB × 0.3) + (20GB × 0.2) = 6GB + 4GB = 10GB
```

---

## ComfyUI Integration

### Custom Node Architecture

```
ComfyUI Node Hierarchy:
├── ProfilerModelLoader (extends ModelLoader)
│   ├── Input: model_name, profiling_enabled
│   ├── Output: model, profiling_data
│   └── Function: Load model with profiling hooks
├── FragmentExtractor (new node)
│   ├── Input: profiling_data, extraction_threshold
│   ├── Output: fragment_metadata
│   └── Function: Create optimized fragment
├── FragmentLoader (extends ModelLoader)
│   ├── Input: fragment_metadata
│   ├── Output: optimized_model
│   └── Function: Load fragment for fast generation
└── BatchGenerator (extends KSampler)
    ├── Input: model, prompt, seed_list
    ├── Output: image_batch
    └── Function: Generate multiple variations
```

### Workflow Integration

```json
{
  "workflow": {
    "1": {
      "class_type": "ProfilerModelLoader",
      "inputs": {
        "model_name": "stable-diffusion-xl-base",
        "profiling_enabled": true
      }
    },
    "2": {
      "class_type": "CLIPTextEncode",
      "inputs": {
        "text": "A dog on a jet ski",
        "clip": ["1", 1]
      }
    },
    "3": {
      "class_type": "KSampler",
      "inputs": {
        "model": ["1", 0],
        "positive": ["2", 0],
        "profiling_mode": true
      }
    },
    "4": {
      "class_type": "FragmentExtractor",
      "inputs": {
        "profiling_data": ["3", 1],
        "threshold": 0.3
      }
    },
    "5": {
      "class_type": "FragmentLoader",
      "inputs": {
        "fragment_metadata": ["4", 0]
      }
    },
    "6": {
      "class_type": "BatchGenerator",
      "inputs": {
        "model": ["5", 0],
        "prompt": ["2", 0],
        "seed_list": [1, 2, 3, 4, 5]
      }
    }
  }
}
```

---

## Profiling Engine

### Activation Tracking System

```python
class ActivationTracker:
    def __init__(self):
        self.hooks = {}
        self.activation_data = {}
        self.memory_access_log = []
    
    def register_hooks(self, model):
        for name, module in model.named_modules():
            if self.should_track(module):
                hook = module.register_forward_hook(
                    self.create_activation_hook(name)
                )
                self.hooks[name] = hook
    
    def create_activation_hook(self, module_name):
        def hook(module, input, output):
            # Track activation statistics
            activation_stats = {
                'mean': torch.mean(output).item(),
                'std': torch.std(output).item(),
                'sparsity': self.calculate_sparsity(output),
                'memory_usage': output.numel() * output.element_size()
            }
            self.activation_data[module_name] = activation_stats
            
            # Log memory access patterns
            self.memory_access_log.append({
                'module': module_name,
                'timestamp': time.time(),
                'memory_delta': self.get_memory_delta()
            })
        
        return hook
```

### Attention Pattern Analysis

```python
class AttentionAnalyzer:
    def analyze_attention_patterns(self, attention_weights, prompt_tokens):
        # Analyze which attention heads are most active
        head_importance = self.calculate_head_importance(attention_weights)
        
        # Track token-to-token attention patterns
        token_interactions = self.analyze_token_interactions(
            attention_weights, prompt_tokens
        )
        
        # Identify redundant attention patterns
        redundant_heads = self.find_redundant_heads(attention_weights)
        
        return {
            'important_heads': head_importance,
            'token_patterns': token_interactions,
            'redundant_heads': redundant_heads
        }
    
    def calculate_head_importance(self, attention_weights):
        # Calculate importance based on:
        # 1. Entropy of attention distribution
        # 2. Magnitude of attention weights
        # 3. Variance across different prompts
        
        importance_scores = {}
        for layer_idx, layer_attention in enumerate(attention_weights):
            for head_idx, head_attention in enumerate(layer_attention):
                entropy = self.calculate_entropy(head_attention)
                magnitude = torch.mean(torch.abs(head_attention))
                variance = torch.var(head_attention)
                
                importance_scores[f"layer_{layer_idx}_head_{head_idx}"] = {
                    'entropy': entropy.item(),
                    'magnitude': magnitude.item(),
                    'variance': variance.item(),
                    'composite_score': self.composite_importance(
                        entropy, magnitude, variance
                    )
                }
        
        return importance_scores
```

---

## Fragment Extraction

### Weight Selection Algorithm

```python
class WeightExtractor:
    def __init__(self, threshold=0.3, dependency_analysis=True):
        self.threshold = threshold
        self.dependency_analysis = dependency_analysis
        self.importance_calculator = ImportanceCalculator()
    
    def extract_fragment(self, model, usage_data):
        # Phase 1: Calculate importance scores for all weights
        importance_scores = self.calculate_weight_importance(model, usage_data)
        
        # Phase 2: Apply threshold filtering
        selected_components = self.apply_threshold(importance_scores)
        
        # Phase 3: Resolve dependencies
        if self.dependency_analysis:
            selected_components = self.resolve_dependencies(
                selected_components, model
            )
        
        # Phase 4: Extract and package fragment
        fragment = self.create_fragment(model, selected_components)
        
        return fragment
    
    def calculate_weight_importance(self, model, usage_data):
        importance_scores = {}
        
        for name, param in model.named_parameters():
            # Combine multiple importance metrics
            activation_importance = self.get_activation_importance(name, usage_data)
            gradient_importance = self.get_gradient_importance(name, usage_data)
            magnitude_importance = self.get_magnitude_importance(param)
            
            # Weighted combination
            composite_score = (
                0.4 * activation_importance +
                0.4 * gradient_importance +
                0.2 * magnitude_importance
            )
            
            importance_scores[name] = composite_score
        
        return importance_scores
    
    def resolve_dependencies(self, selected_components, model):
        # Analyze architectural dependencies
        dependency_graph = self.build_dependency_graph(model)
        
        # Add missing dependencies
        extended_selection = set(selected_components)
        for component in selected_components:
            dependencies = dependency_graph.get_dependencies(component)
            extended_selection.update(dependencies)
        
        return list(extended_selection)
```

### Fragment Validation System

```python
class FragmentValidator:
    def __init__(self, quality_threshold=0.95):
        self.quality_threshold = quality_threshold
        self.test_prompts = self.load_validation_prompts()
    
    def validate_fragment(self, original_model, fragment, test_prompt):
        # Generate reference image with full model
        reference_image = self.generate_with_model(original_model, test_prompt)
        
        # Generate test image with fragment
        test_image = self.generate_with_model(fragment, test_prompt)
        
        # Calculate similarity metrics
        similarity_scores = {
            'ssim': self.calculate_ssim(reference_image, test_image),
            'lpips': self.calculate_lpips(reference_image, test_image),
            'clip_score': self.calculate_clip_similarity(reference_image, test_image),
            'fid': self.calculate_fid([reference_image], [test_image])
        }
        
        # Composite quality score
        quality_score = self.calculate_composite_quality(similarity_scores)
        
        return {
            'quality_score': quality_score,
            'passes_threshold': quality_score >= self.quality_threshold,
            'detailed_metrics': similarity_scores
        }
```

---

## Caching Strategy

### Fragment Identification and Storage

```python
class FragmentCache:
    def __init__(self, cache_dir, max_size_gb=50):
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size_gb * 1024**3
        self.metadata_db = FragmentMetadataDB()
    
    def generate_fragment_key(self, prompt, model_name, extraction_params):
        # Create unique identifier for fragment
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        param_hash = hashlib.sha256(
            json.dumps(extraction_params, sort_keys=True).encode()
        ).hexdigest()[:8]
        
        return f"{model_name}_{prompt_hash}_{param_hash}"
    
    def store_fragment(self, fragment_key, fragment_data, metadata):
        # Compress fragment for storage
        compressed_data = self.compress_fragment(fragment_data)
        
        # Store to disk
        fragment_path = self.cache_dir / f"{fragment_key}.fragment"
        with open(fragment_path, 'wb') as f:
            f.write(compressed_data)
        
        # Update metadata
        self.metadata_db.store_metadata(fragment_key, {
            **metadata,
            'file_path': str(fragment_path),
            'compressed_size': len(compressed_data),
            'created_timestamp': time.time()
        })
        
        # Enforce cache size limits
        self.enforce_cache_limits()
    
    def load_fragment(self, fragment_key):
        metadata = self.metadata_db.get_metadata(fragment_key)
        if not metadata:
            return None
        
        # Load and decompress
        with open(metadata['file_path'], 'rb') as f:
            compressed_data = f.read()
        
        fragment_data = self.decompress_fragment(compressed_data)
        
        # Update access timestamp
        self.metadata_db.update_access_time(fragment_key)
        
        return fragment_data
```

### Intelligent Caching Strategies

```python
class CachingStrategy:
    def __init__(self):
        self.usage_predictor = UsagePredictor()
        self.similarity_analyzer = PromptSimilarityAnalyzer()
    
    def should_cache_fragment(self, prompt, extraction_metadata):
        # Factors for caching decision:
        
        # 1. Extraction cost (how expensive was it to create?)
        extraction_cost = extraction_metadata['processing_time']
        
        # 2. Fragment size (smaller fragments are cheaper to cache)
        fragment_size = extraction_metadata['fragment_size']
        
        # 3. Prompt popularity (similar prompts seen before?)
        prompt_popularity = self.similarity_analyzer.get_popularity_score(prompt)
        
        # 4. Predicted reuse probability
        reuse_probability = self.usage_predictor.predict_reuse(prompt)
        
        # Weighted decision
        cache_score = (
            0.3 * min(extraction_cost / 60, 1.0) +  # Normalize to max 60s
            0.2 * max(0, 1.0 - fragment_size / (10 * 1024**3)) +  # Prefer <10GB
            0.2 * prompt_popularity +
            0.3 * reuse_probability
        )
        
        return cache_score > 0.6
    
    def find_similar_fragments(self, new_prompt, similarity_threshold=0.8):
        # Find cached fragments that might work for similar prompts
        cached_fragments = self.metadata_db.get_all_fragments()
        
        similar_fragments = []
        for fragment_key, metadata in cached_fragments.items():
            similarity = self.similarity_analyzer.calculate_similarity(
                new_prompt, metadata['original_prompt']
            )
            
            if similarity >= similarity_threshold:
                similar_fragments.append({
                    'key': fragment_key,
                    'similarity': similarity,
                    'metadata': metadata
                })
        
        # Sort by similarity
        similar_fragments.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar_fragments
```

---

## Performance Optimization

### GPU Memory Optimization

```python
class GPUOptimizer:
    def __init__(self):
        self.memory_pool = torch.cuda.memory_pool()
        self.allocation_tracker = AllocationTracker()
    
    def optimize_memory_layout(self, fragment):
        # Reorder tensors for optimal memory access patterns
        optimized_fragment = self.reorder_tensors(fragment)
        
        # Apply memory alignment optimizations
        aligned_fragment = self.align_memory_access(optimized_fragment)
        
        # Enable memory pool optimizations
        self.configure_memory_pool(aligned_fragment)
        
        return aligned_fragment
    
    def dynamic_batch_sizing(self, available_memory, fragment_size):
        # Calculate optimal batch size based on available memory
        overhead_per_sample = self.estimate_generation_overhead()
        usable_memory = available_memory - fragment_size
        
        max_batch_size = usable_memory // overhead_per_sample
        
        # Account for memory fragmentation
        safe_batch_size = int(max_batch_size * 0.8)
        
        return max(1, safe_batch_size)
```

### CPU-GPU Transfer Optimization

```python
class TransferOptimizer:
    def __init__(self):
        self.transfer_queue = asyncio.Queue()
        self.compression_cache = {}
    
    async def optimized_transfer(self, data, transfer_type='cpu_to_gpu'):
        # Apply compression if beneficial
        if self.should_compress(data):
            compressed_data = await self.async_compress(data)
            transfer_data = compressed_data
        else:
            transfer_data = data
        
        # Use pinned memory for faster transfers
        if transfer_type == 'cpu_to_gpu':
            pinned_data = self.pin_memory(transfer_data)
            gpu_data = await self.async_gpu_transfer(pinned_data)
            
            if self.should_compress(data):
                gpu_data = await self.async_decompress(gpu_data)
            
            return gpu_data
        
        elif transfer_type == 'gpu_to_cpu':
            cpu_data = await self.async_cpu_transfer(transfer_data)
            return cpu_data
    
    def should_compress(self, data):
        # Compression is beneficial if:
        # 1. Data size > compression threshold
        # 2. Transfer time > compression + decompression time
        # 3. Compression ratio > minimum ratio
        
        data_size = data.numel() * data.element_size()
        compression_threshold = 100 * 1024 * 1024  # 100MB
        
        if data_size < compression_threshold:
            return False
        
        estimated_compression_ratio = self.estimate_compression_ratio(data)
        estimated_transfer_time = self.estimate_transfer_time(data_size)
        estimated_compression_time = self.estimate_compression_time(data_size)
        
        return (estimated_transfer_time > estimated_compression_time and 
                estimated_compression_ratio > 0.7)
```

---

## Error Handling

### Comprehensive Error Recovery

```python
class ErrorHandler:
    def __init__(self):
        self.fallback_strategies = [
            self.reduce_fragment_size,
            self.fallback_to_cpu_generation,
            self.fallback_to_full_model,
            self.emergency_memory_cleanup
        ]
    
    def handle_memory_error(self, error, context):
        logger.error(f"Memory error in {context}: {error}")
        
        for strategy in self.fallback_strategies:
            try:
                result = strategy(context)
                if result.success:
                    logger.info(f"Recovered using {strategy.__name__}")
                    return result
            except Exception as fallback_error:
                logger.warning(f"Fallback {strategy.__name__} failed: {fallback_error}")
        
        # If all fallbacks fail, raise original error
        raise error
    
    def handle_fragment_corruption(self, fragment_key):
        # Attempt to recover from backup
        backup_fragment = self.try_load_backup(fragment_key)
        if backup_fragment:
            return backup_fragment
        
        # Regenerate fragment if original data available
        if self.can_regenerate(fragment_key):
            return self.regenerate_fragment(fragment_key)
        
        # Fall back to full model
        logger.warning(f"Fragment {fragment_key} corrupted, falling back to full model")
        return None
    
    def validate_system_requirements(self):
        requirements = {
            'gpu_memory': 24 * 1024**3,  # 24GB
            'system_memory': 64 * 1024**3,  # 64GB
            'cuda_version': '12.0',
            'pytorch_version': '2.0'
        }
        
        validation_results = {}
        for requirement, minimum_value in requirements.items():
            actual_value = self.get_system_value(requirement)
            validation_results[requirement] = {
                'required': minimum_value,
                'actual': actual_value,
                'sufficient': actual_value >= minimum_value
            }
        
        return validation_results
```

---

## Fragment Manifest & Packaging

### Purpose
Provide a reproducible, auditable description of any generated fragment so that later runs can (a) verify compatibility with a base model, (b) reproduce quality/performance evaluations, and (c) enable cache de-duplication and sharing.

### Design Principles
- **Deterministic**: Manifest derivable solely from profiling inputs + pruning/compression parameters.
- **Minimal**: Store references to artifacts, not duplicated large weight blobs.
- **Forward-Compatible**: Versioned schema enabling new optional fields (e.g., channel pruning) without breaking old manifests.
- **Integrity-Checked**: Hash every stored tensor file; include base model hash to prevent mismatches.

### Manifest Layers
1. Base Model Metadata (name, hash, framework, dtype)
2. Profiling Context (seeds, steps sampled, prompt embedding hash, scoring method, timestamp)
3. Pruning Decisions (attention heads removed, methodology, thresholds, fallback margin)
4. Compression Decisions (low-rank factors, quantization map, precision tiers)
5. Quality Evaluation (metrics vs baseline, sample size, thresholds passed)
6. Performance Metrics (latency, VRAM deltas, throughput)
7. Reuse & Generalization (paraphrase similarity stats, pass rate)
8. Artifact Index (paths + hashes for state_dict, low-rank factor files, quantization codebooks)

### Example (Abbreviated)
```json
{
    "schema_version": 1,
    "base_model": {"name": "sdxl_base_1.0", "sha256": "...", "dtype": "fp16"},
    "profiling": {"seeds": [111,222,333,444], "steps_profiled": 10, "total_steps": 50, "embedding_hash": "ab12...", "method": "activation_energy", "timestamp": 1723334400},
    "pruning": {"head_retention_ratio": 0.72, "removed": {"block_mid.cross": [0,5,11]}, "strategy": "percentile<30", "safety_margin_heads": 2},
    "compression": {"low_rank": [{"layer": "mid.attn.proj_out", "rank": 128, "energy_retained": 0.93}], "quantization": {"applied": false}},
    "quality": {"clip_delta_mean": 0.015, "lpips_mean": 0.028, "samples": 20, "thresholds_passed": true},
    "performance": {"peak_vram_full_mb": 18100, "peak_vram_fragment_mb": 13400, "latency_full_ms": 4800, "latency_fragment_ms": 3600},
    "reuse": {"paraphrase_pass_rate": 0.8, "similarity_metric": "cosine_clip_text"},
    "artifacts": {"fragment_state_path": "fragments/sdxl/h1_fragment.pt", "manifest_path": "fragments/sdxl/h1_fragment.json", "hashes": {"fragment_state_sha256": "..."}}
}
```

### Integrity Workflow
1. Compute base model hash on load (streaming SHA-256) → store.
2. After pruning/compression, save fragment state_dict; compute hash.
3. Serialize manifest JSON; compute manifest hash (embedded optionally).
4. On fragment load: verify (a) base model hash matches, (b) fragment hash matches; else fallback.

### Extension Points (Future Fields)
- `channel_pruning`: {"blocks": {"block_3": {"kept_channels": [..]}}}
- `precision_map`: {"layer_name": "fp16|int8|int4"}
- `dynamic_routing`: gating statistics for conditional execution.

---

## Metrics Pipeline & Reporting

### Objectives
Automate collection, aggregation, and comparison of quality, performance, and reuse metrics linked to hypotheses (H1–H5) with minimal developer overhead.

### Data Flow
```
Generation Run → Runtime Collectors (hooks) → In-Memory Buffers → Metrics Aggregator → JSON Report Writer → (Optional) Dashboard / CSV -> Cache Index
```

### Components
| Component                | Responsibility                              | Implementation Notes                                |
| ------------------------ | ------------------------------------------- | --------------------------------------------------- |
| Runtime Collectors       | Capture activations, attention, timings     | Hook wrappers; lightweight sampling (every N steps) |
| Metrics Aggregator       | Compute summary statistics                  | Pure Python + NumPy/PyTorch ops                     |
| Report Builder           | Assemble manifest-compatible metric blocks  | Reuses manifest schema fragments                    |
| Regression Comparator    | Compare new report vs baseline, flag deltas | Threshold config in YAML                            |
| Storage Layer            | Persist reports & manifests                 | `/reports/YYYYMMDD-HHMM/` timestamped dirs          |
| Visualization (optional) | Plot trends                                 | Matplotlib / Vega-lite JSON                         |

### Report File Layout
```
reports/
    2025-08-10_1500/
        profiling_raw.parquet
        head_importance.json
        fragment_manifest.json
        metrics_summary.json
        regression_diff.json
        plots/
            latency.png
            quality_tradeoff.png
```

### metrics_summary.json (Skeleton)
```json
{
    "run_id": "2025-08-10_1500",
    "base_model_hash": "...",
    "fragment_hash": "...",
    "hypotheses": {
        "H1": {"pruned_head_pct": 0.28, "clip_delta": 0.012, "status": "PASS"},
        "H2": {"vram_reduction_pct": 0.26, "lpips": 0.029, "status": "PASS"},
        "H3": {"reuse_pass_rate": 0.78, "status": "WARN"},
        "H4": {"batch_speedup": 1.55, "status": "WARN"},
        "H5": {"break_even_variants": 9, "status": "PASS"}
    },
    "quality": {"clip_mean_full": 0.312, "clip_mean_fragment": 0.308},
    "performance": {"latency_full_ms": 4800, "latency_fragment_ms": 3600},
    "memory": {"peak_full_mb": 18100, "peak_fragment_mb": 13400},
    "notes": "Reuse slightly below target; consider expanding paraphrase union."
}
```

### Sampling Strategies
- **Temporal Downsampling**: Profile only selected timesteps (e.g., geometric schedule early-density + late-detail steps).
- **Seed Diversity**: Use stratified seed pool; rotate a subset daily to avoid overfitting.
- **Head Importance Stability**: Track variance of importance ranking across seeds; flag heads with high rank instability for conservative retention.

### Regression Detection Logic
Pseudo-code:
```python
def evaluate_regression(current, baseline, thresholds):
        regressions = []
        for metric, limit in thresholds.items():
                if current[metric] - baseline[metric] > limit['max_delta']:
                        regressions.append({"metric": metric, "delta": current[metric]-baseline[metric]})
        return regressions
```

### Tooling Roadmap
| Phase | Capability                                                            |
| ----- | --------------------------------------------------------------------- |
| 1     | Manual JSON dump via simple collector                                 |
| 2     | Automated report packaging + diff vs previous best                    |
| 3     | CLI: `python tools/report.py --compare latest --gate thresholds.yaml` |
| 4     | Optional web dashboard / static HTML summary                          |

### Gating Threshold Configuration (thresholds.yaml Example)
```yaml
clip_delta_max: 0.02
lpips_max: 0.035
vram_reduction_min: 0.25
batch_speedup_min: 1.5
reuse_pass_rate_min: 0.75
break_even_max_variants: 10
```

### Continuous Integration Hook (Future)
- On PR: Run lightweight synthetic run (small model, reduced steps) → produce metrics_summary.json → compare to baseline snapshot; block merge if regression severity > configured.

### Failure Handling
- If metrics missing: mark run INVALID; do not update baselines.
- If partial success: store report but annotate manifest with `"verified": false`.

---

This technical architecture provides a comprehensive blueprint for implementing the dynamic diffusion model optimization system. The modular design allows for incremental development and testing, while the detailed error handling ensures robust operation in production environments.

Next steps would involve diving deeper into specific implementation details for any of these components, or beginning the actual ComfyUI node development.
