# Hybrid Dimensionality Reduction Techniques: API Design & Implementation Plan

**Status:** Research & Planning Phase
**Target Release:** Phase 2 (Months 4-6)
**Priority:** High - Core architectural decision

---

## Executive Summary

UMAP and other dimensionality reduction methods each have distinct strengths and weaknesses. Rather than forcing users to choose a single algorithm, we propose a **hybrid techniques framework** that:

1. **Chains multiple algorithms** in sequence (e.g., PCA → UMAP → densMAP)
2. **Blends outputs** from multiple methods (weighted averaging of embeddings)
3. **Uses ensemble voting** for consensus embeddings
4. **Provides progressive refinement** (coarse approximation → fine-tuning)
5. **Enables algorithm composition** (high-level API for research)

This positions UMAP as a research platform where users can experiment with novel combinations and discover optimal pipelines for their datasets.

---

## 1. Core Concepts

### 1.1 Hybrid Technique Categories

**Sequential Composition**
```python
# PCA for noise reduction → UMAP for structure preservation
pipeline = DRPipeline([
    ("pca", PCA(n_components=50)),
    ("umap", UMAP(n_neighbors=15)),
])
```

**Parallel Blending**
```python
# Average outputs from t-SNE (local structure) and PaCMAP (global structure)
ensemble = EnsembleDR([
    ("tsne", TSNE(perplexity=30), weight=0.4),
    ("pacmap", PaCMAP(), weight=0.6),
])
```

**Progressive Refinement**
```python
# Start with fast approximation, refine with accurate method
progressive = ProgressiveDR(
    coarse=UMAP(n_neighbors=5),  # Fast initial
    fine=PHATE(),                 # Accurate refinement
    blend_steps=10                # Gradual transition
)
```

**Adaptive Routing**
```python
# Choose algorithm based on data characteristics
adaptive = AdaptiveDR({
    "small": UMAP(n_neighbors=15),
    "medium": PaCMAP(),
    "large": PHATE(),
    "very_large": LargeVis(),
})
```

### 1.2 Key Design Principles

1. **Composability**: Any DR method should work with any other
2. **Transparency**: Users understand what each component does
3. **Efficiency**: Minimize redundant computations
4. **Flexibility**: Support custom combinations
5. **Reproducibility**: Deterministic results with seeds
6. **Research-Friendly**: Easy to experiment with novel combinations

---

## 2. API Design

### 2.1 Base Architecture

```python
# Abstract base class for all DR methods
class DimensionalityReducer(ABC):
    """Base class for all dimensionality reduction techniques."""

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Learn the embedding."""
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data to embedding space."""
        pass

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform in one step."""
        pass

    def get_config(self) -> Dict:
        """Return serializable configuration."""
        pass

    @property
    def intrinsic_dim(self) -> int:
        """Estimated intrinsic dimensionality."""
        pass

    @property
    def metrics(self) -> Dict[str, float]:
        """Quality metrics (trustworthiness, continuity, etc)."""
        pass
```

### 2.2 Sequential Pipeline API

```python
class DRPipeline(DimensionalityReducer):
    """Chain multiple DR techniques in sequence."""

    def __init__(self,
                 steps: List[Tuple[str, DimensionalityReducer]],
                 n_jobs: int = -1,
                 verbose: bool = False):
        """
        Parameters
        ----------
        steps : List[Tuple[str, DimensionalityReducer]]
            Ordered list of (name, transformer) tuples
        n_jobs : int
            Parallel jobs (if supported by individual steps)
        verbose : bool
            Print progress information
        """
        self.steps = steps
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y=None):
        """Fit all transformers in sequence."""
        X_transformed = X.copy()
        for name, transformer in self.steps:
            if self.verbose:
                print(f"Fitting {name}...")
            X_transformed = transformer.fit_transform(X_transformed, y)
        self.fitted_steps_ = self.steps
        return self

    def transform(self, X):
        """Apply all transformations in sequence."""
        X_transformed = X.copy()
        for name, transformer in self.fitted_steps_:
            X_transformed = transformer.transform(X_transformed)
        return X_transformed

    def get_step(self, name: str) -> DimensionalityReducer:
        """Get a specific step by name."""
        for step_name, transformer in self.fitted_steps_:
            if step_name == name:
                return transformer
        raise ValueError(f"Step '{name}' not found")

    def visualize_pipeline(self) -> str:
        """Return ASCII representation of pipeline."""
        pass

# Example Usage
pipeline = DRPipeline([
    ("noise_reduction", PCA(n_components=100)),
    ("local_structure", UMAP(n_neighbors=15, min_dist=0.1)),
    ("density_aware", densMAP()),
])

pipeline.fit(X)
X_embedded = pipeline.transform(X_new)
```

### 2.3 Ensemble Blending API

```python
class EnsembleDR(DimensionalityReducer):
    """Blend outputs from multiple DR methods."""

    def __init__(self,
                 methods: List[Tuple[str, DimensionalityReducer, float]],
                 blend_mode: str = "weighted_average",
                 alignment: str = "procrustes",
                 n_jobs: int = -1):
        """
        Parameters
        ----------
        methods : List[Tuple[str, DimensionalityReducer, float]]
            List of (name, transformer, weight) tuples
        blend_mode : str
            "weighted_average" (default) or "voting"
        alignment : str
            "procrustes", "canonical", or "none"
        n_jobs : int
            Parallel jobs for fitting multiple methods
        """
        self.methods = methods
        self.blend_mode = blend_mode
        self.alignment = alignment
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit all methods in parallel."""
        self.fitted_methods_ = []

        # Parallel fitting
        results = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(self._fit_method)(name, method, X, y)
            for name, method, weight in self.methods
        )

        # Align embeddings to common space (Procrustes)
        if self.alignment != "none":
            results = self._align_embeddings(results)

        self.fitted_methods_ = results
        return self

    def transform(self, X):
        """Get embeddings from all methods and blend."""
        embeddings = [
            transformer.transform(X)
            for _, transformer in self.fitted_methods_
        ]

        # Blend based on weights
        weights = [w for _, _, w in self.methods]
        weights = np.array(weights) / sum(weights)  # Normalize

        if self.blend_mode == "weighted_average":
            return np.average(embeddings, axis=0, weights=weights)
        elif self.blend_mode == "voting":
            return self._ensemble_voting(embeddings)
        else:
            raise ValueError(f"Unknown blend_mode: {self.blend_mode}")

    def _align_embeddings(self, embeddings):
        """Align all embeddings to first one using Procrustes."""
        from scipy.spatial.distance import procrustes

        reference = embeddings[0]
        aligned = [reference]

        for emb in embeddings[1:]:
            _, (d, U) = procrustes(reference, emb)
            aligned_emb = emb @ U.T
            aligned.append(aligned_emb)

        return aligned

    def get_component_embeddings(self) -> Dict[str, np.ndarray]:
        """Return embeddings from each method separately."""
        return {
            name: transformer.embedding_
            for name, transformer in self.fitted_methods_
        }

# Example Usage
ensemble = EnsembleDR([
    ("umap", UMAP(n_neighbors=15), 0.3),         # Preserves some global structure
    ("tsne", TSNE(perplexity=30), 0.4),          # Excellent local structure
    ("pacmap", PaCMAP(), 0.3),                   # Good global structure
])

ensemble.fit(X)
X_blended = ensemble.transform(X_new)

# Inspect individual embeddings
individual = ensemble.get_component_embeddings()
```

### 2.4 Progressive Refinement API

```python
class ProgressiveDR(DimensionalityReducer):
    """Start with coarse approximation, progressively refine."""

    def __init__(self,
                 coarse: DimensionalityReducer,
                 fine: DimensionalityReducer,
                 blend_steps: int = 10,
                 blend_function: str = "linear"):
        """
        Parameters
        ----------
        coarse : DimensionalityReducer
            Fast initial method
        fine : DimensionalityReducer
            Accurate refinement method
        blend_steps : int
            Number of intermediate blends
        blend_function : str
            "linear", "exponential", or "sigmoid"
        """
        self.coarse = coarse
        self.fine = fine
        self.blend_steps = blend_steps
        self.blend_function = blend_function

    def fit(self, X, y=None):
        """Fit both methods."""
        # Fast initial fit
        self.coarse_embedding_ = self.coarse.fit_transform(X, y)

        # Accurate refinement fit
        self.fine_embedding_ = self.fine.fit_transform(X, y)

        # Compute progressive blends
        self.blended_sequence_ = self._create_blend_sequence()
        return self

    def transform(self, X):
        """Return final (fully refined) embedding."""
        # For new data, transform with fine method and adapt based on coarse
        coarse_new = self.coarse.transform(X)
        fine_new = self.fine.transform(X)

        # Blend based on learned alignment
        return self._blend_with_alignment(coarse_new, fine_new)

    def get_progressive_visualizations(self) -> List[np.ndarray]:
        """Return sequence of embeddings showing progression."""
        return self.blended_sequence_

    def _create_blend_sequence(self) -> List[np.ndarray]:
        """Create intermediate blends."""
        sequence = []
        for i in range(self.blend_steps + 1):
            t = i / self.blend_steps
            if self.blend_function == "linear":
                alpha = t
            elif self.blend_function == "exponential":
                alpha = t ** 2
            elif self.blend_function == "sigmoid":
                alpha = 1 / (1 + np.exp(-10 * (t - 0.5)))

            blended = (1 - alpha) * self.coarse_embedding_ + alpha * self.fine_embedding_
            sequence.append(blended)

        return sequence

# Example Usage
# Show progressive refinement: coarse PCA → fine PHATE
progressive = ProgressiveDR(
    coarse=PCA(n_components=50),
    fine=PHATE(),
    blend_steps=20
)

progressive.fit(X)

# Get intermediate visualizations
visualizations = progressive.get_progressive_visualizations()
# Use in animation or interactive visualization
```

### 2.5 Adaptive Router API

```python
class AdaptiveDR(DimensionalityReducer):
    """Automatically choose algorithm based on data characteristics."""

    def __init__(self,
                 method_map: Dict[str, DimensionalityReducer],
                 strategy: str = "size",
                 custom_selector: Optional[Callable] = None):
        """
        Parameters
        ----------
        method_map : Dict[str, DimensionalityReducer]
            Map of strategy outcomes to DR methods
            Keys: "small", "medium", "large", "very_large", or custom
        strategy : str
            "size" (default), "density", "intrinsic_dim", "custom"
        custom_selector : Optional[Callable]
            Custom selection function if strategy="custom"
        """
        self.method_map = method_map
        self.strategy = strategy
        self.custom_selector = custom_selector

    def fit(self, X, y=None):
        """Select and fit appropriate method."""
        category = self._categorize_data(X, y)
        self.selected_method_ = self.method_map[category]
        self.selected_category_ = category

        if self.verbose:
            print(f"Selected method for '{category}' category: {self.selected_method_}")

        self.selected_method_.fit(X, y)
        return self

    def transform(self, X):
        """Transform using selected method."""
        return self.selected_method_.transform(X)

    def _categorize_data(self, X, y=None) -> str:
        """Categorize data based on strategy."""
        if self.strategy == "size":
            n_samples = X.shape[0]
            if n_samples < 1000:
                return "small"
            elif n_samples < 10000:
                return "medium"
            elif n_samples < 100000:
                return "large"
            else:
                return "very_large"

        elif self.strategy == "density":
            # Compute local density
            return self._categorize_by_density(X)

        elif self.strategy == "intrinsic_dim":
            # Estimate intrinsic dimensionality
            return self._categorize_by_intrinsic_dim(X)

        elif self.strategy == "custom":
            return self.custom_selector(X, y)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

# Example Usage
adaptive = AdaptiveDR({
    "small": UMAP(n_neighbors=15),  # Can afford more computation
    "medium": PaCMAP(),             # Balanced approach
    "large": PHATE(),               # Preserves density well
    "very_large": LargeVis(),       # Highly scalable
})

adaptive.fit(X)
```

---

## 3. Advanced Features

### 3.1 Serialization & Configuration

```python
class DRPipeline:
    def to_config(self) -> Dict:
        """Export pipeline as JSON/YAML config."""
        return {
            "type": "pipeline",
            "steps": [
                {
                    "name": name,
                    "transformer": transformer.to_config()
                }
                for name, transformer in self.steps
            ]
        }

    @classmethod
    def from_config(cls, config: Dict) -> "DRPipeline":
        """Load pipeline from config."""
        steps = [
            (step["name"], DimensionalityReducer.from_config(step["transformer"]))
            for step in config["steps"]
        ]
        return cls(steps)

# Save and load pipelines
config = pipeline.to_config()
with open("my_pipeline.yaml", "w") as f:
    yaml.dump(config, f)

# Later, reload
with open("my_pipeline.yaml") as f:
    config = yaml.load(f)
pipeline = DRPipeline.from_config(config)
```

### 3.2 Visualization of Pipelines

```python
def visualize_pipeline(pipeline: DRPipeline) -> str:
    """ASCII visualization of pipeline."""
    lines = ["Dimensionality Reduction Pipeline", "=" * 40]

    for i, (name, transformer) in enumerate(pipeline.steps):
        if i > 0:
            lines.append("         ↓")
        lines.append(f"  [{i+1}] {name}: {transformer.__class__.__name__}")
        if hasattr(transformer, 'n_components'):
            lines.append(f"        → {transformer.n_components}d")

    return "\n".join(lines)

# Example output:
# Dimensionality Reduction Pipeline
# ========================================
#   [1] noise_reduction: PCA
#        → 100d
#           ↓
#   [2] local_structure: UMAP
#   [3] density_aware: densMAP
```

### 3.3 Metrics Across Pipeline Stages

```python
class DRPipeline:
    def get_stage_metrics(self) -> List[Dict[str, float]]:
        """Get quality metrics at each stage."""
        metrics_by_stage = []

        for i, (name, _) in enumerate(self.fitted_steps_):
            if hasattr(self.fitted_steps_[i][1], 'metrics'):
                metrics_by_stage.append({
                    "stage": i,
                    "name": name,
                    "metrics": self.fitted_steps_[i][1].metrics
                })

        return metrics_by_stage
```

---

## 4. Research Platform Benefits

### 4.1 Enables Novel Combinations

Researchers can easily experiment with:
- New algorithm pairs
- Different blending strategies
- Progressive refinement sequences
- Adaptive routing heuristics

### 4.2 Benchmarking Framework

```python
class HybridBenchmark:
    """Benchmark hybrid technique combinations."""

    def __init__(self, X, y=None, ground_truth_labels=None):
        self.X = X
        self.y = y
        self.ground_truth_labels = ground_truth_labels

    def compare_hybrids(self,
                       hybrids: List[Tuple[str, DimensionalityReducer]],
                       metrics: List[str] = None) -> pd.DataFrame:
        """Compare multiple hybrid techniques."""
        if metrics is None:
            metrics = ["trustworthiness", "continuity", "time"]

        results = []
        for name, hybrid in hybrids:
            start = time.time()
            hybrid.fit(self.X, self.y)
            elapsed = time.time() - start

            metrics_dict = {
                "name": name,
                "time": elapsed,
            }

            if hasattr(hybrid, 'metrics'):
                metrics_dict.update(hybrid.metrics)

            results.append(metrics_dict)

        return pd.DataFrame(results)
```

---

## 5. Implementation Timeline

### Phase 2A (Week 1-2): Foundation
- [ ] Abstract `DimensionalityReducer` base class
- [ ] Refactor existing UMAP, PHATE, etc. to inherit from base
- [ ] Create `DRPipeline` for sequential composition
- [ ] Write comprehensive tests

### Phase 2B (Week 3): Ensemble
- [ ] Implement `EnsembleDR` with Procrustes alignment
- [ ] Add weighted blending
- [ ] Add ensemble voting
- [ ] Benchmarks comparing ensemble vs single methods

### Phase 2C (Week 4-5): Advanced Features
- [ ] Implement `ProgressiveDR`
- [ ] Implement `AdaptiveDR`
- [ ] Add serialization/deserialization
- [ ] Add visualization utilities

### Phase 2D (Week 6): Polish & Documentation
- [ ] Comprehensive examples
- [ ] Performance optimization
- [ ] Community feedback
- [ ] Paper/blog post on approach

---

## 6. Configuration Examples

### Example 1: Noise Reduction Pipeline

```python
# For noisy high-dimensional data
pipeline = DRPipeline([
    ("pca", PCA(n_components=100)),           # Remove noise
    ("robust_umap", UMAP(metric='correlation')),  # Robust to outliers
    ("density_map", densMAP()),               # Preserve local density
])
```

### Example 2: Local+Global Ensemble

```python
# Best of both worlds: local structure (t-SNE) + global structure (PaCMAP)
ensemble = EnsembleDR([
    ("local", TSNE(perplexity=30), 0.4),
    ("global", PaCMAP(), 0.6),
])
```

### Example 3: Progressive from Fast to Accurate

```python
# Start with fast PCA approximation, refine with accurate PHATE
progressive = ProgressiveDR(
    coarse=PCA(n_components=50),
    fine=PHATE(),
    blend_steps=20
)
```

---

## 7. Success Metrics

- **Flexibility**: Support any combination of DR methods
- **Ease of Use**: Simple API for common cases, powerful for research
- **Performance**: Minimal overhead vs single method
- **Reproducibility**: Exact same results given same seed
- **Documentation**: Clear examples for common use cases
- **Community**: Enable shared hybrid technique configurations

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| API complexity | Provide high-level `Pipeline()` and `Ensemble()` helpers |
| Performance overhead | Profile and optimize critical paths |
| Alignment issues | Test Procrustes and other alignment methods thoroughly |
| Parameter explosion | Sensible defaults for all hybrid techniques |
| Maintenance burden | Automated testing of all combinations |

---

## 9. Future Extensions

1. **AutoML for DR**: Learn optimal pipelines from data
2. **Gradient-Based Blending**: Learn optimal weights (not fixed)
3. **Adaptive Blending**: Change weights during refinement
4. **Hardware-Specific Routing**: Route to GPU/CPU based on available resources
5. **Interactive Refinement**: UI for progressive blending control

---

## 10. References

- Scikit-learn Pipeline API
- PyTorch nn.Sequential for inspiration
- Procrustes alignment techniques
- Ensemble learning literature
