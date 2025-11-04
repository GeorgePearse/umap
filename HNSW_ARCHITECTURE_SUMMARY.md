# UMAP HNSW-RS (Rust-based HNSW) Backend Architecture Analysis

## Executive Summary

This document provides a comprehensive analysis of the HNSW-RS (Rust-based Hierarchical Navigable Small World) implementation in UMAP. While branded as "HNSW", the current implementation is actually a **brute-force k-NN search** with full distance metric support. The architecture is designed to eventually support true HNSW with hierarchical graph construction, but the current version provides correct results with simplified query logic.

**Key Finding**: This is a foundation implementation (0.1.0) that establishes the infrastructure for HNSW but uses brute-force search. It's positioned as "Phase 1" of a multi-phase optimization roadmap.

---

## 1. Rust Implementation Files (src/)

### 1.1 Core HNSW Index Implementation
**File**: `/home/georgepearse/umap/src/hnsw_index.rs` (382 lines)

#### Architecture Overview
- **Type**: `HnswIndex` - A PyO3-exposed Rust struct
- **Data Structure**: Simple vector-based storage (`Vec<Vec<f32>>`)
- **Search Method**: O(n) brute-force comparison for all points
- **Caching**: Neighbor graph caching for repeated queries
- **Thread Safety**: Uses `Arc<Mutex<T>>` for safe shared state (imported but unused in brute-force)

#### Core Methods

```rust
pub struct HnswIndex {
    data: Vec<Vec<f32>>,              // All indexed points
    n_neighbors: usize,                // Default k for neighbor queries
    metric: String,                    // Distance metric name
    is_angular: bool,                  // Cosine/correlation flag
    neighbor_graph_cache: Option<...>, // Cache for full neighbor graph
}
```

**Key Methods**:
1. `new()` - Initializes index with data, parameters, and validation
2. `query()` - Finds k-nearest neighbors for query points (O(n*k) per query)
3. `neighbor_graph()` - Computes k-nearest neighbors for all indexed points with caching
4. `prepare()` - No-op (for API compatibility)
5. `update()` - Extends index with new data points and invalidates cache
6. `compute_distance()` - Dispatches to appropriate distance metric

#### Current Limitations
- **Search Complexity**: O(n) per query (full scan) instead of O(log n) for true HNSW
- **No Hierarchical Structure**: No multi-level graph construction
- **No Dynamic Insertion**: Must rebuild from scratch for updates
- **HNSW Parameters Unused**: `m` and `ef_construction` parameters accepted for API compatibility but not used

#### Code Comment Analysis
```rust
/// Brute-force approximate nearest neighbor index
///
/// This is a simplified implementation using brute-force k-NN search.
/// It provides correct results but without the logarithmic scaling of HNSW.
/// This can be optimized to use HNSW or other algorithms later.
```

This confirms current implementation is intentionally simplified for correctness verification.

---

### 1.2 Distance Metrics Module
**File**: `/home/georgepearse/umap/src/metrics.rs` (171 lines)

#### Supported Metrics
1. **Euclidean (L2)**: `sqrt(sum((x_i - y_i)^2))`
2. **Manhattan (L1)**: `sum(|x_i - y_i|)`
3. **Cosine**: `1 - (dot(x,y) / (||x|| * ||y||))`
4. **Chebyshev (L∞)**: `max(|x_i - y_i|)`
5. **Minkowski**: Parameterized generalization
6. **Hamming**: Count of differing positions

#### Performance Characteristics
- All metrics use inline computations (no branching within loops)
- Cosine distance handles zero-vector edge case (returns 1.0)
- Chebyshev uses fold with max accumulation (efficient)
- All metrics fully tested with unit tests

#### Optimization Opportunities
- SIMD vectorization potential (using ndarray's BLAS integration)
- Batch distance computation for multiple query points
- Metric-specific optimizations (e.g., normalization preprocessing for cosine)

---

### 1.3 PyO3 Bridge Module
**File**: `/home/georgepearse/umap/src/lib.rs` (10 lines)

```rust
#[pymodule]
fn _hnsw_backend(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<hnsw_index::HnswIndex>()?;
    Ok(())
}
```

- Simple module that exposes `HnswIndex` class to Python
- Uses PyO3 0.21 with ABI3 support (Python 3.9+ compatibility)
- No custom methods or properties at module level

---

## 2. Rust Build Configuration

**File**: `/home/georgepearse/umap/Cargo.toml`

### Dependencies
```toml
[dependencies]
pyo3 = { version = "0.21", features = ["extension-module", "abi3-py39"] }
numpy = "0.21"           # NumPy array interop
ndarray = "0.15"         # N-dimensional arrays (imported, unused)
rayon = "1.8"            # Parallel iteration (imported, unused)
thiserror = "1.0"        # Error types
parking_lot = "0.12"     # Mutex/RwLock (imported, unused)
serde = "1.0"            # Serialization (imported, unused)
serde_json = "1.0"       # JSON (imported, unused)
```

### Optimization Profile
```toml
[profile.release]
opt-level = 3            # Maximum optimizations
lto = "fat"              # Full Link-Time Optimization
codegen-units = 1        # Single-unit compilation (slower build, better optimization)
strip = true             # Strip symbols for smaller binary
```

**Build Output**:
- Binary: `lib_hnsw_backend.so` (~1-2MB after stripping)
- Located: `target/release/lib_hnsw_backend.so`

---

## 3. Python/PyO3 Bindings

### 3.1 HNSW Wrapper Class
**File**: `/home/georgepearse/umap/umap/hnsw_wrapper.py` (278 lines)

#### Purpose
Provides a PyNNDescent-compatible API wrapping the Rust backend, allowing drop-in replacement of UMAP's nearest neighbor search.

#### Key Class: `HnswIndexWrapper`

**Constructor Parameters**:
- Converts PyNNDescent-style parameters to HNSW-compatible ones
- Computes optimal `m` and `ef_construction` from data characteristics
- Ensures float32 data type (required by Rust backend)

**Parameter Mapping**:
```python
n_trees → m (HNSW M parameter)
  Default: min(64, 5 + round((n_samples ** 0.5) / 20.0))
  Maps to: max(8, min(64, n_trees))  # Constrained 8-64 range

n_iters × max_candidates → ef_construction (HNSW parameter)
  Default: max(5, round(log2(n_samples)))
  Maps to: max(200, min(800, n_iters * max_candidates // 2))
  Range: 200-800 for quality control
```

**Query Parameter Mapping**:
```python
epsilon → ef (search parameter)
  Mapping: ef = max(k, int(k * (1.0 + epsilon * 30)))
  Capped at 500 to prevent excessive computation
  Semantics: epsilon controls search radius; ef controls candidate list
```

#### API Methods

| Method | Purpose | Parameters | Returns |
|--------|---------|------------|---------|
| `query()` | Find k neighbors for query points | query_data, k, epsilon | (indices, distances) |
| `neighbor_graph()` | Get k-neighbors for all indexed points | - | (indices, distances) or None |
| `prepare()` | Prepare index for queries | - | None (no-op) |
| `update()` | Add new data to index | X (new data) | None |

#### Properties
- `neighbor_graph` - Cached k-NN graph property (mutable)
- `_angular_trees` - Whether to use angular metrics
- `_raw_data` - Original data used for indexing

#### Current Behavior
```python
def neighbor_graph(self):
    if self._compressed:
        return None  # No graph for compressed indices

    if self._neighbor_graph_cache is None:
        indices, distances = self._index.neighbor_graph()
        self._neighbor_graph_cache = (indices, distances)

    return self._neighbor_graph_cache
```

Cache invalidation on `update()` prevents stale data.

---

## 4. Integration with UMAP

**File**: `/home/georgepearse/umap/umap/umap_.py`

### 4.1 Backend Selection Logic

```python
def _get_nn_backend(metric, sparse_data, use_hnsw=None):
    """Select between HNSW and PyNNDescent backends."""

    # Priority 1: Explicit user choice
    if use_hnsw is False:
        return NNDescent

    # Priority 2: Check HNSW availability
    if not HNSW_AVAILABLE:
        if use_hnsw is True:
            warn("HNSW backend not available")
        return NNDescent

    # Priority 3: Check metric support
    hnsw_metrics = {
        "euclidean", "l2",
        "manhattan", "l1", "taxicab",
        "cosine",
        "chebyshev", "linfinity",
        "hamming",
    }
    if metric not in hnsw_metrics:
        if use_hnsw is True:
            warn(f"Metric '{metric}' not supported, falling back to PyNNDescent")
        return NNDescent

    # Priority 4: Check data type
    if sparse_data:
        if use_hnsw is True:
            warn("Sparse data not supported, falling back to PyNNDescent")
        return NNDescent

    # Priority 5: Default to HNSW if compatible
    if use_hnsw is None:
        use_hnsw = True  # Default choice

    return HnswIndexWrapper if use_hnsw else NNDescent
```

### 4.2 HNSW vs PyNNDescent Comparison

| Aspect | HNSW | PyNNDescent |
|--------|------|-------------|
| **Backend** | Rust + PyO3 | Pure Python + Numba |
| **Metrics** | Euclidean, Manhattan, Cosine, Chebyshev, Hamming | All supported |
| **Sparse Data** | No | Yes |
| **Search Complexity** | O(n) current / O(log n) potential | O(n log n) to O(n) |
| **Memory** | Low (simple data structures) | Higher (tree structures) |
| **Deterministic** | Yes | No |
| **Dynamic Updates** | Yes (rebuild style) | Yes |
| **Dependencies** | Rust compiler needed | None (Python-only) |

### 4.3 Integration Points in `nearest_neighbors()`

```python
use_hnsw_backend = not use_pynndescent if use_pynndescent is not None else None
NNBackend = _get_nn_backend(metric, sparse_data, use_hnsw=use_hnsw_backend)

# Then uses NNBackend (either HnswIndexWrapper or NNDescent)
```

---

## 5. Current Performance Characteristics

### 5.1 Complexity Analysis

**Index Construction**: O(n × d) where n = samples, d = features
- Linear data copy from NumPy to Rust Vec
- No preprocessing or graph construction

**Query Operation**: O(n × d × k) per query
- Full scan: O(n × d) distance computations
- Sorting: O(n log k) per query
- Total: O(n × d) dominates

**Neighbor Graph**: O(n² × d)
- All-pairs distance computation
- Single pass, cached result
- Significant optimization opportunity via caching

### 5.2 Memory Usage
- **Data**: 4 bytes × n × d (float32)
- **Cache**: 8 bytes × n × k (indices) + 4 bytes × n × k (distances)
- **Total**: ~4(nd) + 12(nk) bytes

No significant memory overhead vs Python implementation.

### 5.3 Benchmarking

Current codebase includes:
- `/home/georgepearse/umap/doc/benchmarking.md` - Performance comparison documentation
- `/home/georgepearse/umap/umap/tests/test_chunked_parallel_spatial_metric.py` - Performance tests
- Metrics tracked: Runtime, scaling with dataset size, per-operation timing

Recent benchmarks show:
- UMAP significantly faster than t-SNE on large datasets
- Scaling better than MulticoreTSNE for 50k+ samples
- Comparable to PCA in runtime efficiency

---

## 6. Architecture Decisions and Rationale

### 6.1 Why Brute-Force for Phase 1?

1. **Correctness First**: Simple implementation for verification
2. **API Compatibility**: Establish PyNNDescent interface without graph complexity
3. **Foundation Building**: Infrastructure for later HNSW addition
4. **Parameter Flexibility**: Support all metrics easily
5. **Testing Ground**: Validate Rust/PyO3 integration before optimization

### 6.2 Rust Technology Choices

**PyO3 vs Ctypes/CFFI**:
- Type safety at compile time
- Automatic memory management
- Better Python integration

**NumPy/ndarray**:
- Direct array access without copies
- No overhead for data transfer

**Rayon**:
- Imported but unused (planned for parallel distance computation)
- Designed for data-parallel operations

---

## 7. Identified Optimization Opportunities

### 7.1 Short-Term (Phase 2)

1. **Parallel Query Computation**
   - Use rayon to parallelize neighbor finding per query
   - Complexity reduction: O(n) with pipelining across CPU cores
   - Implementation: `queries_vec.par_iter().map(|q| find_neighbors(q))`

2. **SIMD Vectorization**
   - Use ndarray's BLAS integration for batch distance computation
   - Leverage CPU vector instructions (SSE/AVX)
   - Potential: 4-8x speedup on distance computation

3. **Metric-Specific Optimizations**
   - Cosine: Precompute norms once
   - Euclidean: Use squared distances until final sqrt
   - Hamming: Bitwise operations for faster comparison

4. **Sorted Array Maintenance**
   - Avoid full sort per query
   - Use partial_sort or quickselect for top-k
   - Complexity: O(n) → O(n) average case, better cache usage

### 7.2 Medium-Term (Phase 3)

1. **True HNSW Implementation**
   - Multi-layer hierarchical graph
   - Logarithmic search: O(log n)
   - Construction time increase: manageable with caching

2. **Sparse Data Support**
   - Specialized distance computations for sparse vectors
   - Reduced memory for sparse indices
   - Extend metric support for sparse compatibility

3. **GPU Acceleration** (via CUDA/Metal)
   - Batch distance computation on GPU
   - 10-100x speedup potential for large batch queries
   - Maintain CPU fallback

### 7.3 Long-Term (Phase 4+)

1. **Pluggable ANN Backends**
   - Support multiple algorithms (FAISS, Annoy, HGG)
   - Runtime algorithm selection
   - Comprehensive benchmarking infrastructure

2. **Serialization**
   - Save/load indices without recomputation
   - serde infrastructure in place but unused
   - Enables large-scale embedding caching

3. **Distributed Computing**
   - Multi-machine index coordination
   - Federated search across partitions

---

## 8. Code Quality and Testing

### 8.1 Rust Code
**Test Coverage**:
- Distance metric unit tests: 100% metrics tested
- Basic functionality verified in commit
- No performance regression tests (planned)

**Code Style**:
- Follows Rust conventions
- Type-safe parameter handling
- Comprehensive docstrings (NumPy style)

### 8.2 Python Integration Tests
Located in: `/home/georgepearse/umap/umap/tests/`

Tests confirm:
- UMAP basic functionality with HNSW: PASSING
- Trustworthiness score: 0.978 (excellent quality)
- All core UMAP tests passing
- Parameter validation working

---

## 9. Dependencies and Build

### 9.1 Build System
- **Build Tool**: Maturin (Rust → Python extension)
- **Module Name**: `umap._hnsw_backend`
- **Python Target**: 3.9+ (via ABI3)
- **Build Profile**: Release with LTO, stripping

### 9.2 Runtime Dependencies
- NumPy >= 0.21 (array interop)
- PyO3 0.21 (Python bindings)
- No external C libraries (pure Rust)

### 9.3 Optional/Unused Dependencies
- **ndarray**: Imported but unused (planned for vectorization)
- **rayon**: Imported but unused (planned for parallelization)
- **parking_lot**: Imported but unused (synchronization primitive)
- **serde/serde_json**: Imported but unused (serialization)

These are included for Phase 2+ features.

---

## 10. Development Roadmap Context

From `/home/georgepearse/umap/doc/development_roadmap.md`:

### Current Position in UMAP Evolution

**Phase 1 (Current)**: Brute-force HNSW-RS
- Establishes Rust backend infrastructure
- Validates PyNNDescent API compatibility
- Provides correctness baseline

**Phase 2 (Planned)**:
- True HNSW implementation (hierarchical graphs)
- Parallel distance computation
- SIMD optimizations
- Sparse data support

**Phase 3 (Planned)**:
- Pluggable ANN backend system
- Support multiple algorithms: FAISS, Annoy, HGG, HNSW
- Algorithm selection guide
- Comprehensive benchmarking infrastructure

**Phase 4 (Long-term)**:
- GPU acceleration
- Serialization/persistence
- Distributed computing

---

## 11. Strategic Architecture Insights

### 11.1 The Brute-Force Strategy

The current "HNSW-RS" naming is somewhat misleading - it's better understood as:
- **Rust-based nearest neighbor backend** (general purpose)
- **HNSW-compatible interface** (future true HNSW)
- **Brute-force implementation** (current phase)

This staged approach:
1. **De-risks** the project by validating infrastructure first
2. **Enables incremental optimization** without breaking changes
3. **Provides fallback** if HNSW proves problematic
4. **Facilitates comparison** with other algorithms

### 11.2 Why This Matters

The development roadmap explicitly aims to "remove pynndescent dependency" by providing:
- Multiple ANN algorithm options (not just HNSW)
- Automatic algorithm selection based on data characteristics
- Comprehensive benchmarking against ann-benchmarks.com reference

Current HNSW-RS is **Phase 1 of this vision**, establishing the infrastructure for pluggable backends.

### 11.3 Key Design Principles

1. **API Compatibility**: PyNNDescent-compatible interface enables drop-in replacement
2. **Metric Flexibility**: Support all distance metrics, not just a few
3. **Correctness First**: Brute-force guarantees correct results
4. **Gradual Optimization**: Add complexity only when validated
5. **Future-Proof**: serde/serialization ready for distributed use

---

## 12. Summary of Findings

### What's Currently Implemented

✅ Rust-based nearest neighbor index with PyO3 bindings
✅ 6 distance metrics (Euclidean, Manhattan, Cosine, Chebyshev, Minkowski, Hamming)
✅ PyNNDescent-compatible wrapper API
✅ Integration with UMAP's nearest_neighbors function
✅ Automatic backend selection (HNSW for supported metrics, PyNNDescent fallback)
✅ Neighbor graph caching
✅ Dynamic index updates
✅ Comprehensive testing and validation

### What's Currently Missing (Phase 2+)

❌ True HNSW hierarchical graph structure
❌ Logarithmic search complexity (O(log n) vs current O(n))
❌ Parallel distance computation
❌ SIMD vectorization
❌ Sparse data support
❌ Index serialization
❌ GPU acceleration
❌ Multiple ANN algorithm support

### Optimization Potential

**Immediate** (2-3 weeks): Parallel queries with rayon → 4-8x speedup
**Short-term** (1-2 months): SIMD + metric optimizations → 2-3x additional
**Medium-term** (2-3 months): True HNSW → 10-100x for large datasets
**Long-term**: GPU/distributed → varies by use case

### Performance Profile

- **Current Search**: O(n × d) per query (comparable to brute-force)
- **Potential (HNSW)**: O(log n × d) per query (10-100x improvement)
- **Memory**: Minimal overhead, lower than PyNNDescent
- **Determinism**: Fully deterministic (no randomization)

---

## 13. File Location Summary

### Core Implementation
- Rust source: `/home/georgepearse/umap/src/`
  - `lib.rs` (10 lines) - PyO3 module definition
  - `hnsw_index.rs` (382 lines) - Core brute-force k-NN
  - `metrics.rs` (171 lines) - Distance metrics
- Python wrapper: `/home/georgepearse/umap/umap/hnsw_wrapper.py` (278 lines)
- Integration: `/home/georgepearse/umap/umap/umap_.py` (backend selection)

### Build Configuration
- Rust: `/home/georgepearse/umap/Cargo.toml`
- Python: `/home/georgepearse/umap/pyproject.toml`
- Maturin config: `[tool.maturin]` section

### Documentation
- Architecture: `/home/georgepearse/umap/doc/development_roadmap.md`
- Performance: `/home/georgepearse/umap/doc/benchmarking.md`

### Testing
- Tests: `/home/georgepearse/umap/umap/tests/` (all passing)
- Benchmarks: `test_chunked_parallel_spatial_metric.py`

---

## 14. Recommendations for Next Steps

### For Performance Optimization
1. **Profile current bottleneck** (distance computation vs sorting vs indexing)
2. **Implement parallel queries** first (lowest risk, high gain)
3. **Benchmark against ann-benchmarks.com** to set baselines
4. **Plan HNSW impl** after validating parallel approach

### For Feature Development
1. **Extend metric support** for pynndescent compatibility
2. **Implement sparse support** for complete API coverage
3. **Add serialization** for production use cases
4. **Build algorithm comparison** framework for future backends

### For Maintainability
1. **Document the Phase 1/2/3 progression** explicitly
2. **Add performance regression tests** to catch slowdowns
3. **Create optimization checklist** for prioritization
4. **Establish benchmarking CI/CD** pipeline
