# Sparse Vector Capability: Design & Implementation Plan

**Status:** Research & Planning Phase
**Target Release:** Phase 2 (Months 4-6)
**Priority:** High - Enables new use cases

---

## Executive Summary

Many real-world datasets are naturally sparse:
- **Text**: Bag-of-words, TF-IDF vectors (98-99% sparse)
- **Genomics**: Single-cell RNA-seq (95-98% sparse)
- **Networks**: Adjacency matrices, social graphs (99%+ sparse)
- **Sensor Data**: Time-series with many zeros

**Current Problem:** Existing UMAP must densify data, wasting memory and computation on zeros.

**Proposed Solution:** Native sparse vector support throughout the UMAP pipeline:
1. **Sparse Input Handling**: Accept scipy.sparse, PyTorch sparse, etc.
2. **Sparse k-NN Construction**: Build graphs without densification
3. **Sparse Metric Computation**: Calculate distances in sparse space
4. **Sparse HNSW Integration**: Optimize HNSW for sparse vectors
5. **Compression & Caching**: Reduce memory footprint further

This positions UMAP as the go-to method for sparse, high-dimensional data in biology, NLP, and network analysis.

---

## 1. Data Format Support

### 1.1 Input Formats

```python
# Current: Must densify
X = dense_array  # (n_samples, n_features)

# Proposed: Native sparse support
X = scipy.sparse.csr_matrix(dense_data)      # CSR format (default)
X = scipy.sparse.csc_matrix(dense_data)      # CSC format
X = scipy.sparse.coo_matrix(dense_data)      # COO format
X = torch.sparse_csr_tensor(...)             # PyTorch sparse
X = tf.SparseTensor(...)                     # TensorFlow sparse
X = cupy.sparse.csr_matrix(...)              # GPU sparse (CuPy)

# Lazy loading for very large datasets
X = LazySparseTensor("path/to/data.h5", format='sparse')

# Mixed format support
X_dense = np.array([...])
X_sparse = scipy.sparse.csr_matrix([...])
# UMAP should handle both transparently
```

### 1.2 Format Detection

```python
class SparseFormatDetector:
    """Automatically detect and validate sparse format."""

    @staticmethod
    def is_sparse(X):
        """Check if X is sparse."""
        return scipy.sparse.issparse(X) or isinstance(X, (torch.sparse_csr_tensor, tf.SparseTensor))

    @staticmethod
    def to_canonical(X, target_format='csr'):
        """Convert to canonical format for processing."""
        if isinstance(X, scipy.sparse.spmatrix):
            if target_format == 'csr':
                return X.tocsr()
            elif target_format == 'csc':
                return X.tocsc()
        elif isinstance(X, torch.sparse_csr_tensor):
            return X
        elif isinstance(X, tf.SparseTensor):
            return X
        else:
            # Dense array
            return scipy.sparse.csr_matrix(X)

    @staticmethod
    def get_sparsity(X):
        """Calculate sparsity percentage."""
        if scipy.sparse.issparse(X):
            nnz = X.nnz
            total = X.shape[0] * X.shape[1]
            return 1 - (nnz / total)
        else:
            return 0  # Dense

    @staticmethod
    def estimate_memory_savings(X):
        """Estimate memory saved by using sparse format."""
        if scipy.sparse.issparse(X):
            dense_mb = X.shape[0] * X.shape[1] * 8 / (1024**2)  # 8 bytes per float64
            sparse_mb = (X.nnz * 2 * 8 + X.shape[0] * 8) / (1024**2)  # Approximate
            return dense_mb - sparse_mb
        return 0
```

---

## 2. Core Pipeline Integration

### 2.1 Sparse k-NN Graph Construction

```python
class SparseKNNGraph:
    """Build k-NN graphs efficiently for sparse vectors."""

    def __init__(self, metric='euclidean', n_neighbors=15, sparse_mode='auto'):
        """
        Parameters
        ----------
        metric : str
            "euclidean", "cosine", "manhattan", "jaccard" (for binary sparse)
        n_neighbors : int
            Number of neighbors
        sparse_mode : str
            "auto" (best for data), "exact", "hnsw", "lsh"
        """
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.sparse_mode = sparse_mode

    def build(self, X):
        """Build k-NN graph on sparse data."""
        if self.sparse_mode == 'auto':
            mode = self._select_best_mode(X)
        else:
            mode = self.sparse_mode

        if mode == 'exact':
            return self._build_exact_sparse(X)
        elif mode == 'hnsw':
            return self._build_hnsw_sparse(X)
        elif mode == 'lsh':
            return self._build_lsh_sparse(X)

    def _build_exact_sparse(self, X):
        """O(n²) exact sparse k-NN."""
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            algorithm='brute'  # Required for sparse
        ).fit(X)

        distances, indices = nbrs.kneighbors(X)
        return self._to_graph(distances, indices, X.shape[0])

    def _build_hnsw_sparse(self, X):
        """Optimized HNSW for sparse vectors."""
        from umap._hnsw_sparse import HNSWSparseIndex

        index = HNSWSparseIndex(metric=self.metric)
        index.build(X, n_neighbors=self.n_neighbors)
        return index.to_graph()

    def _build_lsh_sparse(self, X):
        """Locality-Sensitive Hashing for very high-dimensional sparse."""
        from datasketch import MinHashLSH

        lsh = MinHashLSH(num_perm=128)
        # LSH construction and querying
        return self._to_graph_from_lsh(lsh, X)

    def _select_best_mode(self, X):
        """Automatically select best algorithm."""
        n_samples = X.shape[0]
        sparsity = 1 - (X.nnz / (X.shape[0] * X.shape[1]))

        # Decision tree
        if n_samples < 1000:
            return 'exact'  # Exact is fast enough
        elif sparsity > 0.99:
            return 'lsh'    # Ultra-sparse: use LSH
        elif n_samples > 100000:
            return 'hnsw'   # Large: use HNSW
        else:
            return 'hnsw'   # Default to HNSW

class SparseUMAP(UMAP):
    """UMAP with native sparse vector support."""

    def __init__(self,
                 n_components=2,
                 n_neighbors=15,
                 min_dist=0.1,
                 metric='euclidean',
                 sparse_mode='auto',
                 **kwargs):
        """
        Parameters
        ----------
        sparse_mode : str
            "auto", "exact", "hnsw", "lsh", or "none" (force densification)
        """
        super().__init__(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            **kwargs
        )
        self.sparse_mode = sparse_mode
        self.is_sparse_ = False

    def fit(self, X, y=None):
        """Fit UMAP with automatic sparse handling."""
        # Detect sparsity
        self.is_sparse_ = SparseFormatDetector.is_sparse(X)

        if self.is_sparse_ and self.sparse_mode != 'none':
            return self._fit_sparse(X, y)
        else:
            return self._fit_dense(X, y)

    def _fit_sparse(self, X, y=None):
        """Fit with sparse optimization."""
        # Build sparse k-NN graph
        knn_graph = SparseKNNGraph(
            metric=self.metric,
            n_neighbors=self.n_neighbors,
            sparse_mode=self.sparse_mode
        ).build(X)

        # Compute sparse fuzzy simplicial set
        self.graph_ = self._compute_sparse_fss(X, knn_graph)

        # Layout optimization (can work with sparse graph)
        self.embedding_ = self._layout_from_sparse_graph()

        return self

    def transform(self, X):
        """Transform sparse data."""
        if self.is_sparse_:
            return self._transform_sparse(X)
        else:
            return self._transform_dense(X)

    def _transform_sparse(self, X):
        """Efficient transform for sparse vectors."""
        # Find nearest neighbors in original space
        # Project to embedding space
        # Refine with optimization
        pass
```

### 2.2 Sparse Distance Metrics

```python
class SparseMetrics:
    """Optimized distance calculations for sparse vectors."""

    @staticmethod
    def sparse_euclidean(X, Y, squared=False):
        """
        Compute Euclidean distance between sparse matrices.

        For sparse X, Y:
        ||x - y||² = ||x||² + ||y||² - 2⟨x, y⟩

        This avoids explicit densification.
        """
        # Compute norms efficiently
        X_norm = np.sqrt(np.array(X.power(2).sum(axis=1)).ravel())
        Y_norm = np.sqrt(np.array(Y.power(2).sum(axis=1)).ravel())

        # Compute dot products
        dot_product = X @ Y.T

        # Distance matrix: ||x - y||²
        distance_squared = (
            X_norm[:, np.newaxis]**2 +
            Y_norm[np.newaxis, :]**2 -
            2 * dot_product
        )

        if squared:
            return distance_squared
        else:
            return np.sqrt(np.maximum(distance_squared, 0))

    @staticmethod
    def sparse_cosine(X, Y):
        """
        Compute cosine distance between sparse matrices.

        For sparse vectors, this is the most efficient.
        """
        # Normalize rows
        X_norm = X / np.sqrt(np.array(X.power(2).sum(axis=1)).ravel())[:, np.newaxis]
        Y_norm = Y / np.sqrt(np.array(Y.power(2).sum(axis=1)).ravel())[:, np.newaxis]

        # Cosine distance = 1 - cosine_similarity
        similarity = (X_norm @ Y_norm.T).toarray()
        return 1 - similarity

    @staticmethod
    def sparse_manhattan(X, Y):
        """Compute Manhattan distance between sparse matrices."""
        n, m = X.shape[0], Y.shape[0]
        distances = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                distances[i, j] = np.sum(np.abs(X[i].data - Y[j].data))

        return distances

    @staticmethod
    def sparse_jaccard(X, Y):
        """
        Jaccard distance for sparse binary vectors.

        J(A, B) = |A ∩ B| / |A ∪ B|
        """
        # Convert to binary if needed
        X_bool = (X != 0).astype(bool)
        Y_bool = (Y != 0).astype(bool)

        # Compute Jaccard
        intersection = X_bool @ Y_bool.T
        union = (
            np.array(X_bool.sum(axis=1)).ravel()[:, np.newaxis] +
            np.array(Y_bool.sum(axis=1)).ravel()[np.newaxis, :] -
            intersection
        )

        # Avoid division by zero
        jaccard = np.zeros_like(union, dtype=float)
        mask = union > 0
        jaccard[mask] = 1 - (intersection[mask] / union[mask])

        return jaccard
```

### 2.3 Sparse Fuzzy Simplicial Set

```python
class SparseFuzzySimplicialSet:
    """Construct fuzzy simplicial set from sparse k-NN graph."""

    def __init__(self, metric='euclidean', local_connectivity=1.0):
        self.metric = metric
        self.local_connectivity = local_connectivity

    def compute(self, X, knn_indices, knn_distances):
        """
        Compute sparse fuzzy simplicial set.

        Returns sparse matrix where entries are membership strengths.
        """
        n_samples = X.shape[0]

        # Initialize as COO for efficient construction
        rows, cols, data = [], [], []

        for i in range(n_samples):
            # Get k-NN for point i
            neighbors = knn_indices[i]
            distances = knn_distances[i]

            # Compute membership strengths (smooth decay)
            # Higher strength for closer neighbors
            strengths = np.exp(-distances)

            for neighbor, strength in zip(neighbors, strengths):
                rows.append(i)
                cols.append(neighbor)
                data.append(strength)

        # Create sparse matrix
        simplicial_set = scipy.sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(n_samples, n_samples)
        ).tocsr()

        return simplicial_set
```

---

## 3. Memory Optimization

### 3.1 Sparse Data Structures

```python
class SparseDataStore:
    """Efficient storage for sparse vectors and distances."""

    def __init__(self, X, format='csr'):
        self.X = SparseFormatDetector.to_canonical(X, format)
        self.format = format
        self.shape = X.shape
        self.nnz = self.X.nnz
        self.sparsity = 1 - (self.nnz / (self.shape[0] * self.shape[1]))

    def get_row(self, i):
        """Get single row efficiently."""
        return self.X.getrow(i)

    def get_column(self, j):
        """Get single column efficiently."""
        return self.X.getcol(j)

    def slice_rows(self, indices):
        """Get multiple rows efficiently."""
        return self.X[indices]

    def estimate_memory_mb(self):
        """Estimate memory usage."""
        # Sparse matrix storage: data + indices + pointers
        data_mb = self.nnz * 8 / (1024**2)  # float64
        indices_mb = self.nnz * 4 / (1024**2)  # int32
        pointers_mb = (self.shape[0] + 1) * 4 / (1024**2)  # int32
        return data_mb + indices_mb + pointers_mb

    def compare_to_dense(self):
        """Compare memory vs dense storage."""
        dense_mb = self.shape[0] * self.shape[1] * 8 / (1024**2)
        sparse_mb = self.estimate_memory_mb()
        ratio = sparse_mb / dense_mb if dense_mb > 0 else 1
        return {
            "dense_mb": dense_mb,
            "sparse_mb": sparse_mb,
            "ratio": ratio,
            "savings_mb": dense_mb - sparse_mb,
            "sparsity": self.sparsity
        }
```

### 3.2 Caching Strategy

```python
class SparseDistanceCache:
    """Smart caching for sparse distance computations."""

    def __init__(self, X, max_cache_mb=1000):
        self.X = X
        self.max_cache_mb = max_cache_mb
        self.cache = {}
        self.access_count = {}

    def get_distance(self, i, j, metric='euclidean'):
        """Get distance with caching."""
        key = (min(i, j), max(i, j), metric)

        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]

        # Compute and cache
        dist = self._compute_distance(i, j, metric)

        # Check cache size
        if self._estimate_cache_size() > self.max_cache_mb:
            self._evict_least_used()

        self.cache[key] = dist
        self.access_count[key] = 1
        return dist

    def _evict_least_used(self):
        """Remove least frequently accessed entries."""
        sorted_keys = sorted(
            self.access_count.items(),
            key=lambda x: x[1]
        )

        # Remove bottom 10%
        for key, _ in sorted_keys[:len(sorted_keys) // 10]:
            del self.cache[key]
            del self.access_count[key]

    def _estimate_cache_size(self):
        """Estimate cache size in MB."""
        return len(self.cache) * 8 / (1024**2)
```

---

## 4. Sparse Metric Support

### 4.1 Recommended Metrics by Data Type

| Data Type | Recommended Metric | Why |
|-----------|-------------------|-----|
| **Text (TF-IDF)** | cosine | Natural for normalized vectors |
| **RNA-seq** | manhattan, euclidean | Counts are non-negative |
| **Binary** | jaccard, hamming | Natural for presence/absence |
| **Network** | cosine | Adjacency matrix similarity |
| **General** | euclidean | Works for most sparse data |

```python
class SparseMetricSelector:
    """Recommend best metric for sparse data."""

    @staticmethod
    def suggest_metric(X, data_type='unknown'):
        """Suggest best metric based on data characteristics."""
        if data_type == 'text':
            return 'cosine'
        elif data_type == 'binary':
            return 'jaccard'
        elif data_type == 'counts':
            return 'manhattan'
        elif data_type == 'network':
            return 'cosine'
        else:
            # Auto-detect
            has_negative = (X.data < 0).any()
            is_binary = np.all(np.isin(X.data, [0, 1]))

            if is_binary:
                return 'jaccard'
            elif has_negative:
                return 'euclidean'
            else:
                return 'cosine'
```

---

## 5. Benchmarking Sparse vs Dense

```python
class SparseBenchmark:
    """Compare sparse vs dense UMAP performance."""

    def __init__(self, X_dense, sparsity_levels=[0.5, 0.9, 0.99]):
        self.X_dense = X_dense
        self.sparsity_levels = sparsity_levels

    def create_sparse_versions(self):
        """Create sparse versions at different sparsity levels."""
        sparse_versions = []

        for sparsity in self.sparsity_levels:
            n_keep = int(self.X_dense.size * (1 - sparsity))
            mask = np.random.choice(
                self.X_dense.size,
                size=n_keep,
                replace=False
            )
            X_sparse = self.X_dense.copy()
            X_sparse.flat[~np.isin(np.arange(self.X_dense.size), mask)] = 0

            sparse_versions.append({
                "sparsity": sparsity,
                "X": scipy.sparse.csr_matrix(X_sparse)
            })

        return sparse_versions

    def benchmark_all(self):
        """Benchmark both dense and sparse implementations."""
        results = []

        # Dense baseline
        start = time.time()
        umap_dense = UMAP()
        umap_dense.fit(self.X_dense)
        dense_time = time.time() - start

        results.append({
            "type": "dense",
            "time": dense_time,
            "memory_mb": self.X_dense.nbytes / (1024**2),
            "sparsity": 0
        })

        # Sparse versions
        for version in self.create_sparse_versions():
            X_sparse = version["X"]
            sparsity = version["sparsity"]

            start = time.time()
            umap_sparse = SparseUMAP()
            umap_sparse.fit(X_sparse)
            sparse_time = time.time() - start

            results.append({
                "type": "sparse",
                "time": sparse_time,
                "memory_mb": X_sparse.data.nbytes / (1024**2),
                "sparsity": sparsity,
                "speedup": dense_time / sparse_time
            })

        return pd.DataFrame(results)
```

---

## 6. Use Cases

### 6.1 Single-Cell RNA-seq

```python
# scRNA-seq data is 95-98% sparse
import scanpy as sc

adata = sc.read_h5ad("data.h5ad")
X = adata.X  # Already scipy.sparse.csr_matrix

# Direct sparse support - no densification!
reducer = SparseUMAP(n_neighbors=15, metric='manhattan')
X_emb = reducer.fit_transform(X)

adata.obsm['X_umap'] = X_emb
```

### 6.2 NLP / Document Embedding

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = ["doc1", "doc2", ...]
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(documents)  # scipy.sparse.csr_matrix (99% sparse)

# Native sparse support
reducer = SparseUMAP(metric='cosine')
X_emb = reducer.fit_transform(X)
```

### 6.3 Network Analysis

```python
import networkx as nx

G = nx.read_graphml("network.graphml")
adj = nx.to_scipy_sparse_array(G)  # scipy.sparse matrix

# Sparse adjacency matrix
reducer = SparseUMAP(metric='cosine')
X_emb = reducer.fit_transform(adj)
```

---

## 7. Implementation Phases

### Phase 2A (Week 1-2): Foundation
- [ ] `SparseFormatDetector` - auto-detect formats
- [ ] Sparse k-NN graph construction
- [ ] Sparse distance metrics (cosine, euclidean, manhattan)
- [ ] Tests for sparse format support

### Phase 2B (Week 3): Integration
- [ ] `SparseUMAP` class with sparse mode
- [ ] Sparse fuzzy simplicial set
- [ ] Layout algorithm with sparse graph
- [ ] Benchmarks vs dense

### Phase 2C (Week 4): Optimization
- [ ] HNSW sparse variant
- [ ] Distance caching
- [ ] Memory profiling
- [ ] Performance optimization

### Phase 2D (Week 5): Polish
- [ ] Documentation & examples
- [ ] Use-case tutorials (scRNA-seq, NLP, networks)
- [ ] Sparse metric recommendations
- [ ] Community feedback

---

## 8. API Examples

```python
# Basic sparse UMAP
X_sparse = scipy.sparse.load_npz("data.npz")
reducer = SparseUMAP()
X_emb = reducer.fit_transform(X_sparse)

# Custom metric
reducer = SparseUMAP(metric='cosine', sparse_mode='hnsw')

# Mixed input
X_dense = np.array([...])
X_sparse = scipy.sparse.csr_matrix([...])

pipeline = DRPipeline([
    ("dense_input", DensePreprocessor()),
    ("sparse_reducer", SparseUMAP(metric='cosine')),
])

# Benchmark sparse efficiency
bench = SparseBenchmark(X)
results = bench.benchmark_all()
print(results)
```

---

## 9. Success Metrics

- **Memory Efficiency**: 10-100x memory savings on sparse data
- **Speed**: 2-10x faster on sparse data vs densified
- **Compatibility**: Works with scipy.sparse, PyTorch sparse, TensorFlow sparse
- **Accuracy**: Identical results to dense implementation
- **Usability**: Simple API, transparent sparse handling
- **Documentation**: Clear examples for common sparse use cases

---

## 10. References

- SciPy sparse matrix documentation
- PyTorch sparse tensor documentation
- Sparse distance metrics (cosine, Jaccard, Hamming)
- k-NN for sparse data
- LSH and HNSW with sparse metrics
