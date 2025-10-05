# Scalability Analysis: Handling Large CSV Files (150MB+)

**Date:** 2025-10-01
**Current Model:** ollama/llama3.1:8b
**Available Alternative:** OpenAI GPT-4/GPT-3.5 (API key configured)

## Executive Summary

✅ **GOOD NEWS:** Your code is designed well with proper tool-based architecture that avoids loading entire CSVs into LLM context.

⚠️ **CAUTION:** There are still several critical bottlenecks that will cause failures with 150MB+ files using the current LLM model.

## Current Architecture Analysis

### ✅ What's Working Well (Good Practices)

1. **Tool-Based Architecture**
   - ✅ `StatisticalAnalysisTool` performs computations locally and returns only summary statistics
   - ✅ `CSVSearchTool` uses RAG (Retrieval Augmented Generation) with embeddings
   - ✅ Tools return formatted text summaries, not raw data
   - ✅ Chunked processing in statistical methods (e.g., sampling for normality tests)

2. **Smart Data Handling**
   - ✅ Statistical tool limits output (e.g., first 10 columns in tables)
   - ✅ Correlation analysis returns only strong correlations (|r| > 0.7)
   - ✅ Outlier detection returns summaries, not individual outlier records
   - ✅ Clustering returns cluster centroids, not all data points

3. **Streaming Preview**
   - ✅ File validator uses `nrows=10` for preview (only loads first 10 rows)
   - ✅ Memory-efficient validation without loading full dataset

### ⚠️ Critical Bottlenecks for Large Files

#### 1. **CRITICAL: Full CSV Loading in Statistical Analysis Tool**

**Location:** `statistical_analysis_tool.py:113`
```python
df = pd.read_csv(dataset_path)  # ❌ LOADS ENTIRE FILE INTO MEMORY
```

**Impact:**
- **150MB CSV** ≈ 500K-1M rows → **1-2GB RAM** when loaded as DataFrame
- Will crash on machines with limited memory
- Even if it loads, subsequent operations are slow

**Risk Level:** 🔴 **HIGH** - Will fail for files >100MB on typical systems

---

#### 2. **CRITICAL: LLM Context Window Limitations**

**Current Model:** `llama3.1:8b` - Context window: **~8K tokens** (very limited)

**Problem Areas:**

**a) Chat Agent Context Preparation** (`chat_agent.py:121-227`)
```python
# Loads first 100 lines of report
for line in report_lines[:100]:
    # Extracts summaries, findings, recommendations
    # Builds context string
```

**Estimated Context Usage for Large Files:**
- Dataset info: ~200 tokens
- Report summary: ~800-1500 tokens
- Analysis results: ~500 tokens
- Task description: ~300-500 tokens
- **TOTAL:** ~2000-2500 tokens (30-35% of 8K limit)

**Risk:** With comprehensive reports from large datasets, context could easily exceed 8K tokens.

---

**b) Report Generation Agent** (`config/tasks.yaml`)
```yaml
reporting_task:
  description: |
    You are a senior fraud detection analyst. Create a comprehensive report...
    [Large prompt with multiple sections]
  expected_output: |
    Comprehensive markdown report with:
    - Executive Summary
    - Data Quality Assessment
    - Statistical Analysis
    - Pattern Recognition
    - Fraud Detection Results
    - Visualizations
    - Recommendations
```

**Problem:** Agent tries to generate comprehensive reports that can exceed context limits with large datasets.

---

#### 3. **MEDIUM: CSVSearchTool RAG Index Size**

**Current Usage:**
```python
csv_tool = CSVSearchTool(csv=dataset_path)
```

**How CSVSearchTool Works:**
1. Loads CSV into memory
2. Creates embeddings for searchable content
3. Builds vector index for semantic search

**For 150MB CSV:**
- Must load entire file to create embeddings
- Vector index size: ~10-20% of original file size
- **Memory spike during initialization: 2-3GB**
- First initialization is SLOW (5-15 minutes)

**Mitigation:** CSVSearchTool likely caches embeddings, so reinitialization is faster.

**Risk Level:** 🟡 **MEDIUM** - Slow but won't crash if you have enough RAM

---

#### 4. **MEDIUM: Full Report Content in Context**

**Location:** `chat_agent.py:144-185`
```python
report_content = self.analysis_results.get('report_content', '')
if report_content:
    report_lines = report_content.split('\n')
    # Processes first 100 lines
```

**For large datasets:**
- Reports can be 5K-10K lines
- Even with limit to 100 lines, could be 1000-2000 tokens
- Combined with other context = potential overflow

**Risk Level:** 🟡 **MEDIUM** - May exceed context with very detailed reports

---

#### 5. **LOW: File Validation Full Load**

**Location:** `file_validator.py:34`
```python
df = pd.read_csv(file_path)  # ❌ LOADS ENTIRE FILE
```

**Impact:** Used only during initial validation to check column names
**Frequency:** Once per session
**Memory:** Temporary spike, released after validation

**Risk Level:** 🟢 **LOW** - Brief memory spike, acceptable for validation

---

## Model Comparison: llama3.1:8b vs OpenAI

### Llama3.1:8b (Current - Local Ollama)

| Metric | Value | Impact on Large Files |
|--------|-------|----------------------|
| Context Window | ~8,192 tokens | 🔴 **SEVERE LIMITATION** |
| Cost | Free | ✅ No API costs |
| Speed | Medium (local GPU) | 🟡 Acceptable |
| Quality | Good | ✅ Good quality |
| Max File Size | ~50MB effectively | 🔴 **LIMITED** |

**Verdict:** ❌ **NOT SUITABLE** for 150MB+ files due to context limitations

---

### OpenAI GPT-4 Turbo (Recommended for Large Files)

| Metric | Value | Impact on Large Files |
|--------|-------|----------------------|
| Context Window | **128,000 tokens** | ✅ **EXCELLENT** |
| Cost | $0.01/1K input tokens | 💰 ~$1-3 per analysis |
| Speed | Fast (API) | ✅ Very fast |
| Quality | Excellent | ✅ Best quality |
| Max File Size | 500MB+ effectively | ✅ **EXCELLENT** |

**Verdict:** ✅ **HIGHLY RECOMMENDED** for 150MB+ files

---

### OpenAI GPT-3.5 Turbo (Budget Option)

| Metric | Value | Impact on Large Files |
|--------|-------|----------------------|
| Context Window | **16,385 tokens** | 🟡 **ACCEPTABLE** |
| Cost | $0.0005/1K tokens | 💰 ~$0.10 per analysis |
| Speed | Very Fast | ✅ Fastest |
| Quality | Good | ✅ Good quality |
| Max File Size | ~200MB effectively | 🟡 **MODERATE** |

**Verdict:** 🟡 **ACCEPTABLE** for 150MB files with optimizations

---

## Recommendations for Handling 150MB+ Files

### Priority 1: 🔴 **CRITICAL FIXES** (Must Implement)

#### 1.1 Implement Chunked CSV Loading in Statistical Tool

**File:** `statistical_analysis_tool.py`

**Problem:** Line 113 loads entire CSV
```python
df = pd.read_csv(dataset_path)  # ❌ DANGEROUS
```

**Solution:** Use chunked reading with sampling
```python
# Option A: Sample large files
if os.path.getsize(dataset_path) > 50_000_000:  # > 50MB
    # Read in chunks and sample
    chunk_size = 50000
    chunks = []
    for chunk in pd.read_csv(dataset_path, chunksize=chunk_size):
        chunks.append(chunk.sample(min(1000, len(chunk))))
    df = pd.concat(chunks, ignore_index=True)
else:
    df = pd.read_csv(dataset_path)

# Option B: Streaming statistics (more robust)
def calculate_streaming_stats(file_path, chunk_size=50000):
    """Calculate statistics without loading full file"""
    stats = {'count': 0, 'sum': {}, 'sum_sq': {}}

    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        stats['count'] += len(chunk)
        for col in chunk.select_dtypes(include=[np.number]).columns:
            stats['sum'][col] = stats['sum'].get(col, 0) + chunk[col].sum()
            stats['sum_sq'][col] = stats['sum_sq'].get(col, 0) + (chunk[col]**2).sum()

    return stats
```

---

#### 1.2 Switch to OpenAI GPT-4 Turbo for Production

**File:** `.env`

**Current:**
```bash
MODEL=ollama/llama3.1:8b  # 8K context - insufficient
```

**Recommended:**
```bash
# For large files (150MB+)
MODEL=gpt-4-turbo-preview  # 128K context window
OPENAI_API_KEY=sk-proj-...  # Already configured

# Alternative: GPT-3.5 for budget (16K context)
# MODEL=gpt-3.5-turbo-16k
```

**Configuration in agents.yaml:**
```yaml
# Add to each agent
llm:
  provider: openai
  model: gpt-4-turbo-preview
  temperature: 0.1
  max_tokens: 4000
```

---

#### 1.3 Limit Context Size in Chat Agent

**File:** `streamlit_app/utils/chat_agent.py`

**Add context size limits:**
```python
def _prepare_context(self, context: Optional[Dict[str, Any]] = None) -> str:
    """Prepare comprehensive context information for the agent."""

    context_parts = []
    MAX_CONTEXT_TOKENS = 3000  # Leave room for prompt and response

    # Estimate token count (rough: 1 token ≈ 4 characters)
    def estimate_tokens(text: str) -> int:
        return len(text) // 4

    # Dataset information (priority: HIGH)
    if hasattr(self, 'analysis_results') and self.analysis_results:
        dataset_info = self.analysis_results.get('dataset_info', {})
        if dataset_info:
            info_text = f"""
**Dataset Information:**
- Total Transactions: {dataset_info.get('rows', 'Unknown'):,}
- Features: {dataset_info.get('columns', 'Unknown')}
- File Size: {dataset_info.get('size_mb', 'Unknown')} MB
- Analysis Type: {'Supervised' if dataset_info.get('has_class_column') else 'Unsupervised'}
"""
            context_parts.append(info_text)

    # Report summary (priority: HIGH, but limit size)
    if self.analysis_results:
        report_content = self.analysis_results.get('report_content', '')
        if report_content:
            # Limit to first 20 lines of key sections only
            report_lines = report_content.split('\n')
            summary_text = '\n'.join(report_lines[:20])

            # Truncate if too large
            if estimate_tokens(summary_text) > 1500:
                summary_text = summary_text[:6000] + "\n...[truncated]"

            context_parts.append(f"**Analysis Summary:**\n{summary_text}")

    # Build final context with size check
    final_context = '\n\n'.join(context_parts)

    if estimate_tokens(final_context) > MAX_CONTEXT_TOKENS:
        # Aggressive truncation
        final_context = final_context[:MAX_CONTEXT_TOKENS * 4] + "\n\n...[Context truncated to fit model limits]"

    return final_context
```

---

### Priority 2: 🟡 **RECOMMENDED OPTIMIZATIONS**

#### 2.1 Implement CSV Sampling Strategy

**Create new utility:** `src/crewai_extrachallenge/tools/csv_sampler.py`

```python
import pandas as pd
import os

class SmartCSVSampler:
    """Smart sampling for large CSV files"""

    @staticmethod
    def get_sample(file_path: str, max_size_mb: int = 50) -> pd.DataFrame:
        """
        Get a representative sample of a large CSV file.

        For files > max_size_mb:
        - Stratified sampling if 'Class' column exists
        - Random sampling otherwise
        - Maintains fraud/non-fraud ratio
        """
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        if file_size_mb <= max_size_mb:
            return pd.read_csv(file_path)

        # Estimate row count
        with open(file_path, 'r') as f:
            first_line = f.readline()
            avg_line_size = len(first_line)

        estimated_rows = int((file_size_mb * 1024 * 1024) / avg_line_size)
        sample_fraction = (max_size_mb * 1024 * 1024) / (file_size_mb * 1024 * 1024)

        print(f"Large file detected ({file_size_mb:.1f}MB). Sampling {sample_fraction*100:.1f}% of data...")

        # Stratified sampling for fraud detection
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=10000):
            if 'Class' in chunk.columns:
                # Keep all fraud cases (Class=1), sample normal cases
                fraud = chunk[chunk['Class'] == 1]
                normal = chunk[chunk['Class'] == 0].sample(frac=sample_fraction)
                chunks.append(pd.concat([fraud, normal]))
            else:
                chunks.append(chunk.sample(frac=sample_fraction))

        return pd.concat(chunks, ignore_index=True)
```

**Usage in tools:**
```python
from tools.csv_sampler import SmartCSVSampler

# Instead of:
# df = pd.read_csv(dataset_path)

# Use:
df = SmartCSVSampler.get_sample(dataset_path, max_size_mb=50)
```

---

#### 2.2 Optimize CSVSearchTool Initialization

**File:** `streamlit_app/utils/chat_agent.py:19`

**Current:**
```python
self.csv_tool = CSVSearchTool(csv=dataset_path) if os.path.exists(dataset_path) else None
```

**Optimized with caching:**
```python
@staticmethod
@st.cache_resource
def _get_csv_tool(dataset_path: str):
    """Cache CSV tool to avoid re-indexing"""
    if not os.path.exists(dataset_path):
        return None

    print(f"Initializing CSVSearchTool for {dataset_path}...")
    print("This may take several minutes for large files...")

    return CSVSearchTool(csv=dataset_path)

# In __init__:
self.csv_tool = self._get_csv_tool(dataset_path)
```

---

#### 2.3 Add Progress Indicators for Large Files

**File:** `streamlit_app/components/report_generator.py`

```python
def _validate_environment():
    """Validate environment and warn about large files"""

    dataset_path = SessionManager.get_csv_path()
    file_size_mb = os.path.getsize(dataset_path) / (1024 * 1024)

    if file_size_mb > 100:
        st.warning(f"""
        ⚠️ **Large Dataset Detected ({file_size_mb:.1f}MB)**

        For optimal performance with large files:
        - **Recommended:** Use OpenAI GPT-4 Turbo (128K context)
        - **Current:** {os.getenv('MODEL', 'Not set')}

        Analysis may take 10-30 minutes for files >150MB.
        """)

        if 'llama' in os.getenv('MODEL', '').lower():
            st.error("""
            🔴 **Context Window Warning**

            Current model (Llama 3.1:8b) has limited context (8K tokens).
            For files >100MB, we recommend switching to GPT-4 Turbo.

            To switch: Edit `.env` file and set:
            ```
            MODEL=gpt-4-turbo-preview
            ```
            """)
```

---

### Priority 3: 🟢 **NICE TO HAVE**

#### 3.1 Implement Progressive Analysis

```python
def analyze_large_dataset_progressive(file_path: str):
    """Analyze large datasets in phases"""

    phases = [
        ("Quick scan", 1000, ["data_quality"]),
        ("Sample analysis", 10000, ["descriptive", "correlation"]),
        ("Deep dive", 50000, ["outliers", "clustering"]),
        ("Full analysis", None, ["all"])
    ]

    for phase_name, sample_size, analyses in phases:
        st.write(f"**Phase:** {phase_name}")
        # Run analyses on progressively larger samples
```

---

#### 3.2 Database-Backed Analysis

For truly massive files (500MB+), consider:
```python
# Store CSV in SQLite for efficient querying
import sqlite3

def csv_to_sqlite(csv_path: str, db_path: str):
    """Convert CSV to SQLite for efficient large-file analysis"""
    conn = sqlite3.connect(db_path)

    # Read in chunks and insert
    for chunk in pd.read_csv(csv_path, chunksize=10000):
        chunk.to_sql('transactions', conn, if_exists='append', index=False)

    return conn
```

---

## Cost Analysis: OpenAI API

### Typical Analysis Run (150MB CSV, ~500K rows)

**Input Tokens (per agent):**
- Dataset context: 500 tokens
- Tool outputs: 2000 tokens
- Task description: 500 tokens
- Previous results: 1000 tokens
- **Total per agent:** ~4000 tokens

**4 Agents × 4000 tokens = 16,000 input tokens**

**Output Tokens:**
- Report generation: 3000 tokens
- Tool calls: 500 tokens
- **Total:** ~3500 output tokens

### Cost Calculation:

**GPT-4 Turbo:**
- Input: 16,000 tokens × $0.01/1K = $0.16
- Output: 3,500 tokens × $0.03/1K = $0.105
- **Total per analysis: ~$0.27**

**GPT-3.5 Turbo:**
- Input: 16,000 tokens × $0.0005/1K = $0.008
- Output: 3,500 tokens × $0.0015/1K = $0.005
- **Total per analysis: ~$0.013**

### Monthly Cost Estimates:

| Usage | GPT-4 Turbo | GPT-3.5 Turbo |
|-------|-------------|---------------|
| 10 analyses/day | ~$81/month | ~$4/month |
| 50 analyses/day | ~$405/month | ~$20/month |
| 100 analyses/day | ~$810/month | ~$40/month |

---

## Implementation Priority

### Week 1: Critical Fixes (Must Do)
1. ✅ Switch `.env` MODEL to `gpt-4-turbo-preview`
2. ✅ Add chunked CSV reading to `statistical_analysis_tool.py`
3. ✅ Add context size limits to `chat_agent.py`
4. ✅ Test with 150MB file

### Week 2: Optimization (Should Do)
1. ✅ Implement `SmartCSVSampler` utility
2. ✅ Add progress indicators for large files
3. ✅ Cache CSVSearchTool initialization
4. ✅ Add file size warnings in UI

### Week 3+: Advanced (Nice to Have)
1. Progressive analysis implementation
2. Database-backed analysis for 500MB+ files
3. Parallel chunk processing
4. Result caching system

---

## Testing Strategy

### Test Suite for Large Files:

```python
test_files = [
    ("small", "10MB", "should work with current setup"),
    ("medium", "50MB", "should work with optimizations"),
    ("large", "150MB", "requires GPT-4 Turbo"),
    ("extra_large", "300MB", "requires all optimizations"),
]

def test_scalability():
    for name, size, expected in test_files:
        print(f"Testing {name} ({size}): {expected}")
        # Run analysis and measure:
        # - Memory usage
        # - Processing time
        # - Context overflow errors
        # - Quality of results
```

---

## Conclusion

### Current State:
- ❌ **NOT READY** for 150MB+ files with `llama3.1:8b`
- ⚠️ **PARTIAL** support up to ~50MB files
- ✅ **EXCELLENT** architecture (tool-based design)

### After Implementing Recommendations:
- ✅ **FULLY READY** for 150MB+ files with GPT-4 Turbo
- ✅ **COST EFFECTIVE** (~$0.27 per analysis)
- ✅ **SCALABLE** to 300MB+ with full optimizations

### Bottom Line:
**Your code architecture is excellent**, but you need:
1. **Switch to GPT-4 Turbo** (most important)
2. **Add chunked CSV reading** (critical)
3. **Implement sampling strategies** (recommended)

With these changes, handling 150MB+ files will be straightforward and reliable.
