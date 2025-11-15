# Pandas Best Practices: Elegant and Efficient Data Manipulation

This skill helps you write clean, performant pandas code using method chaining, vectorization, and modern pandas 2.x+ features.

## Core Principles

### 1. **Always Vectorize - Never Loop**
Vectorization isn't just syntactic convenience—it uses SIMD (Single Instruction, Multiple Data) operations in C/Cython for 100-740x speedups over Python loops.

### 2. **Method Chaining for Readability**
Chain operations to create clear, maintainable pipelines treating DataFrames as immutable. Each step should be obvious.

### 3. **Use Modern Pandas Features**
Enable Copy-on-Write, use nullable dtypes (Int64, string), and PyArrow backend for performance and safety.

### 4. **Avoid Temporary Variables**
Use `.assign()`, `.pipe()`, and chaining to reduce intermediate DataFrames.

---

## 🚀 Vectorization Techniques

### ❌ NEVER Do This (Loops)
```python
# WRONG: Iterating through rows
for idx, row in df.iterrows():
    df.loc[idx, 'total'] = row['price'] * row['quantity']

# WRONG: Apply with lambda for simple operations
df['total'] = df.apply(lambda row: row['price'] * row['quantity'], axis=1)
```

### ✅ ALWAYS Do This (Vectorization)
```python
# CORRECT: Vectorized operations
df['total'] = df['price'] * df['quantity']

# CORRECT: Multiple conditions with np.select
conditions = [
    df['age'] < 18,
    df['age'] < 65,
    df['age'] >= 65
]
choices = ['minor', 'adult', 'senior']
df['age_group'] = np.select(conditions, choices, default='unknown')

# CORRECT: String operations (vectorized)
df['email_domain'] = df['email'].str.split('@').str[1]

# CORRECT: Conditional assignment with np.where
df['status'] = np.where(df['score'] >= 70, 'pass', 'fail')

# CORRECT: Multiple conditions
df['grade'] = np.where(df['score'] >= 90, 'A',
              np.where(df['score'] >= 80, 'B',
              np.where(df['score'] >= 70, 'C', 'F')))
```

### When Vectorization Isn't Enough
```python
# If you MUST iterate, use these (in order of preference):
# 1. List comprehension with zip (fastest iteration)
df['result'] = [complex_func(a, b) for a, b in zip(df['col1'], df['col2'])]

# 2. NumPy arrays directly (faster than Series)
values = df['col'].to_numpy()
result = [process(v) for v in values]
df['result'] = result

# 3. Only as last resort: apply (but profile first!)
df['result'] = df['col'].apply(complex_func)
```

---

## 🆕 Modern Pandas 2.x+ Features

### Copy-on-Write (Mandatory in pandas 3.0)
```python
# Enable Copy-on-Write (do this at the top of your code)
import pandas as pd
pd.options.mode.copy_on_write = True

# ❌ NEVER: Chained assignment (raises error with CoW)
df['foo'][df['bar'] > 5] = 100

# ✅ ALWAYS: Use .loc for assignment
df.loc[df['bar'] > 5, 'foo'] = 100

# Benefits: No SettingWithCopyWarning, lazy copies improve performance
```

### Nullable Dtypes (Prevent float conversion on nulls)
```python
# ❌ OLD: Nulls force conversion to float
pd.Series([1, 2, None])  # dtype: float64

# ✅ NEW: Nullable types preserve int/bool/string
pd.Series([1, 2, None], dtype='Int64')  # dtype: Int64 (capital I)
pd.Series([True, False, None], dtype='boolean')  # dtype: boolean
pd.Series(['a', 'b', None], dtype='string')  # dtype: string

# Convert existing DataFrame to nullable types
df = df.convert_dtypes()

# Use pd.NA (not np.nan) for missing values with nullable dtypes
```

### PyArrow Backend (35x faster I/O, better memory)
```python
# Read files with PyArrow backend
df = pd.read_csv('data.csv', dtype_backend='pyarrow', engine='pyarrow')

# Convert existing DataFrame
df = df.convert_dtypes(dtype_backend='pyarrow')

# Enable PyArrow strings by default
pd.options.future.infer_string = True

# Best for: Large datasets, string-heavy data, interop with DuckDB/Polars
```

### Modern Conditional Logic
```python
# ✅ NEW: case_when() for SQL-like conditional logic (pandas 2.2+)
df['tier'] = (
    pd.Series('default', index=df.index)
    .case_when([
        (df['revenue'] > 10000, 'platinum'),
        (df['revenue'] > 5000, 'gold'),
        (df['revenue'] > 1000, 'silver'),
    ])
)

# Still useful: np.where for simple if-else, np.select for arrays
```

---

## 🔗 Method Chaining Patterns

### Basic Chain Structure
```python
result = (
    df
    .drop_duplicates(subset=['id'])
    .query('age >= 18 & status == "active"')
    .assign(
        full_name=lambda x: x['first_name'] + ' ' + x['last_name'],
        age_group=lambda x: pd.cut(x['age'], bins=[0, 18, 35, 50, 100],
                                    labels=['youth', 'young_adult', 'adult', 'senior'])
    )
    .sort_values('created_date', ascending=False)
    .reset_index(drop=True)
)
```

### Advanced Chaining with .pipe()
```python
def add_age_features(df):
    """Add age-related features"""
    return df.assign(
        age_group=lambda x: pd.cut(x['age'], bins=[0, 18, 35, 50, 100]),
        is_adult=lambda x: x['age'] >= 18
    )

def add_revenue_features(df):
    """Add revenue calculations"""
    return df.assign(
        revenue=lambda x: x['price'] * x['quantity'],
        revenue_per_unit=lambda x: x['revenue'] / x['quantity']
    )

# Chain custom functions
result = (
    df
    .query('status == "active"')
    .pipe(add_age_features)
    .pipe(add_revenue_features)
    .pipe(lambda x: x[x['revenue'] > 0])  # Inline lambda for filtering
)
```

### Using .assign() Effectively
```python
# ✅ CORRECT: Multiple columns in one assign
df_clean = (
    df.assign(
        # Simple operations
        total=lambda x: x['price'] * x['quantity'],
        # Reference newly created columns
        total_with_tax=lambda x: x['total'] * 1.1,
        # Conditional logic
        discount=lambda x: np.where(x['total'] > 100, 0.1, 0),
        # String operations
        name_upper=lambda x: x['name'].str.upper(),
        # Date operations
        year=lambda x: pd.to_datetime(x['date']).dt.year
    )
)

# ❌ AVOID: Modifying in place (breaks chaining)
df['total'] = df['price'] * df['quantity']  # Don't do this in chains
```

---

## 🎯 Common Patterns & Solutions

### Pattern 1: Filtering with query()
```python
# ✅ CORRECT: Clean and readable
df_filtered = (
    df
    .query('age >= 18 & age < 65')
    .query('status == "active" | status == "pending"')
    .query('revenue > @revenue_threshold')  # Use @ for variables
)

# Instead of:
df_filtered = df[(df['age'] >= 18) & (df['age'] < 65) &
                 ((df['status'] == 'active') | (df['status'] == 'pending'))]
```

### Pattern 2: Groupby with Multiple Aggregations
```python
# ✅ CORRECT: Named aggregations (pandas >= 1.0)
summary = (
    df
    .groupby(['category', 'region'])
    .agg(
        total_revenue=('revenue', 'sum'),
        avg_price=('price', 'mean'),
        customer_count=('customer_id', 'nunique'),
        max_date=('order_date', 'max')
    )
    .reset_index()
    .query('total_revenue > 1000')
    .sort_values('total_revenue', ascending=False)
)

# Multiple operations per column
summary = (
    df
    .groupby('category')
    .agg(
        revenue_sum=('revenue', 'sum'),
        revenue_mean=('revenue', 'mean'),
        revenue_std=('revenue', 'std'),
        count=('revenue', 'count')
    )
)
```

### Pattern 3: Handling Missing Data
```python
# ✅ CORRECT: Chained null handling
df_clean = (
    df
    .dropna(subset=['id', 'email'])  # Drop if critical fields missing
    .assign(
        age=lambda x: x['age'].fillna(x['age'].median()),
        category=lambda x: x['category'].fillna('unknown'),
        score=lambda x: x['score'].fillna(0)
    )
)
```

### Pattern 4: String Operations (Always Vectorized)
```python
# ✅ CORRECT: Vectorized string operations
df_clean = (
    df
    .assign(
        email_clean=lambda x: x['email'].str.lower().str.strip(),
        domain=lambda x: x['email'].str.split('@').str[1],
        name_parts=lambda x: x['full_name'].str.split(' ', n=1),
        first_name=lambda x: x['name_parts'].str[0],
        last_name=lambda x: x['name_parts'].str[1],
        has_keyword=lambda x: x['description'].str.contains('important', case=False, na=False)
    )
    .drop(columns=['name_parts'])
)
```

### Pattern 5: Date/Time Operations
```python
# ✅ CORRECT: Vectorized datetime operations
df_with_dates = (
    df
    .assign(
        date=lambda x: pd.to_datetime(x['date_str']),
        year=lambda x: x['date'].dt.year,
        month=lambda x: x['date'].dt.month,
        day_name=lambda x: x['date'].dt.day_name(),
        is_weekend=lambda x: x['date'].dt.dayofweek >= 5,
        days_since=lambda x: (pd.Timestamp.now() - x['date']).dt.days
    )
)
```

### Pattern 6: Merging in Chains
```python
# ✅ CORRECT: Merge as part of pipeline
result = (
    df1
    .query('status == "active"')
    .merge(df2, on='id', how='left', validate='m:1')
    .merge(df3, left_on='category_id', right_on='id', how='left')
    .drop(columns=['id_y'])
    .rename(columns={'id_x': 'id'})
    .assign(
        combined_score=lambda x: x['score1'] + x['score2']
    )
)
```

### Pattern 7: Reshaping Data
```python
# ✅ CORRECT: Pivot and melt in chains
wide_format = (
    df
    .pivot_table(
        values='revenue',
        index='customer_id',
        columns='product_category',
        aggfunc='sum',
        fill_value=0
    )
    .reset_index()
    .assign(
        total=lambda x: x.select_dtypes(include=[np.number]).sum(axis=1)
    )
)

# Long format
long_format = (
    df
    .melt(
        id_vars=['customer_id', 'date'],
        value_vars=['product_a', 'product_b', 'product_c'],
        var_name='product',
        value_name='quantity'
    )
    .query('quantity > 0')
)
```

---

## 🔥 Performance Tips

### 1. **Use Categorical Data for Repeated Strings**
```python
# ✅ CORRECT: Convert to categorical for memory/speed
df = df.assign(
    category=lambda x: x['category'].astype('category'),
    status=lambda x: x['status'].astype('category')
)
# Can reduce memory by 90%+ for low-cardinality string columns
```

### 2. **Avoid apply() for Numeric Operations**
```python
# ❌ WRONG: Using apply unnecessarily
df['total'] = df.apply(lambda row: row['a'] + row['b'], axis=1)

# ✅ CORRECT: Direct vectorization (100x+ faster)
df['total'] = df['a'] + df['b']
```

### 3. **Use eval() for Complex Expressions**
```python
# ✅ CORRECT: eval for complex numeric expressions (faster)
df = df.eval('''
    total = price * quantity
    profit = total - cost
    margin = profit / total
''')
```

### 4. **Boolean Indexing vs query()**
```python
# Both are vectorized, choose based on readability
# query() is cleaner for complex conditions
result = df.query('age >= 18 & status == "active" & score > 80')

# Boolean indexing for simple cases or when you need the mask
mask = df['age'] >= 18
result = df[mask]
```

### 5. **Work with NumPy Directly for Heavy Computation**
```python
# ✅ CORRECT: Drop to NumPy for complex calculations
values = df[['a', 'b', 'c']].to_numpy()
result = complex_numpy_operation(values)
df['result'] = result
```

---

## 📊 Complete Example: Real-World Pipeline

```python
import pandas as pd
import numpy as np

def clean_and_process_sales_data(raw_df, min_revenue=100):
    """
    Process raw sales data with method chaining.

    - Cleans data
    - Adds features
    - Filters
    - Aggregates
    All using vectorized operations and method chaining.
    """
    return (
        raw_df
        # 1. Initial cleaning
        .drop_duplicates(subset=['order_id'])
        .dropna(subset=['customer_id', 'order_date'])
        .assign(
            order_date=lambda x: pd.to_datetime(x['order_date']),
            email=lambda x: x['email'].str.lower().str.strip(),
            product_category=lambda x: x['product_category'].astype('category')
        )

        # 2. Feature engineering (all vectorized)
        .assign(
            # Numeric features
            revenue=lambda x: x['price'] * x['quantity'],
            discount_amount=lambda x: x['revenue'] * x['discount_rate'],
            final_amount=lambda x: x['revenue'] - x['discount_amount'],

            # Date features
            year=lambda x: x['order_date'].dt.year,
            month=lambda x: x['order_date'].dt.month,
            quarter=lambda x: x['order_date'].dt.quarter,
            is_weekend=lambda x: x['order_date'].dt.dayofweek >= 5,

            # Categorical features
            revenue_tier=lambda x: pd.cut(
                x['revenue'],
                bins=[0, 100, 500, 1000, np.inf],
                labels=['low', 'medium', 'high', 'premium']
            ),

            # Boolean features
            is_bulk_order=lambda x: x['quantity'] >= 10,
            is_discounted=lambda x: x['discount_rate'] > 0,
            is_high_value=lambda x: x['final_amount'] > min_revenue
        )

        # 3. Filtering
        .query('final_amount > @min_revenue')
        .query('year >= 2023')

        # 4. Sorting
        .sort_values(['customer_id', 'order_date'])

        # 5. Final cleanup
        .reset_index(drop=True)
    )

# Usage
processed_df = clean_and_process_sales_data(raw_df, min_revenue=100)

# Group analysis (also vectorized)
customer_summary = (
    processed_df
    .groupby('customer_id')
    .agg(
        total_orders=('order_id', 'count'),
        total_revenue=('final_amount', 'sum'),
        avg_order_value=('final_amount', 'mean'),
        first_order=('order_date', 'min'),
        last_order=('order_date', 'max'),
        favorite_category=('product_category', lambda x: x.mode()[0] if len(x.mode()) > 0 else None)
    )
    .reset_index()
    .assign(
        days_active=lambda x: (x['last_order'] - x['first_order']).dt.days,
        customer_tier=lambda x: pd.qcut(x['total_revenue'], q=4, labels=['bronze', 'silver', 'gold', 'platinum'])
    )
    .sort_values('total_revenue', ascending=False)
)
```

---

## 🎓 Key Takeaways

### Performance & Correctness
1. **Never use iterrows()**: 740x slower than vectorization. Use vectorized ops or itertuples() if absolutely necessary
2. **Vectorization uses SIMD**: Not just syntax—it's C/Cython operations processing multiple elements simultaneously
3. **Profile first**: Use `%%timeit` to measure actual bottlenecks (90/10 rule applies)

### Modern Pandas (2.x → 3.0)
4. **Enable Copy-on-Write**: `pd.options.mode.copy_on_write = True` (mandatory in pandas 3.0)
5. **Use nullable dtypes**: Int64, boolean, string (prevent float conversion on nulls)
6. **PyArrow for large data**: `dtype_backend='pyarrow'` for 35x faster I/O and better memory
7. **Avoid inplace=True**: No performance benefit, breaks chaining, discouraged by pandas developers

### Code Patterns
8. **Chain everything**: Treat DataFrames as immutable; use `.assign()`, `.pipe()`, `.query()`
9. **Never use chained indexing**: Always use `.loc` explicitly to avoid SettingWithCopyWarning
10. **Specify dtypes at load**: Never accept defaults—use category for low-cardinality strings (100x memory reduction)
11. **Use case_when()**: For multi-condition logic (pandas 2.2+); cleaner than nested np.where()
12. **apply() only for strings/complex logic**: Never for numeric operations—vectorize instead

---

## 🚫 Anti-Patterns to Avoid

### 1. Never Use iterrows() (740x slower than vectorization)
```python
# ❌ WORST: iterrows() creates Series objects (devastating performance)
for idx, row in df.iterrows():
    df.loc[idx, 'new_col'] = row['a'] + row['b']

# ⚠️ FALLBACK: If you MUST iterate, use itertuples() (49x faster than iterrows)
for row in df.itertuples():
    # Access as row.column_name
    result = row.a + row.b

# ✅ BEST: Vectorize (740x faster than iterrows)
df['new_col'] = df['a'] + df['b']
```

### 2. Avoid Chained Indexing (causes SettingWithCopyWarning)
```python
# ❌ WRONG: Chained indexing - may modify copy, not original!
df['column'][df['condition'] > value] = new_value
df[df['x'] > 0]['y'] = 10  # Unpredictable - view or copy?

# ✅ CORRECT: Use .loc explicitly
df.loc[df['condition'] > value, 'column'] = new_value
df.loc[df['x'] > 0, 'y'] = 10

# Or explicitly copy if working with subset
subset = df[df['condition']].copy()
subset['column'] = new_value
```

### 3. Don't Use apply() for Numeric Operations
```python
# ❌ WRONG: apply for simple arithmetic (10-100x slower)
df['total'] = df.apply(lambda row: row['a'] + row['b'], axis=1)
df['total'] = df.apply(lambda row: row['price'] * row['quantity'], axis=1)

# ✅ CORRECT: Direct vectorization
df['total'] = df['a'] + df['b']
df['total'] = df['price'] * df['quantity']
```

### 4. Avoid inplace=True (no performance benefit)
```python
# ❌ WRONG: inplace doesn't save memory and breaks chaining
df.dropna(inplace=True)
df.drop(columns=['col'], inplace=True)

# ✅ CORRECT: Return new DataFrame (enables chaining)
df = df.dropna()
df = df.drop(columns=['col'])

# Or in a chain
df = (
    df
    .dropna()
    .drop(columns=['col'])
)
```

### 5. Use Proper dtypes (avoid object dtype waste)
```python
# ❌ WRONG: Default dtypes waste memory
df = pd.read_csv('data.csv')  # All defaults: int64, float64, object

# ✅ CORRECT: Specify types at load time
df = pd.read_csv(
    'data.csv',
    dtype={
        'id': 'Int32',  # Not Int64 if values fit
        'status': 'category',  # For repeated strings (90%+ memory reduction)
        'name': 'string',  # Not object
        'is_active': 'boolean'  # Not object
    },
    parse_dates=['created_at']
)

# Convert low-cardinality strings to category (100x memory reduction)
# Rule: if unique_values < 50% of total rows, use category
df['country'] = df['country'].astype('category')
```

---

## 📚 Quick Reference

### Essential Methods
- `.assign()` - Add/modify columns (immutable, chainable)
- `.pipe()` - Chain custom functions
- `.query()` - Filter with string expressions (cleaner than boolean indexing)
- `.loc[]` - Label-based indexing (always use for assignment)
- `.case_when()` - Multi-condition logic (pandas 2.2+)

### Performance Helpers
- `np.select()` - Complex conditionals (faster than nested np.where)
- `pd.cut()` / `pd.qcut()` - Binning/bucketing
- `.str` accessor - Vectorized string operations
- `.dt` accessor - Vectorized datetime operations
- Named aggregations in `.agg()` - Clear, performant aggregations

### Tools
- Profile: `%%timeit` in Jupyter notebooks
- Memory: `df.info(memory_usage='deep')` and `df.memory_usage(deep=True).sum()`
- Type conversion: `df.convert_dtypes()` or `df.convert_dtypes(dtype_backend='pyarrow')`

### Further Learning
Based on Matt Harrison's **"Effective Pandas"** (2nd edition) and pandas 2.x/3.0 migration guides. Key insight: most pandas performance problems stem from code patterns, not library limitations. Replacing iterrows() with vectorization yields 740x speedups; proper dtypes reduce memory by 100x.

---

**Remember**: Readable code that's 2x slower is better than unreadable code that's 2x faster. But readable code that's also 100x faster? That's the pandas way.
