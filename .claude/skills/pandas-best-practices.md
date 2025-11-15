# Pandas Best Practices: Elegant and Efficient Data Manipulation

This skill helps you write clean, performant pandas code using method chaining, vectorization, and modern patterns.

## Core Principles

### 1. **Always Vectorize - Never Loop**
Vectorized operations are 100-2400x faster than loops. Use built-in pandas/NumPy operations.

### 2. **Method Chaining for Readability**
Chain operations to create clear, maintainable pipelines. Each step should be obvious.

### 3. **Avoid Temporary Variables**
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

1. **Vectorize Everything**: Use built-in pandas/NumPy operations instead of loops (100-2400x faster)
2. **Chain Methods**: Use `.assign()`, `.query()`, `.pipe()` for readable pipelines
3. **No Intermediate Variables**: Reduce memory by chaining instead of creating temp DataFrames
4. **Use lambda in assign()**: Access the DataFrame with `lambda x:` for sequential operations
5. **Profile Before Optimizing**: Use `%%timeit` to measure; don't sacrifice readability for negligible gains
6. **Categorical for Strings**: Convert low-cardinality strings to `category` dtype
7. **NumPy for Heavy Lifting**: Drop to `.to_numpy()` for complex numerical operations
8. **String Operations**: Always use `.str` accessor (already vectorized)
9. **Avoid apply()**: Especially for numeric operations; use vectorized alternatives
10. **Named Aggregations**: Use `.agg()` with named aggregations for clarity

---

## 🚫 Anti-Patterns to Avoid

```python
# ❌ NEVER: Loop through DataFrame
for idx, row in df.iterrows():
    df.loc[idx, 'new_col'] = row['a'] + row['b']

# ❌ NEVER: Modify in place during chain (breaks chain)
df['new_col'] = df['a'] + df['b']

# ❌ NEVER: Use apply for simple arithmetic
df['total'] = df.apply(lambda row: row['a'] + row['b'], axis=1)

# ❌ AVOID: Deeply nested boolean indexing
df = df[df['a'] > 0]
df = df[df['b'] == 'active']
df = df[df['c'] < 100]

# ✅ DO THIS: Chain everything
df_result = (
    df
    .assign(new_col=lambda x: x['a'] + x['b'])
    .query('a > 0 & b == "active" & c < 100')
)
```

---

## 📚 References

- Use `.pipe()` for custom function chains
- Use `.assign()` for adding columns (immutable)
- Use `.query()` for filtering (cleaner than boolean indexing)
- Use `np.select()` for complex conditionals (faster than nested np.where)
- Use `pd.cut()` / `pd.qcut()` for binning
- Use `.str` accessor for vectorized string operations
- Use `.dt` accessor for vectorized datetime operations
- Use named aggregations in `.agg()` for clarity
- Profile with `%%timeit` in Jupyter or `pd.set_option('compute.use_numexpr', True)`

Remember: **Readable code that's 2x slower is better than unreadable code that's 2x faster. But readable code that's also 100x faster? That's the pandas way.**
