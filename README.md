# RNN NLP Assignment

This project contains a simple RNN implementation using Keras to:
1. Classify educational texts into categories: Math, Science, History, English.
2. Generate the next 20 words given a starting sentence.

## Folder Structure

```
.
├── datasets/
│   ├── classification_data.txt
│   └── generation_data.txt
├── src/
│   ├── classification_model.py
│   └── generation_model.py
├── utils/ (optional)
```

## How to Run

1. Clone the repository.
2. Navigate to the root directory.
3. Install requirements:
   ```
   pip install keras tensorflow scikit-learn pandas
   ```
4. Run:
   - `python src/classification_model.py`
   - `python src/generation_model.py`

### Classification
```
Test Accuracy: 1.0
```

### ✅ Text Generation

**Example Input:**
science is the study of the structure and behavior of the physical and

**Model Output:**
science is the study of the structure and behavior of the physical and chemical


