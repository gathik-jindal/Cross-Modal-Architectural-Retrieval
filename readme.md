# Cross-Modal Architectural Retrieval

This project processes floor plan SVG files and converts them into structured "contract" JSON representations for downstream NLP or retrieval tasks.

---

## 📁 Project Structure

```
contracts/
data/
floor_plan_nlp/
src/
  ├── batch_runner.py
  ├── constants.py
  ├── geometry.py
  ├── svg_parser.py
```

⚠️ **Note:**

- `data/` and `contracts/` are in `.gitignore`
- You must generate these locally (see steps below)

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <repo-name>
```

---

### 2. Install Dependencies

Make sure you are using Python 3.8+.

```bash
pip install -r requirements.txt
```

(Adjust if your project uses additional libraries.)

---

## 📥 Download & Prepare Dataset

This project uses the **Floor PlanCAD dataset**.

### Steps:

1. Download dataset from:
   https://floorplancad.github.io/

2. Create the following directory structure:

```
data/
  ├── train/
  └── test/
```

3. Place SVG files:

- Training SVGs → `data/train/`
- Testing SVGs → `data/test/`

---

## ⚙️ Generate Contracts (Main Step)

The `contracts/` folder is NOT included in the repo. It will be generated automatically.

In `src/batch_runner.py` change the following variable to your convenience, if you want to run it on the full dataset, set it to `None`

```python
TEST_LIMIT = 5
```

### Run the batch processing script:

```bash
python src/batch_runner.py
```

---

## 🔄 What This Script Does

The script :

- Scans:
  - `data/train/*.svg`
  - `data/test/*.svg`

- Converts each SVG into a structured JSON contract
- Saves outputs to:

  ```
  contracts/
    ├── train/
    └── test/
  ```

- Uses parallel processing for speed

---

## 📤 Output Structure

After running the script:

```
contracts/
  ├── train/
  │     ├── file1.json
  │     ├── file2.json
  │     └── ...
  └── test/
        ├── file1.json
        ├── file2.json
        └── ...
```

---

## 🛠️ Troubleshooting

### ❌ No SVG files found

- Ensure files are placed correctly:

  ```
  data/train/*.svg
  data/test/*.svg
  ```

### ❌ Import errors

- Make sure you're running from project root:

  ```bash
  python src/batch_runner.py
  ```

### ❌ Slow performance

- The script uses multiprocessing by default
- You can modify `max_workers` inside `batch_runner.py` if needed

---

## 🧠 Notes

- The parser logic is implemented in:

  ```
  src/svg_parser.py
  ```

- Key parameters you can tweak:
  - `epsilon`
  - `max_edges_per_node`

---

## ✅ Quick Start (TL;DR)

```bash
git clone <repo>
cd <repo>
pip install -r requirements.txt

# Download dataset and place SVGs into:
# data/train and data/test

python src/batch_runner.py
```

---

## 📌 Summary

| Step | Action                      |
| ---- | --------------------------- |
| 1    | Clone repo                  |
| 2    | Install dependencies        |
| 3    | Download dataset            |
| 4    | Place SVGs in `data/`       |
| 5    | Run batch script            |
| 6    | Get outputs in `contracts/` |

---

You're now ready to run the full pipeline 🚀
