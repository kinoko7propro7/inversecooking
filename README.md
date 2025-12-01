# Food Embeddings & Inverse Cooking Project

This is an extended project based on **Inverse Cooking**, focusing on **Deep Analysis and Visualization of Ingredient Embeddings**. This project not only generates recipes from food images but also provides powerful tools to explore semantic relationships between ingredients, supporting similarity calculation, replacement suggestions, and various visualizations.

## 🚀 Key Features

### 1. 🍳 Recipe Generation from Images
Using a deep learning model (ResNet + Transformer), input a food image to generate:
- **Title**
- **Ingredients List**
- **Cooking Instructions**

### 2. 📊 Embeddings Visualization & Analysis
This project deeply explores the ingredient vector representations (Embeddings) learned by the model, providing multiple visualization tools:
- **Heatmaps**: Intuitively display cosine similarity between different ingredients/instructions.
- **2D Projection (PCA / t-SNE)**: Maps high-dimensional ingredient vectors to a 2D plane, showing ingredient clustering relationships (e.g., "meats" and "vegetables" clustering together).
- **Dendrogram**: Displays hierarchical classification relationships between ingredients.

### 3. 🔄 Ingredient Replacement Suggestions
Based on vector space distance in Embeddings, the system can intelligently recommend substitute ingredients.
- *Example: If "onion" is missing, the system suggests "shallot" or "leek".*

### 4. 🖥️ Interactive Demo
A complete Jupyter Notebook (`src/demo.ipynb`) is provided, supporting:
- Uploading local images or using URLs to generate recipes.
- Interactive queries for ingredient replacement options.
- One-click generation of all analysis charts.

---

## 📂 Project Structure

```text
inversecooking-master/
├── data/                   # Stores model weights and vocabulary files
│   ├── modelbest.ckpt      # Pretrained model weights
│   ├── ingr_vocab.pkl      # Ingredient vocabulary
│   └── instr_vocab.pkl     # Instruction vocabulary
├── src/                    # Source code directory
│   ├── demo.ipynb          # [Core] Main entry for interactive demo and visualization analysis
│   ├── 相似度.py            # [Tool] Standalone script for ingredient similarity calculation
│   ├── model.py            # Model definition (Transformer Decoder + Attention)
│   ├── train.py            # Training script
│   └── ...
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

## 🛠️ Installation

### 1. Dependencies
This project is based on Python 3.6+ and PyTorch.
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
Please ensure the following files are present in the `data/` directory (download if missing):
- `ingr_vocab.pkl`
- `instr_vocab.pkl`
- `modelbest.ckpt`

---

## 📖 Usage

### Method 1: Using Jupyter Notebook (Recommended)
Open `src/demo.ipynb` to experience the full functionality:
1. **Load Model**: Run the initialization cells.
2. **Generate Recipe**: Set the image path and generate a recipe.
3. **Visualization Analysis**: Run visualization cells to view Heatmaps, PCA scatter plots, and Dendrograms.
4. **Ingredient Replacement**: Use `what_can_replace(recipe_num, 'ingredient')` to query replacement suggestions.

### Method 2: Using Similarity Script
If you only want to quickly query ingredient similarity, you can run:
```bash
cd src
python Simularity.py
```
*(Ensure path configurations inside the script are correct)*

---

## 📚 Citation

This project is based on the code from the following paper:

*Amaia Salvador, Michal Drozdzal, Xavier Giro-i-Nieto, Adriana Romero.*
**[Inverse Cooking: Recipe Generation from Food Images.](https://arxiv.org/abs/1812.06164)**
*CVPR 2019*

```bibtex
@InProceedings{Salvador2019inversecooking,
author = {Salvador, Amaia and Drozdzal, Michal and Giro-i-Nieto, Xavier and Romero, Adriana},
title = {Inverse Cooking: Recipe Generation From Food Images},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

## License
MIT License.
