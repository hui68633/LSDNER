# LSDNER  
Enhancing Few-Shot Named Entity Recognition via Label Semantic Description and Diversity Text  

## 🚀 Quick Start  
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate clues
python data/askForClues.py

# 3. Clustering & sampling
python demo_cluster.py

# 4. Generate responses
python ICL/askForResponse.py

# 5. Evaluation
python ICL/CADEC/metric/compute_F1.py
