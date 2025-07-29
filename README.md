# Cake-Classification
Cake‑image classification on 15 categories / 1 800 pictures (224 × 224). Benchmarks colour‑histogram + MLP baseline (≈21 % test acc.) against PVMLNet transfer‑learning; hidden‑layer ‑6 features push a linear head to 90 % test accuracy, with confusion analysis and handcrafted‑feature fusion experiments.

## 🎯 Goal
Build an image‑based classifier that recognises 15 cake types using a tiny dataset, starting from traditional colour/edge descriptors and ending with CNN transfer‑learning (PVMLNet).

## 📦 Dataset
* **Cake set** – 120 RGB images × 15 classes (100 train, 20 test, 224 × 224 px).
* Total: **1 800** photos (1 500 train, 300 test).

## 🛠️ Approaches
| Stage | Features | Model | Test acc. |
|-------|----------|-------|-----------|
| Baseline | Colour histogram (192‑D) | MLP (2 × 5000 ep) | **20.7 %** |
| Classical fusion | Colour + edge + co‑occ | MLP | **26.7 %** (best combo) |
| CNN transfer | PVMLNet last‑layer activations | Linear | **79.7 %** |
| **Best** | PVMLNet activations ‑6 | Linear | **90.3 %** |

## 🔍 Pipeline
1. **Low‑level feature extraction** – colour / edge histograms & co‑occurrence matrices.  
2. **MLP grid‑search** – batch, LR, epochs.  
3. **PVMLNet feature dump** – grab hidden activations, train shallow heads.  
4. **Transfer learning** – replace final layer, fine‑tune on cakes.  
5. **Evaluation** – confusion matrix & mis‑tagged samples analysis.  

## 🔑 Key findings
* **CNN beats handcrafting** – a frozen PVMLNet already triples baseline accuracy.  
* **Layer choice matters** – middle activations (‑6) give best linear separability.  
* **Feature fusion helps small nets** but stalls far below CNN performance.  
