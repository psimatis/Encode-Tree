# encode-tree-cpp
**tl;dr** Use AutoEncoders to facilate high dimensional data indexing (C++, LibTorch)

**Problem/Motivation:** Multidimensional querying is inefficient (often slower than a linear search). Traditional indexing structures (e.g., R*-Tree, X-Tree) succumb to the curse of dimensionality or other structure-specific issues (e.g., boundary overlapping). Indexes such as the Pyramid technique and iMinMax adapt better to high dimensional spaces but require manual fine-tuning according to the data distribution. In addition, all these methodologies predate the current big data boom. Thus, novel solutions are required for efficient indexing on high dimensions.

**Solution:** 1) Use machine learning (i.e., Autoencoders) to automatically reduce the data dimensionality and 2) index the embedded space using a conventional index structure (e.g., B+-Tree).

**Input:** Multidimensional data in tabular form (e.g., Spotify songs)

**Output:** An index for efficient multidimensional queries.

**Notes:** 
1. This project includes an AutoEncoder written in C++ using LibTorch (PyTorch's C++ API).
2. The input data follow a standard csv format. Examples and Python scripts to create the data are located in the Data folder.
3. The '''CustomLoader.h''' and '''CSVLoader.h''' are custom made data loaders to load your data.
4. LibTorch is a great C++ library for machine learning but lacks documentation and tutorials. I hope my code will help others.  
