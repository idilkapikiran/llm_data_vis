# 3D Part Segmentation with Language-Guided 2D Masks

This project provides an interactive framework for segmenting parts of 3D objects using natural language prompts, multi-view 2D segmentation, and vision-language models. The interface is built using Streamlit.

## Setup & Usage

```bash
# Initialize submodules
git submodule update --init --recursive

# Install Git LFS and pull large files
git lfs install
git lfs pull

# Install Python dependencies
pip install -r requirements.txt

# Install submodule dependencies
cd Grounded_Segment_Anything
pip install -e .

# Run the Streamlit app
streamlit run app.py
```
