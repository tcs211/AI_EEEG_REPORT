# EEG-AI-Report-Generator

A hybrid AI system that performs automated EEG background analysis and generates clinical reports. This is the implementation of our IEEE JBHI paper (2024).

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FJBHI.2024.3496996-blue)](https://doi.org/10.1109/JBHI.2024.3496996)

[![arXiv](https://arxiv.org/abs/2411.09874b)](https://arxiv.org/abs/2411.09874)

## Paper Information

This code implements the methodology described in:

"A Hybrid Artificial Intelligence System for Automated EEG Background Analysis and Report Generation"  
*IEEE Journal of Biomedical and Health Informatics (2024)*  
https://doi.org/10.1109/JBHI.2024.3496996

The paper and its contents are © 2024 IEEE. 
<!-- Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works. -->

## Code License

This code is released under the GNU General Public License v3.0 (GPL-3.0). You are free to use, modify, and distribute this code according to the terms of the GPL-3.0 license.

## Overview

This repository provides an implementation of automated EEG background analysis and report generation using a hybrid AI approach. The system combines deep learning for EEG analysis with large language models for report generation.

### Key Features

- Automated EEG background analysis
- AI-powered report generation using multiple LLM providers
- Support for standard EEG file formats (EDF, FIF)
- Support for SPIS dataset MAT files
- Multi-language report generation
- PDF report export

## Installation

### Prerequisites

We recommend using a Conda environment for installation.

### Dependencies

Install required packages:

```bash
pip install tensorflow google-generativeai anthropic openai mne python-dotenv \
    ipykernel matplotlib pyod pandas scikit-learn seaborn tqdm ipywidgets \
    PyWavelets beautifulsoup4 fpdf2 mne-qt-browser PyQt6 dit librosa \
    statsmodels pyinform pymatreader6
```

### API Configuration

1. Create a `config.env` file in the project root
2. Add your API keys:
```
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY # for Google Gemini
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_API_KEY # for Anthropic Claude
OPENAI_KEY=YOUR_OPENAI_KEY # for OpenAI gpt models
```

## Usage

### Supported File Formats
- EDF: General EEG recordings
- FIF: General EEG recordings
- MAT: SPIS dataset files only


### Required EEG Channels

The system requires the following 10-20 system electrodes (or equivalent mapped channels):
- Fp1, Fp2
- F7, F8, F3, F4, Fz
- C3, C4, Cz
- P3, P4, Pz
- T3, T4, T5, T6
- O1, O2

## You can get following datasets from the following links and use them to test the code.

### SPIS Dataset (MAT format)
Open-source resting-state EEG data:
- [SPIS Dataset Repository](https://github.com/mastaneht/SPIS-Resting-State-Dataset/tree/master/Pre-SART%20EEG)

### Temple University Hospital (TUH) EEG Dataset (EDF format)
Large clinical EEG database:
- [TUH EEG Dataset](https://isip.piconepress.com/projects/tuh_eeg/)

### Command Parameters

```bash
python report.py <eeg_file> [options]
```

#### Required Parameters
- `eeg_file`: Path to the input EEG data file (EDF, FIF, or MAT format)

#### Optional Parameters
- `--pdf`: Generate output in PDF format
- `--out <directory>`: Specify output directory for generated reports (default: current directory)
- `--ai`: Enable automated report generation using Large Language Models (LLMs)
  - Uses configured LLM APIs (Google PaLM, Anthropic Claude, OpenAI) for report generation
  - Requires valid API keys in config.env
- `--lang <language>`: Specify report language
  - Default: "english"
  - Supports any language available in the configured LLM models
  - Common options: English, Chinese (Simplified/Traditional), Japanese, Korean, Spanish, French, German, etc.
  - Language support depends on the capabilities of the configured LLM models
- `--llm <model>`: Specify LLM model for report generation
  - Default: "gemini-1.5-pro" 
  - Suggestions: "gemini-1.5-pro", "claude-3-5-sonnet-20240620", "gpt-4o"

### Example Commands

```bash
# Basic usage with SPIS dataset (MAT format)
python report.py ./SPIS_dataset/S04_restingPre_EC.mat \
    --pdf \            # Generate PDF report
    --out ./pdf \      # Save to ./pdf directory
    --ai \             # Enable LLM report generation
    --lang "english"   # Generate report in English
    --llm "gemini-1.5-pro" # Use Google Gemini LLM model

# Generate report in Traditional Chinese
python report.py ./recordings/patient001.edf \
    --pdf \
    --out ./reports \
    --ai \
    --lang "traditional chinese"
    --llm "gpt-4o"

# Generate report in Japanese
python report.py ./recordings/patient002.edf \
    --pdf \
    --out ./reports \
    --ai \
    --lang "japanese"
    --llm "claude-3-5-sonnet-20240620"
```

### Note on Language Support

The `--lang` parameter accepts languages supported by the configured LLM models. Language availability and quality may vary depending on the specific LLM model being used. Please refer to the documentation of your configured LLM providers (Google Gemini, Anthropic Claude, OpenAI) for detailed language support information.
The '--llm' parameter specifies the LLM model to be used for report generation. The default model is "gemini-1.5-pro". Latest models can be found on the respective LLM provider websites.


## Code Attribution

The `eegFeatureExtract.py` module is adapted from:
> S. Saba-Sadiya, et al. "Unsupervised EEG artifact detection and correction," Frontiers in Digital Health, vol. 2, 2021. 
> [Original Repository](https://github.com/sari-saba-sadiya/EEGExtract)

## Citation

If you use this code in your research, please cite:

```bibtex
@ARTICLE{10752384,
  author={Tung, Chin-Sung and Liang, Sheng-Fu and Chang, Shu-Feng and Young, Chung-Ping},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={A Hybrid Artificial Intelligence System for Automated EEG Background Analysis and Report Generation}, 
  year={2024},
  pages={1-13},
  keywords={Electroencephalography (EEG);artificial intelligence;deep learning;report generation;large language models},
  doi={10.1109/JBHI.2024.3496996}}
```