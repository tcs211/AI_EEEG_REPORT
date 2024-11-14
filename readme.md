# A Hybrid AI System for Automated EEG Background Analysis and Report Generation
## Introduction
This repository contains the code for the paper titled "A Hybrid Artificial Intelligence System for Automated EEG Background Analysis and Report Generation" by Chin-Sung Tung, Sheng-Fu Liang, Shu-Feng Chang, and Chung-Ping Young (doi: 10.1109/JBHI.2024.3496996). The paper is accepted for publication in the IEEE Journal of Biomedical and Health Informatics. The code in this repository demonstrates the example code to create a full automated EEG background analysis and report generation by LLM API. The code includes the following functionalities:

1. The file - eegFeatureExtract.py is from the following repository and it's original paper:  S. Saba-Sadiya, E. Chantland, T. Alhanai, T. Liu, and M. M. Ghassemi,
“Unsupervised eeg artifact detection and correction,” Frontiers in Digital
Health, vol. 2, 2021. https://github.com/sari-saba-sadiya/EEGExtract

1. Since the dataset in this work is not available for ethical reasons, you can use the online open-source datasets to test the code, such as the SPIS Dataset, the link is provided below.
1. The CNN model architecture can be found in the models.py file. 
1. The prompts for LLMs to generate and validate the EEG report can be found in the prompts.py file.


## Environment Setup

To set up the environment for running the code, follow these steps:

1. Conda Environment is recommended for running the code. 
2. Install the required python dependencies by running the following command:
   ```
   pip install tensorflow google-generativeai anthropic openai mne python-dotenv ipykernel matplotlib pyod pandas scikit-learn seaborn tqdm ipywidgets PyWavelets beautifulsoup4 fpdf2 mne-qt-browser PyQt6 dit librosa statsmodels pyinform pymatreader6 
   ```

## Create a config.env File

Create a `config.env` file in the root directory of the project and add the following API keys:

```python
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_API_KEY
OPENAI_KEY=YOUR_OPENAI_KEY
```

Replace `YOUR_GOOGLE_API_KEY`, `YOUR_ANTHROPIC_API_KEY`, and `YOUR_OPENAI_KEY` with your actual API keys.

## Test with open-source Dataset

### SPIS Dataset
[https://github.com/mastaneht/SPIS-Resting-State-Dataset/tree/master/Pre-SART%20EEG](https://github.com/mastaneht/SPIS-Resting-State-Dataset/tree/master/Pre-SART%20EEG)

### Temple University Hospital EEG Dataset
[https://isip.piconepress.com/projects/tuh_eeg/](https://isip.piconepress.com/projects/tuh_eeg/)


## Example Usage

To run the code, use the following command:

```python
python report.py ./SPIS_dataset/S04_restingPre_EC.mat --pdf --out ./pdf --ai --lang "english"
```

The command takes the following arguments:
- `./SPIS_dataset/S04_restingPre_EC.mat`: The path to the input EEG data file.
- `--pdf`: Flag to generate the report in PDF format.
- `--out ./pdf`: The output directory where the generated report will be saved.
- `--ai`: Flag to enable AI-based analysis and report generation.
- `--lang "english"`: The language of the generated report (in this case, English).

Make sure to replace `./SPIS_dataset/S04_restingPre_EC.mat` with the path to your desired input EEG data file.

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). If you use or share this work, you must cite the following citation.

## Citation

This article is accepted for publication in the IEEE Journal of Biomedical and Health Informatics. If you use this code, please cite the following article:

```bibtex
@ARTICLE{10752384,
  author={Tung, Chin-Sung and Liang, Sheng-Fu and Chang, Shu-Feng and Young, Chung-Ping},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={A Hybrid Artificial Intelligence System for Automated EEG Background Analysis and Report Generation}, 
  year={2024},
  volume={},
  number={},
  pages={1-13},
  keywords={Electroencephalography (EEG);artificial intelligence;deep learning;report generation;large language models},
  doi={10.1109/JBHI.2024.3496996}}
```
