##################################################
# This file is used to test the autimatic EEG report generation system.
# The origin dataset is not available due to ethical reasons.
# You can test the system with the open-source dataset.
# SPIS_dataset: download the dataset from the following link:
# https://github.com/mastaneht/SPIS-Resting-State-Dataset/tree/master/Pre-SART%20EEG
# Here is an example of how to run the system:
# python report.py ./SPIS_dataset/S04_restingPre_EC.mat --pdf  --out ./pdf --ai --lang "english"
# 
# For using the LLMs to generate the report, you need to set the GOOGLE_API_KEY in the config.env file.
##################################################

from auto_report import CreateReport
import argparse
import os
from dotenv import load_dotenv
def main():
    parser = argparse.ArgumentParser(description='create report automatically')
    # filename
    parser.add_argument('edf_file', type=str, help='edf filename')    
    parser.add_argument('--pdf', action='store_true', help='output pdf')
    parser.add_argument('--ai', action='store_true', help='output ai')
    parser.add_argument('--lang', type=str, help='report language')
    parser.add_argument('--out', type=str, help='output folder')
    # llm model
    parser.add_argument('--llm', type=str, help='llm model')
    # TUH eeg
    parser.add_argument('--tuh', type=bool, help='tuh eeg')

    args = parser.parse_args()
    if '/' in args.edf_file:
        edf_filename = args.edf_file.split('/')[-1]
        edf_path = '/'.join(args.edf_file.split('/')[:-1])
    else:
        edf_filename = args.edf_file.split('\\')[-1]
        edf_path = '\\'.join(args.edf_file.split('\\')[:-1])

    outputPdf = args.pdf
    aiReport = args.ai
    reportLang = args.lang
    output_folder = args.out
    llm_model = args.llm
    tuh_eeg = args.tuh
    if output_folder is None:
        output_folder = './'

    try :
    # create report
        envFile=os.path.join(os.getcwd(),'config.env')
        load_dotenv(envFile)
        Google_API_KEY= os.environ.get('GOOGLE_API_KEY')
        OPENAI_API_KEY=os.environ.get('OPENAI_KEY')
        ANTHROPIC_API_KEY=os.environ.get('ANTHROPIC_API_KEY')

        # check model is gpt, claude or gemini
        if llm_model is None:
            llm_model = 'gemini-1.5-flash'
            print('No LLM model specified, using default model: gemini-1.5-flash')
        if llm_model is not None:
            if 'gpt' in llm_model.lower():
                LLM_API_KEY = OPENAI_API_KEY
            elif 'claude' in llm_model.lower():
                LLM_API_KEY = ANTHROPIC_API_KEY
            elif 'gemini' in llm_model.lower():
                LLM_API_KEY = Google_API_KEY
            else:
                LLM_API_KEY = None
        
        if LLM_API_KEY is None:
            print('Please set the LLM_API_KEY in the config.env file')
            return

        CreateReport(edf_filename,edf_path, outputPdf=outputPdf, LLM_API_KEY=LLM_API_KEY,
                    llm_model=llm_model, unit_uV= not tuh_eeg,
            aiReport=aiReport, reportLang=reportLang,dest_pdfPath=output_folder)
    except Exception as e:
        print(e)
    

if __name__ == '__main__':
    main()
