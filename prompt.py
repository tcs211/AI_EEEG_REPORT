############################################
# This file contains the prompt messages
# for the EEG report generation task.
############################################

def reportPrompt(finalResults, reportLang='English', promptLength='long'):
    if promptLength=='long':
       
        message="""You are a neurologist with access to a comprehensive neurological database from Medline and textbooks. 
            JSON Findings:
            {}
            Meanings of JSON findings:
            'backgroundFrequency': the background frequency of the EEG in the left and right hemispheres. Differences <=0.5 Hz are considered symmetric.
            'bg_active': the overall presence of background activity.
            'bg_amp_sym', 'bg_freq': symmetry of alpha amplitude and background frequency.            
            'abnormalFindings': all abnormal findings in the EEG examination.

            Task: Generate a detailed and structured EEG report based on the provided EEG findings.
            === EEG Findings ===
            Present the EEG findings in a list format in concise and professional language.
            === Conclusion ===
            - If the results are normal: "The EEG examination reveals normal findings."
            - If the results are abnormal: "Abnormal EEG findings are observed."
            - A brief description of the abnormal observations should be included.
            === Clinical Correlation ===
            - For normal EEG results: "No evidence of cortical dysfunction or epileptiform activity is observed."
            === Advanced Strategies ===
            - - For abnormal EEG results, very briefly suggest further investigations or follow-up tests.

            Examples for Abnormal Clinical Correlations:
            - Diffuse background slowing may indicate associations with degenerative diseases, metabolic encephalopathy, and bilateral cortical lesions.
            - Slow waves in a specific region or hemisphere might suggest a structural lesion in the corresponding brain region.
            - Detection of spikes or sharp waves raises concerns about an increasing risk of epilepsy.
            - Excess beta activity is linked to factors like anxiety or the effects of certain drugs, such as benzodiazepines.

        Examples for Advanced Strategies:
            - For structural lesions: Recommend neuroimaging studies, such as MRI or CT, to identify abnormalities.
            - In cases of epilepsy: Suggest long-term video EEG monitoring for detecting epileptiform activity.
            - In the presence of artifacts: Advise a repeat EEG examination to confirm findings.
            - Emphasize that correlation with clinical symptoms and other laboratory tests is essential for establishing a diagnosis.

            Important Notes:
                - Must Enclose the title of each section in ===

            Report: 
            Your detailed and structured EEG report in {}.
        """.format(finalResults, reportLang)
    elif  promptLength=='medium':
        message="""
            You are a neurologist with access to a comprehensive neurological database from Medline and textbooks. 
            JSON Findings:
            {}
            Meanings of JSON findings:
            'backgroundFrequency': the background frequency of the EEG in the left and right hemispheres. Differences <=0.5 Hz are considered symmetric.
            'bg_active': the overall presence of background activity.
            'bg_amp_sym', 'bg_freq': symmetry of alpha amplitude and background frequency.            
            'abnormalFindings': all abnormal findings in the EEG examination.

            Task: Generate a detailed and structured EEG report based on the provided EEG findings.
            === EEG Findings ===
            Present the EEG findings in a list format in concise and professional language.
            === Conclusion ===
            - A brief description of the normal or abnormal observations.
            === Clinical Correlation ===
            === Advanced Strategies ===
            - - For abnormal EEG results, very briefly suggest further investigations or follow-up tests.
            Report: 
            Your detailed and structured EEG report in {}.
        """.format(finalResults, reportLang)
    elif promptLength=='short':
        message="""
            You are a neurologist. 
            Task: Generate a EEG report based on the provided EEG findings.
            EEG Findings, Conclusion, Clinical Correlation, and Advanced Strategies should be included.
            JSON Findings:
            {}
            Meanings of JSON findings:
            'bg_active': presence of background activity.
            'bg_amp_sym', 'bg_freq': symmetry of background alpha amplitude and  frequency.            
            'abnormalFindings': all abnormal findings in the EEG examination.
            Your EEG report in {}:
        """.format(finalResults, reportLang)

    return message

def validatePrompt(text):
    prompt="""
    Task: Read the EEG report and answer the following questions, answering 1 for yes and 0 for no.
    Questions:
    a.Does the report mention "diffuse background slowing" or "Increased background slow waves ratio diffusely"?
    b.Dose the report mention background asymmetry such as "lower amplitude" or "lower frequency" in right or left hemisphere, or "focal/regional slowing(delta/theta)"?
    Important Note: 
    1.if the left and right background frequency differ by within 1 Hz, it is considered as normal.
    2. Excessive beta activity is not classified as abnormal diffuse background slowing nor focal slow wave.
    The EEG report:
    {}
    Your answer should be in array format ([int, int]), and do not include any other information.
    Your answer array:
    """.format(text)
    
    return prompt