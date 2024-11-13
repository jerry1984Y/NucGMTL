# Identifying Protein-nucleotide Binding Residues via Grouped Multi-task Learning and Pre-trained Protein Language Models

The accurate identification of protein-nucleotide binding residues is crucial for protein function annotation and drug discovery. Numerous computational methods have been proposed to predict these binding residues, achieving remarkable performance. However, due to the limited availability and high variability of nucleotides, predicting binding residues for diverse nucleotides remains a significant challenge. To address these, we propose NucGMTL, a new grouped deep multi-task learning approach designed for predicting binding residues of all observed nucleotides in the BioLiP database. NucGMTL leverages pre-trained protein language models to generate robust sequence embedding and incorporates multi-scale learning along with scale-based self-attention mechanisms to capture a broader range of feature dependencies. To effectively harness the shared binding patterns across various nucleotides, deep multi-task learning is utilized to distill common representations, taking advantage of auxiliary information from similar nucleotides selected based on task grouping. Performance evaluation on benchmark datasets shows that NucGMTL achieves an average area under the Precision-Recall curve (AUPRC) of 0.594, surpassing other state-of-the-art methods. Further analyses highlight that the predominant advantage of NucGMTL can be reflected by its effective integration of grouped multi-task learning and pre-trained protein language models. The dataset and source code are freely accessible at: https://github.com/jerry1984Y/NucGMTL.

The pre-trained model is at http://pan.njust.edu.cn/#/link/P645H8uoTMo2pIBJMoNe

# 1. Requirements
Python >= 3.10.6

torch = 2.0.0

pandas = 2.0.0

scikit-learn = 1.2.2

ProtTrans (ProtT5-XL-UniRef50 model)

EMS2 (esm2_t33_650M_UR50D model) 

# 2 Datasets
We provided a total of three benchmark datasets, namely Nuc-1892, Nuc-798, and Nuc-849. Among them,  Nuc-1892 includes five frequently observed nucleotides (ATP, ADP, AMP, GTP, and GDP) and ten infrequently observed nucleotides (TMP, CTP, CMP, UTP, UMP, UDP, IMP, GMP, CDP, and TTP) binding proteins; Nuc-798 and Nuc-849 each consist of five common nucleotide (ATP, ADP, AMP, GTP, GDP) binding proteins constructed at different times.

# 3. How to use
## 3.1 Set up environment for ProtTrans and EMS2
Set ProtTrans follow procedure from https://github.com/agemagician/ProtTrans/tree/master
Set EMS2 follow procedure from https://github.com/facebookresearch/esm


## 3.2 Extract features
Extract pLMs embedding: cd to the NucGMTL/FeatureExtract dictionary.


run "python3 extract_prot1024.py", the pLMs embedding matrixs generated from ProtTrans model will be extracted to embedding/prot_embedding folder.


run "python3 extractdata_esm1280.py", the pLMs embedding matrixs generated from ESM2 model will be extracted to embedding/esm_embedding1280 folder.

## 3.3 Train and test
cd to the NucGMTL home dictionary.  
run "python3 NucGMTL.py" for training and testing the model for all nucleotides binding residues prediction in Nuc-1892.  
 

## 3.4 Only for nucleotides binding residues prediction purpose
1. download the pre-trained model (in http://pan.njust.edu.cn/#/link/P645H8uoTMo2pIBJMoNe) to pre_model folder；
 
 pre_model folder   
   |--   task_GDP   
&nbsp;&nbsp;&nbsp;&nbsp;|--   Result_GDP_ADP_ATP_GTP_0_2024-06-17 11_40_15.252408.pkl   
&nbsp;&nbsp;&nbsp;&nbsp;|--  Result_GDP_ADP_ATP_GTP_1_2024-06-17 11_40_15.252408.pkl   
&nbsp;&nbsp;&nbsp;&nbsp;|--  ……   
   |--  task_ADP_ATP_AMP  
&nbsp;&nbsp;&nbsp;&nbsp;|--  
   |--  task_CDP  
&nbsp;&nbsp;&nbsp;&nbsp;|--  
   |--  task_GMP  
&nbsp;&nbsp;&nbsp;&nbsp;|--  
   |--  task_IMP  
&nbsp;&nbsp;&nbsp;&nbsp;|--  
   |--  task_TTP  
&nbsp;&nbsp;&nbsp;&nbsp;|--  
   |--  task_UMP  
 &nbsp;&nbsp;&nbsp;&nbsp;|--  
   |--  task_UTP  
 &nbsp;&nbsp;&nbsp;&nbsp;|--  
   |--  task_CMP  
&nbsp;&nbsp;&nbsp;&nbsp;|--  
   |--  task_CTP  
&nbsp;&nbsp;&nbsp;&nbsp; |--  
   |--  task_TMP  
&nbsp;&nbsp;&nbsp;&nbsp;|--  
   |--task_GTP  
&nbsp;&nbsp;&nbsp;&nbsp; |--  

2. write your query sequence (once one sequence) in file with file name 'query.txt' like below：
   
   TTVAQILKAKPDSGRTIYTVTKNDFVYDAIKLMAEKGIGALLVVDGDDIAGIVTERDYARKVVLQERSSKATRVEEIMTAKVRYVEPSQSTDECMALMTEHRMRHLPVLDGGKLIGLISIGDLVKSVIADQQFTIS  
   

   and put the query.txt into customer_test folder.
   
   
