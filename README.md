# NTIRE2026_CDFSOD_Freav
solution of team freav in NTIRE2026_CDFSOD contest

Following the steps listed bellow for reproduction:
1) Downloading baseline methods: git clone https://github.com/ohMargin/NTIRE2026_CDFSOD.git
2) Copy all the files into the baseline folder, the files in 'tools' folder should be copied to the corresponding baseline 'tools' folder
3) Preparing all the datasets
4) run 'bash gen_pseudo_all.sh' for generating pseudo gt boxes for all datasets, and replace the original 'k_shot.json' file with the generated annotation file
5) run 'bash main_results.sh' in baseline repository for finetuning
6) (Optional) run 'bash direct_infer.sh' to directly using generated pseudo gts for prediction
