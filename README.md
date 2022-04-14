# PERFECT: Prompt-free and Efficient Few-shot Learning with Language Models
This repo contains the PyTorch implementation of Rabeeh Karimi Mahabadi, Luke Zettlemoyer, james Henderson, Marzieh Saeidi, Lambert Mathias, Veselin ‪Stoyano, and Majid Yazdani [PERFECT: Prompt-free and Efficient Few-shot Learning with Language Models](), ACL 2022.

For any questions, please contact the first author([email](mailto:rkarimi@idiap.ch)) or leave issues.





# Installation 
```
conda create --name perfect python=3.8
python setup.py develop 
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

# data pre-processing 
For SST-2, SST-5, CR, MR, Subj, TREC datasets, we used the datasets from Gao et al. ACL 2021 ([paper][lm-bff-paper]),
which are processed as below, for other datasets we used the huggingface datasets, which automatically get downloaded:
```
wget https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar
tar xvf datasets.tar
rm datasets.tar 
mv original/ datasets
python process_datasets.py
rm -r datasets 
```

# How to run the models
We provide the example scripts to run each model in the paper in `fewshot/scripts`
folder with their config files in `fewshot/configs`. To run the models, please do `cd fewshot` and run:
**Please note on top of each script, I wrote how we modified the hyper-parameters**

  #### Reproducing results in table 1 
  Perfect results
  ```
  bash scripts/perfect.sh
  ```
  Finetune results:
   ```
   bash scripts/finetune.sh
   ```
  PET
  ```
  bash scripts/pet.sh 
  ```  
  [Logan IV et al](https://arxiv.org/pdf/2106.13353.pdf)'s results
  ```
  bash scripts/loganIV.sh
  ```
  Prompt+mte ablation 
  ```
  bash scripts/prompt_mte_ablation.sh
  ```
  bitfit+mte ablation results:
  ```
  bash scripts/bitfit_mte_ablation.sh
  ```
  perfect+init ablation results: 
  ```
  scripts/perfect_init_ablation.sh
  ```

  #### Reproducing results in table 3
  Pattern-Free ablation results
  ```
  bash scripts/pattern_free.sh
  ```

  #### Reproducing results in table 4
  ```
  bash scripts/perfect_without_adapters_ablation.sh
  ```

  #### Reproducing results in table 5
  ```
  bash scripts/perfect_num_masks_ablation.sh
  ```


  #### Reproducing results in table 7
  ```
  bash scripts/perfect_mask_position_ablation.sh
  ```

  #### Reproducing results in table 8
  ```
  bash scripts/perfect_init_range_ablation.sh 
  ```

  #### Reproducing results in table 9 
  Hinge loss ablation 
  ```
  bash scripts/perfect_hinge_loss_ablation.sh
  ```
  +Label Embed ablation 
  ```
  bash scripts/perfect_label_embed_ablation.sh
  ```
  -Prototypical ablation 
  ```
  bash scripts/perfect_prototypical_ablation.sh
  ```


# Bibliography 
If you find this repo useful, please cite our paper.

```
@inproceedings{karimi2022perfect,
  title={PERFECT: Prompt-free and Efficient Few-shot Learning with Language Models},
  author={Karimi Mahabadi, Rabeeh and Zettlemoyer, Luke and Henderson, James and Saeidi, Marzieh and Mathias, Lambert and ‪Stoyano, Veselin and Yazdani, Majid},
  booktitle={Annual Meeting of the Association for Computational Linguistics},
  year={2022}
}
```




[lm-bff-paper]: https://arxiv.org/abs/2012.15723
