# Winning Solution of the NeurIPS 2020 Competition on Predicting Generalization in Deep Learning

We present various complexity measures that may be predictive of generalization in deep learning. The intention is to create intuitive and simple measures 
that can be applied post-hoc on any trained model to get a relative measure of its generalization ability.

Our solutions based on consistency, robustness, and separability of representations achieved the highest (22.92) and second-highest (13.93) scores 
on the final phase of the [NeurIPS Competitition on Predicting Generalization in Deep Learning](https://sites.google.com/view/pgdl2020/home?authuser=0). We are **Team Interpex** on the [leaderboard](https://sites.google.com/view/pgdl2020/leaderboard?authuser=0).

Detailed descriptions of our solution can be found in our paper:

```
@misc{natekar2020representation,
      title={Representation Based Complexity Measures for Predicting Generalization in Deep Learning}, 
      author={Parth Natekar and Manik Sharma},
      year={2020},
      eprint={2012.02775},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

Our solution is based on the quality of internal representations of deep neural networks, inspired by neuroscientific theories on how the human visual system creates invariant object representations.

To run our solution on the public task of the PGDL Competition, clone this repository to the ```sample_code_submission``` folder of the PGDL directory, then run 
```python3 ../ingestion_program/ingestion.py``` from this folder.

# Available Measures 

The following complexity measures are currently available:

1. Davies Bouldin Index 
2. Mixup Performance
3. Perturbed Margin
4. Manifold Mixup Performance
5. Frobenius/Spectral Norm 

The following measures are in the pipeline: FisherRegret, Silhouette Coefficient, Ablation Performance, Pac Bayes, Noise Attenutation.

# Results

Scores of our final measures on various tasks of PGDL are as follows:

|              MEASURE              |   | CIFAR-10 |  SVHN | CINIC-10 | CINIC-10 (No BatchNorm) | Oxford Flowers | Oxford Pets | Fashion MNIST | CIFAR 10 (With Augmentations) |
|:---------------------------------:|:-:|:--------:|:-----:|:--------:|:-----------------------:|:--------------:|:-----------:|:-------------:|:-----------------------------:|
| Davies Bouldin * Label-Wise Mixup |   |   25.22  | 22.19 |   31.79  |          15.92          |      43.99     |    12.59    |      9.24     |             25.86             |
|            Mixup Margin           |   |   1.11   | 47.33 |   43.22  |          34.57          |      11.46     |    21.98    |      1.48     |             20.78             |
|           Augment Margin          |   |   15.66  | 48.34 |   47.22  |          22.82          |      8.67      |    11.97    |      1.28     |             15.25             |


Currently the code only works in the PGDL starting-kit framework available at <https://competitions.codalab.org/competitions/25301#learn_the_details-get_starting_kit>. You will need to manually choose the required measure in the ```complexity.py``` file and then run ```python3 ../ingestion_program/ingestion.py``` as mentioned above.




