# Winning Solution of the NeurIPS 2020 Competition on Predicting Generalization in Deep Learning

We present various complexity measures that may be predictive of generalization in deep learning. The intention is to create intuitive and simple measures 
that can be applied post-hoc on any trained model to get a relative measure of its generalization ability.

Our solutions based on consistency, robustness, and separability of representations achieved the highest (22.92) and second-highest (13.93) scores 
on the final phase of the [NeurIPS Competitition on Predicting Generalization in Deep Learning](https://sites.google.com/view/pgdl2020/home?authuser=0). We are **Team Interpex** on the [leaderboard](https://sites.google.com/view/pgdl2020/leaderboard?authuser=0).

Our solution is based on the quality of internal representations of deep neural networks, inspired by neuroscientific theories on how the human visual system creates invariant object representations.

To run our solution on the public task of the PGDL Competition, clone this repository to the ```sample_code_submission``` folder of the PGDL directory, then run 
```python3 ../ingestion_program/ingestion.py``` from this folder.

The following complexity measures are currently available:

1. Davies Bouldin Index 
2. Mixup Performance
3. Perturbed Margin
4. Manifold Mixup Performance
5. Frobenius/Spectral Norm 

The following measures are in the pipeline: FisherRegret, Silhouette Coefficient, Ablation Performance, Pac Bayes, Noise Attenutation.

Currently the code only works in the PGDL starting-kit framework available at <https://competitions.codalab.org/competitions/25301#learn_the_details-get_starting_kit>. You will need to manually choose the required measure in the ```complexity.py``` file and then run ```python3 ../ingestion_program/ingestion.py``` as mentioned above.




