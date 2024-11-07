# Domain Adaption

Domain Adaptation is a method developed to enable a model to use the information learnt during training in a different data distribution when the distributions of training and test data are different.

In particular, source domain and target domain may have different distributions. This is commonly known as domain shift and may adversely affect the performance of the model on target domain. Domain adaptation tries to learn a mapping between source and target domains by taking these differences into account.

## Dataset $^{[1]}$

The dataset used for this experiment is Modern-Office-31, a well-known dataset for domain adaptation, comprising images from four domains: Amazon, Webcam, DSLR, and Synthetic. Each domain contains images from 31 distinct categories representing office items, including objects like keyboards, monitors, trash cans, and more. These domains vary in terms of lighting, background, and object positioning, providing a realistic scenario for domain shift.

In this experiment:

- The Amazon domain, containing high-quality images with consistent backgrounds, was designated as the source domain. It was split into training, validation, and test subsets.
- The Webcam domain, which includes images with lower quality and varied lighting conditions, was chosen as the target domain. This domain was also divided into training, validation, and test sets, though without labels in training for the unsupervised domain adaptation phase.

The Modern-31-Office cluster was not used directly but manually split into train, validation and test subsets. The source domain data was partitioned by sequentially selecting certain ratios, while the target domain data was partitioned so that images of the same object taken from different angles were not in different subsets. In this way, it is aimed to prevent data leakage. This new derived dataset was uploaded to the Kaggle platform.

[Moder-Office-31-Seperated | Kaggle Link](https://www.kaggle.com/datasets/iriscaius/modern-office-31-seperated/data)

## Unsupervised Domain Adaptation by Backpropagation $^{[2]}$

Unsupervised Domain Adaptation by Backpropagation aims to adapt a deep learning model trained on labeled data from one domain (source) to another domain (target) without labels. The key idea is to make the features learned by the model both discriminative for the source task and invariant across domains.

![Gradient Reversal](/images/Gradient%20Reversal%20Model.png)

This is achieved by adding a domain classifier to the network, connected through a unique gradient reversal layer (GRL). The GRL reverses gradients during backpropagation, forcing the network to learn features that make it difficult to distinguish between the source and target domains.

The first model without GRL was trained using only source data. The adversarial model was trained on both source domain and target domain. The evaluation was based on accuracy on both the source and target test datasets. 

- **Source Only Training:** The model without GRL was trained exclusively on the source domain with labeled data.
- **Adversarial Training:** The adversarial model was trained on both the source domain with labels and the target domain without labels, utilizing the adversarial strategy to align feature distributions.

The experiment consisted of the following steps:
1. **Data Preparation:** Images were preprocessed and loaded into training, validation, and test sets for both domains.
2. **Model Definition:** While the first model consists only of the Vision Transformer feature extractor and classifier layers, the adversarial model is implemented as a neural network with a gradient reversal layer (GRL) and a domain classifier in addition to the first model.
3. **Training:** The models was trained for 10 epochs, first with source-only training followed by adversarial training.
4. **Evaluation:** The models' performance was assessed on both the source and target test datasets, focusing on accuracy and confusion matrices.

To start experiment run below code in terminal:

```python.exe .\grad_rev.py```

### Results

|  Training Method  | Source Test Accuracy | Target Test Accuracy |
|-------------------|:--------------------:|:--------------------:|
|    Source Only    |        97.55%        |        85.02%        |
| Gradient Reversal |        96.44%        |        87.22%        |

The results show that the source-only model achieved a higher accuracy on the source test set (97.33%) but demonstrated a lower performance on the target test set (85.02%). The adversarial training method (Gradient Reversal) resulted in a slightly lower accuracy on the source test set (96.44%), and it led to an improvement in the accuracy on the target test set (87.22%).

## Adversarial Discriminative Domain Adaption $^{[3]}$

Adversarial Discriminative Domain Adaptation (ADDA) is a method designed to adapt a model trained on labeled source data to an unlabeled target domain using an adversarial learning strategy. Unlike other domain adaptation techniques that employ tied weights or symmetric mappings, ADDA allows the source and target domains to have different mappings, improving flexibility and effectiveness in handling substantial domain shifts. ADDA’s approach combines a discriminative model with untied weight sharing and a GAN-based loss to learn domain-invariant features in the target domain.

### Method

ADDA first pre-trains a classifier on the labeled source data to learn a discriminative representation for the source domain. Then, a target encoder is trained adversarially using a domain discriminator to align target features with the source feature space, without access to target labels. The domain discriminator is trained to distinguish between the source and target domains, while the target encoder learns to fool the discriminator by mapping target data into the same feature space as the source.

The ADDA training process consists of:

- Source Pre-training: A source encoder and classifier are trained on labeled source data to establish a source feature space.
- Adversarial Training: The target encoder is initialized with the source encoder’s weights and then trained adversarially using a GAN-based objective. The discriminator is trained to classify feature representations as source or target, while the target encoder is updated to fool the discriminator.

The final model consists of the trained source classifier and the target encoder, enabling the model to classify target domain samples without the need for target labels.

![ADDA_Model](/images/ADDA_Model.png)


### Results

|  Training Method  | Source Test Accuracy | Target Test Accuracy |
|-------------------|:--------------------:|:--------------------:|
|    Source Only    |        97.55%        |        85.02%        |
|       ADDA        |        95.32%        |        90.31%        |

The results indicate that ADDA improves the model’s performance on the target domain significantly over the source-only baseline and the gradient reversal approach. The method achieves higher target accuracy by leveraging the untied weights and adversarial training, which more effectively aligns source and target distributions than gradient-based domain adaptation method.

## References

[1]: Ringwald, Tobias and Stiefelhagen, Rainer. "Adaptiope: A Modern Benchmark for Unsupervised Domain Adaptation." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), Jan. 2021, pp. 101-110.

[2]: Ganin, Yaroslav, and Victor Lempitsky. "Unsupervised domain adaptation by backpropagation." *International conference on machine learning.* PMLR, 2015.

[3]: Tzeng, E., Hoffman, J., Saenko, K., & Darrell, T. (2017). Adversarial Discriminative Domain Adaptation.

