**Can we "see" a cyberattack?**

This project explores the intersection of Cybersecurity and Deep Learning by redefining network flow analysis as an image classification problem. By transforming raw network traffic into visual representations, I leveraged Concatenated CNNs (Binary + Multiclass) to extract complex spatial features, achieving a **99.14% accuracy** rate on independent test sets—outperforming traditional numerical methods. 

To tackle the common challenge of class imbalance in security datasets, I implemented a **Conditional GAN (C-GAN)** to synthesize 75,000 high-fidelity network flow samples via adversarial training. 

This work demonstrates that a "vision-first" approach to NIDS provides superior feature extraction and more robust detection capabilities against modern cyber threats.
