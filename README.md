Implementation of a Network Intrusion Detection System based on images. The main idea behind this project was to leverage CNNs' ability to extract complex features from visual samples for application in NIDS environment, and the goal was to prove that image based approach could outperform numerical data methods in classifying network flows. The second goal was to solve the dataset imbalances by implementing a Conditional GAN able to generate, through an adversarial training, conditioned images on specific class characteristics that resemble the network flows from the dataset.

A Concatenated CNN, composed by binary and multiclass CNN, was created, able to reach 99.14% of accuracy over a different dataset from the training one. Through the C-GAN it was possible to obtained new 75000 samples of network flows.

This work showed the advantages of using visual samples in NIDS environment rather than using numerical based approaches.
