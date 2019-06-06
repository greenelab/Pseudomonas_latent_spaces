# Latent space interpolation

**Background**
Chronic Pseudomonas aeruginosa infections are a serious problem to patients with Cystic fibrosis (CF). These chronic infections lead to persistent inflammation which damages the lungs and ultimately results in respiratory failure, which is the leading cause of premature death in CF patients.

We know that the the mucus layer in CF patient airways have lower oxygen concentration.  *P. aeruginosa* can sense and respond to low oxygen levels through the activity of the transcription factor called Anr, which directly regulates genes in a threshold manner.  Additionally, Anr was found to regulate the expression of pathways that are associated with persistent *P. aeruginosa* infection.   

**Question:**
Can we determine the mechanistic role of Anr in driving Pa persistent infection?  In particular, can we model the nonlinear activation of Anr-regulated transcriptional pathways that are associated with Pa persistence?


**Hypothesis:**
A variational autoencoder (VAE) can model nonlinear activation of direct and indirect Anr-dependent transcriptional pathways that associated with Pa persistence.

**Dataset**
Dataset downloaded from ADAGE repository ADAGE. This dataset is publically available P. aeruginosa gene expression data from ArrayExpress. This collection contains ~100 microarray experiments under various conditions. here are approximately ~1K samples with ~5K genes. There are 976 planktonic and 138 biofilm samples.
