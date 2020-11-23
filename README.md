# `AMAZE`: Astrophysical Model Analysis & Evidence Evaluation
A codebase for performing multi-channel model selection using catalogs of compact binaries

### Authors:
Michael Zevin, Chris Pankow
  
  
### Papers:
https://ui.adsabs.harvard.edu/abs/2017ApJ...846...82Z/abstract
https://ui.adsabs.harvard.edu/abs/2020arXiv201110057Z/abstract


Why use one channel when you can use them all? `AMAZE` performs hierarchical inference on branching fractions between any number of population models, where each channel can also be parameterized by physical prescriptions. The executable `model_select` performs the inference, and has many options for including different channels, specifying whether to use mock observations or actual gravitational-wave observations, specifying the prescription for measurement uncertainty, etc. Run `python model_select --help` to learn more about all these options. 

Included in this codebase are a number of notebooks (in the `notebooks/` directory) that were used to pre-processing the data (`process_unweighted_data.ipynb`, `process_GWTC_data.ipynb`) and generate all the figures and numbers (`paper_plots.ipynb`) from Zevin et al. 2020 (https://ui.adsabs.harvard.edu/abs/2020arXiv201110057Z/abstract). Data from this project, including the processed public GW data, processed population models, and inference output, are available on Zenodo (https://zenodo.org/record/4277620#.X7w28RNKjUI). 
