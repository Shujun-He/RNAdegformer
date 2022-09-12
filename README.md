# RNAdegformer


Source code to reproduce results in the paper "RNAdegformer: Accurate Prediction of mRNA Degradation at Nucleotide Resolution with Deep Learning".

<p align="center">
  <img src="https://raw.githubusercontent.com/Shujun-He/RNAdegformer/main/graphics/RNAdegformer.png?token=GHSAT0AAAAAABRGHIRII6LFDFJ7KPWZWBEAYY6X33Q"/>
</p>


## How to use the models

I also made a web app to use the models. Check it out at [https://github.com/Shujun-He/Nucleic-Transformer-WebApp](https://github.com/Shujun-He/RNAdegformer-Webapp)

### Home page
![home_page](https://github.com/Shujun-He/RNAdegformer-Webapp/blob/main/files/home_page.png)



### RNA degradation prediction
In this page you can predict RNA degradation at each nucleotide and visualize the attention weights of the Nucleic Transformer

![RNA degradation](https://github.com/Shujun-He/RNAdegformer-Webapp/blob/main/files/rnapage.png)


## Requirements
I included a file (environment.yml) to recreate the exact environment I used. Since I also use this environment for computer vision tasks, it includes some other packages as well. This should take around 10 minutes. After installing anaconda:


```
conda env create -f environment.yml
```

Then to activate the environment

```
conda activate torch
```

Additionally, you will need Nvidai Apex: https://github.com/NVIDIA/apex

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install .
```

Also you need to install the Ranger optimizer

```bash
git clone https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
cd Ranger-Deep-Learning-Optimizer
pip install -e .
```

## Repo file structure

The src folder includes all the code needed to reproduce results in the paper and the OpenVaccine competition. Additional instructions are in the folder


```src/OpenVaccine``` includes all the code needed to run a ten-fold model for the openvaccine dataset



## Datasets

### OpenVaccine dataset

For original dataset, see https://www.kaggle.com/c/stanford-covid-vaccine/data

In addition to the secondary structure features given by Das Lab, I also generated additional secondary structure features at 2 temperatures with 6 biophysical packages (12x), for these features, see https://www.kaggle.com/shujun717/openvaccine-12x-dataset
