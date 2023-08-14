



# Modeling Life-span Brain Age from Large-scale Dataset based on Multi-level Information Fusion

This is a PyTorch implementation of the paper "Modeling Life-span Brain Age from Large-scale
Dataset based on Multi-level Information Fusion", June, 2023.

## 1. Installation

(1) Create conda env and install pytorch

```bash
conda create -n brain python=3.9
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

(2) Install relevant libraries

```bash
pip install -r requirements.txt
```

## 2. Network Architecture

<img src="http://sailonzn.test.upcdn.net/net_c.png" alt="net" style="zoom: 30%;" />



## 3. Data Distribution

<center><b>Demographic information of 8 cohorts</b></center>

<img src="http://sailonzn.test.upcdn.net/table.png" alt="Screenshot from 2023-08-05 19-04-52" style="zoom: 40%;" />

| Age Distribution on Healthy Controls | Age Distribution on Brain Disorders |
| :----------------------------------: | :---------------------------------: |

<center class="half">    
  <img src="http://sailonzn.test.upcdn.net/HCs_Age_Distribution_8_site_stack.png" alt="Age Distribution on Healthy Controls" style="zoom:40%;"/>
  <img src="http://sailonzn.test.upcdn.net/BDs_Age_Distribution_8_site_stack.png" alt="Age Distribution on Brain Disorders" style="zoom:40%;"/> 
</center>




## 4. Prediction Performance

| Predictions on Healthy Controls | Predictions on Brain Disorders |
| :-----------------------------: | :----------------------------: |

<center class="half">    
  <img src="http://sailonzn.test.upcdn.net/test_HCs_ours_SFCN.png" alt="Predictions on Healthy Controls" style="zoom:40%;"/>
  <img src="http://sailonzn.test.upcdn.net/BDs_BAG_ours_train_val_test_part.png" alt="Predictions on Brain Disorders" style="zoom:40%;"/> 
</center>


## 5. Train the Model

To train the model, run `train_threedim_3view_GAF.py` file provided in the repository.

```bash
batch_size=8
learning_rate=0.001
weight_decay=0.0001
n_epochs=200
n_exps=1 # num of independent experiments

# ============= Training and Parameter Configuration ==============
python train_threedim_3view_GAF.py                       \
--batch_size        $batch_size                          \
--lr_s              $learning_rate                       \
--wd_s              $weight_decay                        \
--n_epochs          $n_epochs                            \
--n_exps            $n_exps                              \

```

## 6. Test on Brain Disorders

To test the model on brain disorders, run `test_BDs.py`.

```bash
python test_BDs.py
```

