mkdir data
cd data
kaggle competitions download -c stanford-covid-vaccine
kaggle datasets download shujun717/openvaccine-12x-dataset
unzip stanford-covid-vaccine.zip
unzip openvaccine-12x-dataset.zip
mv openvaccine_12x_dataset train_test_bpps
cd ..
