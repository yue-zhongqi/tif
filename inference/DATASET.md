# How to install datasets

We suggest putting all datasets under the same folder (say `$DATA`) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like

```
$DATA/
|–– fgvc_aircraft/
|–– duke/
|–– veri/
|–– isic2019/
```

### FGVCAircraft
- Download the data from https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz.
- Extract `fgvc-aircraft-2013b.tar.gz` and keep only `data/`.
- Move `data/` to `$DATA` and rename the folder to `fgvc_aircraft/`.

The directory structure should look like
```
fgvc_aircraft/
|–– images/
|–– ... # a bunch of .txt files
```

### Other datasets
- Download ISIC2019 and put in ./data/ISIC_2019_Training_Input
- Download DukeMTMC-reID and put in ./data/DukeMTMC-reID
- Download VeRi and put in ./data/VeRi
- Run process_*.py to process all three datasets
- We use run_gen_data_splits to generate fixed data splits to evaluate baselines and our method. Please refer to the splits folder in the lora project.