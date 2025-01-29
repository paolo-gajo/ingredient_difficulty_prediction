# Predicting the difficulty of GialloZafferano dishes

This repo contains data extracted from GialloZafferano and the Python script `script.py` to scrape said data.

 `gz_all.json` is as collection of 7,247 recipes, used for training and validation. `gz_101.json` is used as a test set and contins 101 recipes.

The test set corresponds to the dataset compiled by Fossemò et al. (2022):

```
@inproceedings{fossemo2022using,
  title={Using inductive logic programming to globally approximate neural networks for preference learning: challenges and preliminary results},
  author={Fossem{\`o}, Daniele and Mignosi, Filippo and Raggioli, Luca and Spezialetti, Matteo and D'Asaro, Fabio Aurelio and others},
  booktitle={CEUR WORKSHOP PROCEEDINGS},
  pages={67--83},
  year={2022},
  url={https://ceur-ws.org/Vol-3319/paper7.pdf}
}
```