We use _gsat_ to mark the nodes as a preprocessing step to enable the biased infomax. To mark the nodes to be biased, go to `example/` and run the following command:

```
python run.py --dataset goodhiv --shift covariate --domain scaffold --layer 4 --device 0 --epoch 100
```

The code will generate a dictionary of tensors with different thresholds as masking tensors, and save it to `../explain_res/` for future usage of biased infomax.
