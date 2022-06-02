# Surface Form Competition

This is the repo for reproducing the paper "Surface Form Competition: Why the Highest Probability Answer Isn't Always Right". We adopted scripts from the [original repo](https://peterwestuw.github.io/surface-form-competition-project/) for downloading/processing datasets and running the model, and added features on top of it.

## Dependencies
We use python3 and pytorch 1.7.0, but we do not use cutting-edge features from either and expect to be largely forward and backward compatible. That is not a guarantee or promise.

You can use `pip install -r requirements.txt` to install the required libraries.

## OpenAI Beta
To use GPT-3 you must use OpenAI Beta, which is limited access. You can apply for access [here](https://beta.openai.com/). Once you have access you will need to point the `score.py` to your API key with the `--key` argument or put your key in `api.key` which is the default path. 

## Downloading Datasets

`DATA_README.md` has thorough instructions for downloading and processing datasets. We provide automatic downloaders and processers for datasets where possible in `data_downloaders/` but see `DATA_README` for full instructions.

In addition to the data provide in the automatic downloaders, download folders from https://drive.google.com/drive/u/0/folders/1OW4y1ner_G1z-_hXzIMae-bmuA1yWJkV and put it under a folder named data/original.

## Running Scorers
Once you have a dataset downloaded, running all the zero-shot scoring strategies at once is as simple as:

```
python score.py <dataset abbrevation> --model <model> --seed <seed> --seeds <set of seeds>
```

where **<dataset-abbreviation>** is the abbreviation for a given dataset used for table rows in the paper. **--seed** takes an integer and use it to set the random seed. **--seeds** take a list of integers as argument(separated by white space), and will run the experiment using each integer in the list ads the random seed, this overwrite the -seed argument. If there is any confusion, simply look in `score.py` to see how dataset selection works. **--model** is the name of either a GPT-2 or GPT-3 model e.g. `xl`, `davinci`, etc. To speed things up you can use a larger **--batch** if you have enough GPU memory. 
The ten seeds we used in our experiments are 0 143 447 517 481 18 50 12306 2022 12. Notes that for agnews and trec we currently only support these ten seeds, if you want to use more seeds, take a look at load_examples_sst5 for balanced sampling implementation.
