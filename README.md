# Neural Translation
## Intro
This is neural translation project for machine learning. Deadline 2021.6.5.
## Project Reference
1. Torch NLP tutorial: (Being tested by CZY) https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
2. Fast project: https://www.fast.ai/2019/07/08/fastai-nlp/
3. NLP courses: https://github.com/yandexdataschool/nlp_course
4. Useful Pytorch tutorial：[bentrevett/pytorch-seq2seq: Tutorials on implementing a few sequence-to-sequence (seq2seq) models with PyTorch and TorchText. (github.com)](https://github.com/bentrevett/pytorch-seq2seq)

## Milestone:
- [x] Inport multi30K data and having build the dictionary.
- [ ] Build the seq2seq network.
- [ ] Training on small batch data.
- [ ] Training on the whole dataset.
- [x] Set up BLEU test.
- [ ] Other: using other method like transformer layer in nn to train.

## TODO list
1. Check the dataset in torchnlp is the same as requirement.
2. Look for better computation resources.
3. Currently we use RNN+ATN, find the improvement.
4. bmm in reality, how batch size implemented in network?
5. Can it predict different length sentence?