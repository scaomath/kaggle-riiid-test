# Kaggle Riiid Test Answer Correctness
Playground for [Kaggle competition Riiid! Answer Correctness Prediction](https://www.kaggle.com/c/riiid-test-answer-prediction)

## EDA, observations, ideas
- In the inference `iter_env`, the correct target will be updated to `previous_test_df` even when the current prediction is wrong. Adding rolling mean type features is possible.
- In the CV tito and marisakamozz proposed, the test iterator df is sorted by `virtual_timestamp`, which is nice.
- Currently many features do `fillna` using the mean for all users, how about the `fillna` just for one user? 
- How to address the fact that a random guess would yield 25% correct rate?
- How to do cross-validation using a `KFold` or a stratified folds in the current setting?


## Transformer encoder-based

- (Dec 9) Increasing `seq_len` for SAKT new model does not work ~~well~~ as intended.

- (Dec 9) Testing performance of the following configs for both attention layers and after concat att outputs:
    1. bn(relu(f(x))) + x, epoch 1 auc 0.7372, epoch 3 auc 0.7422, epoch 5 auc 0.7445 
    2. bn(relu(f(x)) + x), epoch 1 auc 0.7379, epoch 3 auc 0.7413, epoch 5 auc 0.7443
    3. bn(f(x)) + x: epoch 0 auc 0.7369, epoch 2 auc 0.7415, epoch 4 auc 0.7448
    4. bn(f(x) + x): epoch 0 auc 0.7380, epoch 2 auc 0.7418, epoch 4 auc 0.7445

- (Dec 10) Testing a new model: two attention layers stacked using `question_id` as key
epoch 0 auc 0.7256, epoch 2 auc 0.7399, epoch 4 auc 0.7424. Later epoch does not perform well.

- (Dec 10) Tried a multi-embedding model, still using `question_id` as key, summing up the embeddings of `user`, `prior_question_elapsed_time`, `prior_questions_had_explained`, 0.7563 best, then deteriorate.

- (Dec 10) Multi-embedding SAKT, cat 3 outputs from 3 multi-attention, FNN from `3*n_embed` to `1*embed`, skip connection with the output from the first attention layer. CV 0.7575

- (Dec 10) Multi-embedding SAKT, cat 2 outputs from 2 multi-attention, FNN from `2*n_embed to 1*embed`, skip connection with the output from the first attention layer. CV 

- (Dec 11) Testing a warm-up scheduler with 10 warm-up epochs for a model with two attention layers (no significant improvement CV 7570)

- (Dec 13) Baseline SAKT, a 2-layer attention with shared weights
Change the last several layers to embed_dim->embed_dim//2 ->1, deleted the skip connection.
seq len=150, embed_dim = 256 with 8 heads, CV 0.7577; iter_env CV 0.7270

- (Dec 13) Baseline SAKT, 1 layer attention, no skip connection with layer norm, seq len 150, embed_dim 128, 8 heads, CV scaling = 2, LR = 1e-3, CV 0.7604; iter_env CV 0.7291

- (Dec 15) Baseline SAKT, 1 layer attention, no skip connection with layer norm, seq len 150, embed_dim 128, 8 heads, CV scaling = 3.5, LR = 1e-4 with a scheduler, CV 0.7552, (deleted)

- (Dec 15) One layer attention, seq len 150, embed_dim 128, 8 heads, CV scaling = 2, LR = 1e-3, CV 0.7605; iter_env CV 0.7318
If there is layer normalization, multiplying with a scaling factor does not matter much for AUC

- (Dec 15) Same with 6, embed_dim 160, head 10, CV scaling 2, CV 0.7579

- (Dec 18) Same with 6 Label smoothing with a factor of 0.2, CV scaling = 4, CV < 0.73....

- (Jan 3) Finally found the bug of the transformer-based model, the `train_loader` is way too few to cater the inference pipeline. Done the shifted training sequence and the CV improved to 0.779, LB unchanged 0.776. Ensembled LB increased to 0.788.


TO-DO:
- [x] Test label smoothing using (a) a simple label change, then multiply a factor to the prediction. Does not work well.
- [x] Testing add a separate `previous_answers_correct` embedding
- [ ] Testing adding a "User growth" feature to the embedding...(maybe not very good as the attention-based model should be more automatic)
- [ ] Using a multitarget with the second target being the LGBM oof-prediction/other things.
- [x] Testing a new training strategy using a shifted sequence.
- [ ] Disable `shuffle` in `train_loader` see if the the CV gets better. 


## LightGBM models

- (Dec 18) Testing version of LGB feat gen as of Dec 18

- (Dec 24) Baseline (baseline file), debugging ver (first 12m rows), CV 0.7759, iter_env CV: 0.744681 (7 runs of 100k rows in `cv5_valid`), LB 0.776

- (Dec 25) Added `rolling_w_mean` where `w` is the window size, local CV increased to 0.7784, still working on adding this feature to the inference.

- (Dec 26) Added `rolling_mean` for target shifted by 1 (i.e., previous question correct or not). iter_env CV: 0.7477 (y runs of 50k rows in `cv5_valid`), LB 0.766

- (Dec 29) First full model submit. LB 0.776.

- (Jan 4) LGB loop feature model: after dropping `train[train.content_id==1]`, some users have a starting `timestamp` nonzero, because this user may watch a lecture as the first interaction.

### Features TO-DO:

- [ ] Commonness/difficulty rating of the questions
- [ ] difficulty-weighted interaction time/gap time
- [x] Rolling mean of previous $k$ questions correct or not 
- [ ] Rolling mean of a feature characterizing the user's streak or readiness/status
- [ ] ELO rating of the users



~~## NN and deeper models~~
~~-  TabNet baseline: no time and inference time limit may exceed.~~
