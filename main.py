from util.config import *
from model.model import *
import tokenization
from dataset import *
from torch.utils.data import DataLoader
import optimizer
from tqdm import tqdm

modelConfig = ModelConfig.parse_json('config/model_config.json')
trainConfig = TrainingConfig.parse_json('config/train_config.json')
evalConfig= EvaluateConfig.parse_json('config/eval_config.json')
set_seeds(modelConfig.seed)
tokenizer = tokenization.FullTokenizer(vocab_file=trainConfig.vocab, do_lower_case=True)
if modelConfig.mode in ["train","train/eval"]:
    dataset_class, train_size, train_path, eval_path = get_dataset(trainConfig.task)
else:
    dataset_class, train_size, train_path, eval_path = get_dataset(evalConfig.task)

step_per_epoch = int(train_size/trainConfig.batch_size)
pipeline = [Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
            AddSpecialTokensWithTruncation(modelConfig.seq_len),
            TokenIndexing(tokenizer.convert_tokens_to_ids, dataset_class.labels, modelConfig.seq_len)]

if modelConfig.mode in ["train","train/eval"]:
    # enter training stage
    dataset = dataset_class(train_path, pipeline)
    data_iter = DataLoader(dataset, batch_size=trainConfig.batch_size, shuffle=True, drop_last=True)
    model = BERTClassifier(modelConfig, len(dataset_class.labels))
    model.train()
    device = get_device()

    print("loading model...")
    model = load(model, pretrain_file=trainConfig.pretrain_model_path).to(device)
    if trainConfig.data_parallel: # use Data Parallelism with Multi-GPU
        model = nn.DataParallel(model)
    optimizer = optimizer.optim4GPU(trainConfig, model, step_per_epoch)
    step = 0
    print("start training...")
    for epoch_n in range(trainConfig.epoch):
        loss_sum = 0
        iter_bar = tqdm(data_iter, desc='Iter (loss=X.XXX)')
        for iter_n, batch in enumerate(iter_bar):
            batch = [data.to(device) for data in batch]
            optimizer.zero_grad()
            loss, model_loss, ib_loss = get_loss(model, batch, trainConfig.kl_fac, modelConfig)  # mean() for Data Parallelism
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            step += 1
            loss_sum += loss.item()
            iter_bar.set_description('Iter (loss=%5.3f)' % loss.item())
        save_model(model, step, trainConfig, modelConfig)
        print(f'Epoch {epoch_n + 1}/{trainConfig.epoch} : Average Loss {loss_sum / (iter_n + 1)} model loss {model_loss}'
              f' ib_loss {ib_loss}')

if modelConfig.mode in ["eval","train/eval"]:
    dataset = dataset_class(eval_path, pipeline)
    device = get_device()
    if modelConfig.mode == "eval":
        train_eval_config_switch = evalConfig
    elif modelConfig.mode == "train/eval":
        train_eval_config_switch = trainConfig
    for i in range(train_eval_config_switch.epoch):
        print(f'this is epoch {i}')
        current_step = (i + 1) * step_per_epoch
        model_path = get_eval_path(train_eval_config_switch, modelConfig, current_step)
        data_iter = DataLoader(dataset, batch_size=trainConfig.batch_size, shuffle=True, drop_last=True)
        model = BERTClassifier(modelConfig, len(dataset_class.labels))
        model.eval()
        model = load(model, model_file=model_path).to(device)

        for prune_percentage in [0.2, 0.3, 0.4]:
            prune_threshold = model.transformer.get_prune_threshold_by_remaining_percentage(prune_percentage, modelConfig.prune_emb)
            print(f"prune_threshol:{prune_threshold}")
            model.transformer.setup_prune_threshold(prune_threshold, modelConfig.prune_emb)

            report_compression_rate(modelConfig, model, prune_threshold)
            pruned_percentage_by_layer = model.transformer.get_pruned_neurons_by_layer(threshold=prune_threshold)
            print(f'pruned_neurons_percentage_by_layer = {pruned_percentage_by_layer}')
            attention_head_pruned_percentage = model.transformer.get_attention_head_pruned_percentage(threshold=prune_threshold)
            print(f'attention_head_pruned_percentage = {attention_head_pruned_percentage}')
            compute_acc(data_iter, model, modelConfig.prune_emb, evalConfig.task)
