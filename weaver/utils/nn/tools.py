import numpy as np
import awkward as ak
import tqdm
import time
import torch

from collections import defaultdict, Counter
from .metrics import evaluate_metrics
from ..data.tools import _concat
from ..logger import _logger


def _flatten_label(label, mask=None):
    if label.ndim > 1:
        label = label.view(-1)
        if mask is not None:
            label = label[mask.view(-1)]
    # print('label', label.shape, label)
    return label


def _flatten_preds(model_output, label=None, mask=None, label_axis=1):
    if not isinstance(model_output, tuple):
        # `label` and `mask` are provided as function arguments
        preds = model_output
    else:
        if len(model_output == 2):
            # use `mask` from model_output instead
            # `label` still provided as function argument
            preds, mask = model_output
        elif len(model_output == 3):
            # use `label` and `mask` from model output
            preds, label, mask = model_output

    # preds: (N, num_classes); (N, num_classes, P)
    # label: (N,);             (N, P)
    # mask:  None;             (N, P) / (N, 1, P)
    if preds.ndim > 2:
        preds = preds.transpose(label_axis, -1).contiguous()
        preds = preds.view((-1, preds.shape[-1]))
        if mask is not None:
            preds = preds[mask.view(-1)]
    # print('preds', preds.shape, preds)

    if label is not None:
        label = _flatten_label(label, mask)

    return preds, label, mask


def _compute_pairwise_roc_auc(labels, scores):
    """
    Compute ROC AUC for all pairwise class combinations.
    
    Args:
        labels: 1D array of true class labels (integers 0, 1, 2, ...)
        scores: 2D array of predicted probabilities, shape (n_samples, n_classes)
    
    Returns:
        dict: Dictionary with keys like 'class_0_vs_class_1' and AUC values
    """
    from sklearn.metrics import roc_auc_score
    
    n_classes = scores.shape[1]
    auc_values = {}
    
    # Loop over all pairs of classes
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            # Get samples belonging to class i or class j
            mask = (labels == i) | (labels == j)
            if mask.sum() == 0:
                continue
            
            # Get scores and labels for these two classes
            binary_labels = labels[mask]
            binary_scores_i = scores[mask, i]
            binary_scores_j = scores[mask, j]
            
            # Compute discriminant score: P(i) / (P(i) + P(j))
            discriminant = binary_scores_i / (binary_scores_i + binary_scores_j + 1e-10)
            
            # Convert to binary labels (1 for class i, 0 for class j)
            binary_target = (binary_labels == i).astype(int)
            
            # Check if both classes are present
            if len(np.unique(binary_target)) < 2:
                continue
            
            try:
                auc = roc_auc_score(binary_target, discriminant)
                auc_values[f'class_{i}_vs_class_{j}'] = auc
            except:
                # Skip if AUC computation fails
                pass
    
    return auc_values



def train_classification(
        model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None,
        tb_helper=None, wandb_helper=None, profiler=None):
    model.train()

    data_config = train_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for X, y, _ in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            label = y[data_config.label_names[0]].long().to(dev)
            entry_count += label.shape[0]
            try:
                mask = y[data_config.label_names[0] + '_mask'].bool().to(dev)
            except KeyError:
                mask = None
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                model_output = model(*inputs)
                logits, label, _ = _flatten_preds(model_output, label=label, mask=mask)
                loss = loss_func(logits, label)
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            _, preds = logits.max(1)
            loss = loss.item()

            num_examples = label.shape[0]
            label_counter.update(label.numpy(force=True))
            num_batches += 1
            count += num_examples
            correct = (preds == label).sum().item()
            total_loss += loss
            total_correct += correct

            if profiler is not None:
                profiler.step()

            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'Acc': '%.5f' % (correct / num_examples),
                'AvgAcc': '%.5f' % (total_correct / count)})

            if tb_helper:
                tb_helper.write_scalars([
                    ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ("Acc/train", correct / num_examples, tb_helper.batch_train_count + num_batches),
                ])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=model_output, model=model,
                                            epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (entry_count, entry_count / time_diff))
    _logger.info('Train AvgLoss: %.5f, AvgAcc: %.5f' % (total_loss / num_batches, total_correct / count))
    _logger.info('Train class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("Acc/train (epoch)", total_correct / count, epoch),
        ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches
    
    if wandb_helper:
        metrics = {
            'train/loss_epoch': total_loss / num_batches,
            'train/accuracy': total_correct / count,
            'train/learning_rate': scheduler.get_last_lr()[0] if scheduler else opt.param_groups[0]['lr'],
            'epoch': epoch
        }
        _logger.info(f'Logging to wandb: {metrics}')
        wandb_helper.log(metrics, step=epoch)

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()


def evaluate_classification(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                            eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                            tb_helper=None, wandb_helper=None):
    model.eval()

    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    labels_counts = []
    observers = defaultdict(list)
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                # X, y: torch.Tensor; Z: ak.Array
                inputs = [X[k].to(dev) for k in data_config.input_names]
                label = y[data_config.label_names[0]].long().to(dev)
                entry_count += label.shape[0]
                try:
                    mask = y[data_config.label_names[0] + '_mask'].bool().to(dev)
                except KeyError:
                    mask = None
                model_output = model(*inputs)
                logits, label, mask = _flatten_preds(model_output, label=label, mask=mask)
                scores.append(torch.softmax(logits.float(), dim=1).numpy(force=True))

                if mask is not None:
                    mask = mask.cpu()
                for k, v in y.items():
                    labels[k].append(_flatten_label(v, mask).numpy(force=True))
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v)

                num_examples = label.shape[0]
                label_counter.update(label.numpy(force=True))
                if not for_training and mask is not None:
                    labels_counts.append(np.squeeze(mask.numpy(force=True).sum(axis=-1)))

                _, preds = logits.max(1)
                loss = 0 if loss_func is None else loss_func(logits, label).item()

                num_batches += 1
                count += num_examples
                correct = (preds == label).sum().item()
                total_loss += loss * num_examples
                total_correct += correct

                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / count),
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgAcc': '%.5f' % (total_correct / count)})

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch,
                                                i_batch=num_batches, mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (entry_count, entry_count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)" % tb_mode, total_loss / count, epoch),
            ("Acc/%s (epoch)" % tb_mode, total_correct / count, epoch),
        ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)

    scores = np.concatenate(scores)
    labels = {k: _concat(v) for k, v in labels.items()}
    metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
    _logger.info('Evaluation metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))
    
    # Compute pairwise ROC AUC values for all class combinations (after concatenation)
    pairwise_aucs = {}
    if for_training:  # Only compute during validation, not test
        try:
            pairwise_aucs = _compute_pairwise_roc_auc(labels[data_config.label_names[0]], scores)
            _logger.info('Pairwise ROC AUC:')
            for pair_name, auc_val in pairwise_aucs.items():
                _logger.info(f'  {pair_name}: {auc_val:.4f}')
        except Exception as e:
            _logger.warning(f'Failed to compute pairwise ROC AUC: {e}')
    
    if wandb_helper:
        wandb_mode = 'val' if for_training else 'test'
        metrics = {
            f'{wandb_mode}/loss_epoch': total_loss / count,
            f'{wandb_mode}/accuracy': total_correct / count,
            'epoch': epoch
        }
        # Add pairwise AUCs to wandb metrics
        if pairwise_aucs:
            for pair_name, auc_val in pairwise_aucs.items():
                metrics[f'{wandb_mode}/auc_{pair_name}'] = auc_val
        wandb_helper.log(metrics, step=epoch)
        
        # Log ROC curves and score distributions during validation
        if for_training and epoch is not None:
            try:
                # Get class names from data config if available
                class_names = getattr(data_config, 'label_value', None)
                if class_names is None:
                    class_names = [f'class_{i}' for i in range(scores.shape[1])]
                wandb_helper.log_roc_curves(
                    labels=labels[data_config.label_names[0]],
                    scores=scores,
                    class_names=class_names,
                    epoch=epoch,
                    log_score_distributions=True
                )
                _logger.info(f'Logged ROC curves and score distributions to wandb for epoch {epoch}')
            except Exception as e:
                _logger.warning(f'Failed to log ROC curves to wandb: {e}')

    if for_training:
        return total_correct / count
    else:
        # convert 2D labels/scores
        if len(scores) != entry_count:
            if len(labels_counts):
                labels_counts = np.concatenate(labels_counts)
                scores = ak.unflatten(scores, labels_counts)
                for k, v in labels.items():
                    labels[k] = ak.unflatten(v, labels_counts)
            else:
                assert (count % entry_count == 0)
                scores = scores.reshape((entry_count, int(count / entry_count), -1)).transpose((1, 2))
                for k, v in labels.items():
                    labels[k] = v.reshape((entry_count, -1))
        observers = {k: _concat(v) for k, v in observers.items()}
        return total_correct / count, scores, labels, observers


def evaluate_onnx(model_path, test_loader, eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix']):
    import onnxruntime
    sess = onnxruntime.InferenceSession(model_path)

    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_correct = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    observers = defaultdict(list)
    start_time = time.time()
    with tqdm.tqdm(test_loader) as tq:
        for X, y, Z in tq:
            # X, y: torch.Tensor; Z: ak.Array
            inputs = {k: v.numpy(force=True) for k, v in X.items()}
            label = y[data_config.label_names[0]].numpy(force=True)
            num_examples = label.shape[0]
            label_counter.update(label)
            score = sess.run([], inputs)[0]
            preds = score.argmax(1)

            scores.append(score)
            for k, v in y.items():
                labels[k].append(v.numpy(force=True))
            for k, v in Z.items():
                observers[k].append(v)

            correct = (preds == label).sum()
            total_correct += correct
            count += num_examples

            tq.set_postfix({
                'Acc': '%.5f' % (correct / num_examples),
                'AvgAcc': '%.5f' % (total_correct / count)})

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))

    scores = np.concatenate(scores)
    labels = {k: _concat(v) for k, v in labels.items()}
    metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
    _logger.info('Evaluation metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))
    observers = {k: _concat(v) for k, v in observers.items()}
    return total_correct / count, scores, labels, observers


def train_regression(
        model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None,
        tb_helper=None, wandb_helper=None, profiler=None):
    model.train()

    data_config = train_loader.dataset.config

    total_loss = 0
    num_batches = 0
    sum_abs_err = 0
    sum_sqr_err = 0
    count = 0
    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for X, y, _ in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            label = y[data_config.label_names[0]].float()
            num_examples = label.shape[0]
            label = label.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                model_output = model(*inputs)
                preds = model_output.squeeze()
                loss = loss_func(preds, label)
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            loss = loss.item()

            num_batches += 1
            count += num_examples
            total_loss += loss
            e = preds - label
            abs_err = e.abs().sum().item()
            sum_abs_err += abs_err
            sqr_err = e.square().sum().item()
            sum_sqr_err += sqr_err

            if profiler is not None:
                profiler.step()

            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'MSE': '%.5f' % (sqr_err / num_examples),
                'AvgMSE': '%.5f' % (sum_sqr_err / count),
                'MAE': '%.5f' % (abs_err / num_examples),
                'AvgMAE': '%.5f' % (sum_abs_err / count),
            })

            if tb_helper:
                tb_helper.write_scalars([
                    ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ("MSE/train", sqr_err / num_examples, tb_helper.batch_train_count + num_batches),
                    ("MAE/train", abs_err / num_examples, tb_helper.batch_train_count + num_batches),
                ])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=model_output, model=model,
                                            epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Train AvgLoss: %.5f, AvgMSE: %.5f, AvgMAE: %.5f' %
                 (total_loss / num_batches, sum_sqr_err / count, sum_abs_err / count))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("MSE/train (epoch)", sum_sqr_err / count, epoch),
            ("MAE/train (epoch)", sum_abs_err / count, epoch),
        ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches
    
    if wandb_helper:
        wandb_helper.log({
            'train/loss_epoch': total_loss / num_batches,
            'train/mse': sum_sqr_err / count,
            'train/mae': sum_abs_err / count,
            'train/learning_rate': scheduler.get_last_lr()[0] if scheduler else opt.param_groups[0]['lr'],
            'epoch': epoch
        }, step=epoch)

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()


def evaluate_regression(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                        eval_metrics=['mean_squared_error', 'mean_absolute_error', 'median_absolute_error',
                                      'mean_gamma_deviance'],
                        tb_helper=None, wandb_helper=None):
    model.eval()

    data_config = test_loader.dataset.config

    total_loss = 0
    num_batches = 0
    sum_sqr_err = 0
    sum_abs_err = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    observers = defaultdict(list)
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                # X, y: torch.Tensor; Z: ak.Array
                inputs = [X[k].to(dev) for k in data_config.input_names]
                label = y[data_config.label_names[0]].float()
                num_examples = label.shape[0]
                label = label.to(dev)
                model_output = model(*inputs)
                preds = model_output.squeeze().float()

                scores.append(preds.numpy(force=True))
                for k, v in y.items():
                    labels[k].append(v.numpy(force=True))
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v)

                loss = 0 if loss_func is None else loss_func(preds, label).item()

                num_batches += 1
                count += num_examples
                total_loss += loss * num_examples
                e = preds - label
                abs_err = e.abs().sum().item()
                sum_abs_err += abs_err
                sqr_err = e.square().sum().item()
                sum_sqr_err += sqr_err

                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / count),
                    'MSE': '%.5f' % (sqr_err / num_examples),
                    'AvgMSE': '%.5f' % (sum_sqr_err / count),
                    'MAE': '%.5f' % (abs_err / num_examples),
                    'AvgMAE': '%.5f' % (sum_abs_err / count),
                })

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch,
                                                i_batch=num_batches, mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)" % tb_mode, total_loss / count, epoch),
            ("MSE/%s (epoch)" % tb_mode, sum_sqr_err / count, epoch),
            ("MAE/%s (epoch)" % tb_mode, sum_abs_err / count, epoch),
        ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)
    
    if wandb_helper:
        wandb_mode = 'val' if for_training else 'test'
        wandb_helper.log({
            f'{wandb_mode}/loss_epoch': total_loss / count,
            f'{wandb_mode}/mse': sum_sqr_err / count,
            f'{wandb_mode}/mae': sum_abs_err / count,
            'epoch': epoch
        }, step=epoch)

    scores = np.concatenate(scores)
    labels = {k: _concat(v) for k, v in labels.items()}
    metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
    _logger.info('Evaluation metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))

    if for_training:
        return total_loss / count
    else:
        # convert 2D labels/scores
        observers = {k: _concat(v) for k, v in observers.items()}
        return total_loss / count, scores, labels, observers


class TensorboardHelper(object):

    def __init__(self, tb_comment, tb_custom_fn):
        self.tb_comment = tb_comment
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(comment=self.tb_comment)
        _logger.info('Create Tensorboard summary writer with comment %s' % self.tb_comment)

        # initiate the batch state
        self.batch_train_count = 0

        # load custom function
        self.custom_fn = tb_custom_fn
        if self.custom_fn is not None:
            from weaver.utils.import_tools import import_module
            from functools import partial
            self.custom_fn = import_module(self.custom_fn, '_custom_fn')
            self.custom_fn = partial(self.custom_fn.get_tensorboard_custom_fn, tb_writer=self.writer)

    def __del__(self):
        self.writer.close()

    def write_scalars(self, write_info):
        for tag, scalar_value, global_step in write_info:
            self.writer.add_scalar(tag, scalar_value, global_step)


class WandbHelper(object):
    """Helper class for Weights & Biases logging."""

    def __init__(self, config, project='weaver-training', entity=None, name=None, tags=None, notes=None):
        """
        Initialize wandb logging.
        
        Args:
            config: Training configuration dict to log
            project: Wandb project name
            entity: Wandb team/user
            name: Run name (auto-generated if None)
            tags: List of tags or comma-separated string
            notes: Run notes
        """
        try:
            import wandb
        except ImportError:
            raise ImportError('wandb is not installed. Please install it with: pip install wandb')
        
        # Convert comma-separated string to list if needed
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',') if t.strip()]
        
        # Initialize wandb run
        wandb.init(
            project=project,
            entity=entity,
            name=name,
            tags=tags,
            notes=notes,
            config=config,
            resume='allow'  # Allow resuming runs
        )
        
        self.wandb = wandb
        _logger.info(f'Initialized Wandb logging for project: {project}')
        if entity:
            _logger.info(f'  Entity: {entity}')
        if tags:
            _logger.info(f'  Tags: {tags}')

    def __del__(self):
        """Finish wandb run on cleanup."""
        if hasattr(self, 'wandb'):
            self.wandb.finish()

    def log(self, metrics, step=None):
        """
        Log metrics to wandb.
        
        Args:
            metrics: Dict of metric_name -> value
            step: Step number (epoch or batch)
        """
        self.wandb.log(metrics, step=step)

    def watch_model(self, model, log_freq=100):
        """Watch model for gradient and parameter tracking."""
        self.wandb.watch(model, log='all', log_freq=log_freq)

    def log_figure(self, key, figure, step=None):
        """
        Log a matplotlib figure to wandb.
        
        Args:
            key: Name for the figure in wandb
            figure: matplotlib figure object
            step: Step number (epoch or batch)
        """
        import matplotlib.pyplot as plt
        self.wandb.log({key: self.wandb.Image(figure)}, step=step)
        plt.close(figure)

    def log_roc_curves(self, labels, scores, class_names, epoch, log_score_distributions=True):
        """
        Generate and log ROC curves and score distributions to wandb.
        
        Args:
            labels: 1D array of true class labels (integers 0, 1, 2, ...)
            scores: 2D array of predicted probabilities, shape (n_samples, n_classes)
            class_names: List of class names corresponding to each class index
            epoch: Current epoch number
            log_score_distributions: Whether to also log score distributions
        """
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_auc_score
        
        n_classes = len(class_names)
        
        # Create masks for each class
        class_masks = {i: (labels == i) for i in range(n_classes)}
        
        # Generate ROC curves for all pairwise combinations
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        fig_roc_log, ax_roc_log = plt.subplots(figsize=(8, 6))
        
        # Count pairs for colormap
        n_pairs = n_classes * (n_classes - 1) // 2
        cmap = plt.get_cmap('cool', max(n_pairs, 1))
        cidx = 0
        
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                # Get samples belonging to class i or class j
                mask = class_masks[i] | class_masks[j]
                if mask.sum() == 0:
                    continue
                
                # Get scores for these two classes
                scores_i = scores[mask, i]
                scores_j = scores[mask, j]
                binary_labels = labels[mask]
                
                # Compute discriminant score: P(i) / (P(i) + P(j))
                discriminant = scores_i / (scores_i + scores_j + 1e-10)
                
                # Get discriminant for each class
                disc_class_i = discriminant[binary_labels == i]
                disc_class_j = discriminant[binary_labels == j]
                
                if len(disc_class_i) == 0 or len(disc_class_j) == 0:
                    continue
                
                # Calculate efficiencies
                thresholds = np.linspace(0, 1, 100)
                eff_i = np.array([np.mean(disc_class_i > t) for t in thresholds])
                eff_j = np.array([np.mean(disc_class_j > t) for t in thresholds])
                
                # Calculate AUC
                binary_target = (binary_labels == i).astype(int)
                try:
                    auc = roc_auc_score(binary_target, discriminant)
                except:
                    auc = 0.5
                
                # Plot ROC curve
                label = f'{class_names[i]} vs {class_names[j]} (AUC: {auc:.3f})'
                ax_roc.plot(eff_j, eff_i, color=cmap(cidx), linewidth=2, label=label)
                ax_roc_log.plot(eff_j, eff_i, color=cmap(cidx), linewidth=2, label=label)
                cidx += 1
        
        # Diagonal reference line
        ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7)
        ax_roc_log.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7)
        
        # Configure ROC plot
        ax_roc.set_xlabel('Background pass-through', fontsize=12)
        ax_roc.set_ylabel('Signal efficiency', fontsize=12)
        ax_roc.set_title(f'ROC Curves (Epoch {epoch})', fontsize=14)
        ax_roc.legend(loc='lower right', fontsize=9)
        ax_roc.grid(True, alpha=0.3)
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1)
        fig_roc.tight_layout()
        
        # Configure log-scale ROC plot
        ax_roc_log.set_xlabel('Background pass-through', fontsize=12)
        ax_roc_log.set_ylabel('Signal efficiency', fontsize=12)
        ax_roc_log.set_title(f'ROC Curves - Log Scale (Epoch {epoch})', fontsize=14)
        ax_roc_log.legend(loc='lower right', fontsize=9)
        ax_roc_log.grid(True, which='both', alpha=0.3)
        ax_roc_log.set_xscale('log')
        ax_roc_log.set_xlim(1e-4, 1)
        ax_roc_log.set_ylim(0, 1)
        fig_roc_log.tight_layout()
        
        # Log ROC figures
        self.log_figure('val/roc_curves', fig_roc, step=epoch)
        self.log_figure('val/roc_curves_log', fig_roc_log, step=epoch)
        
        # Log score distributions if requested
        if log_score_distributions:
            for score_idx, score_name in enumerate(class_names):
                fig_score, ax_score = plt.subplots(figsize=(8, 6))
                
                bins = np.linspace(0, 1, 41)
                for class_idx, class_name in enumerate(class_names):
                    class_scores = scores[class_masks[class_idx], score_idx]
                    if len(class_scores) > 0:
                        hist, _ = np.histogram(class_scores, bins=bins)
                        norm = np.sum(hist * np.diff(bins))
                        if norm > 0:
                            ax_score.stairs(hist / norm, edges=bins, 
                                          label=class_name, linewidth=2)
                
                ax_score.set_xlabel(f'Score ({score_name})', fontsize=12)
                ax_score.set_ylabel('Normalized counts', fontsize=12)
                ax_score.set_title(f'Score Distribution: {score_name} (Epoch {epoch})', fontsize=14)
                ax_score.legend(fontsize=10)
                ax_score.set_xlim(0, 1)
                fig_score.tight_layout()
                
                # Sanitize name for wandb key
                safe_name = score_name.replace('recojet_is', '').lower()
                self.log_figure(f'val/score_dist_{safe_name}', fig_score, step=epoch)
