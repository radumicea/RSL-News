# Took best epoch by val ctc loss; epoch 22 val per of 29.23% (that per is for natural Romanian, we want best for sign language),
# val/loss	val/loss_att	val/loss_ctc
# 140.50 	66.93       	148.68
# Compared to best by val per of 27.56% at epoch 40, but
# val/loss	val/loss_att	val/loss_ctc
# 164.86	67.40	        175.69
# And also empirical tests on signer mouthing were worse than epoch 22


"""
Fine-tune the auto_avsr VSR model on Romanian phonemes.

Architecture
    Frontend + Linear proj + encoder layers 0-5:
        Loaded from pretrained auto_avsr checkpoint, frozen.
    Encoder layers 6-11 + LayerNorm:
        Randomly initialized, trainable.
    Transformer decoder (6 layers):
        Randomly initialized, trainable.
        Acts as a training regularizer (alignment gradients).
    CTC head (Linear 768->39):
        Randomly initialized, trainable.

Loss:  0.9 * CTC + 0.1 * attention (label smoothing).
At inference, only CTC greedy decode is used.

Validation: val/ctc_per (monitored), val/ctc_ver, val/acc.
Pre-encoder features (frontend + proj) are cached to disk.

Usage::

    python finetune.py \\
        --root-dir dataset \\
        --train-file train_phoneme.csv \\
        --val-file val_phoneme.csv \\
        --pretrained-model-path vsr_trlrs2lrs3vox2avsp_base.pth \\
        --cache-dir ./cache_pre_enc \\
        --lr 1e-3 --encoder-lr 3e-4 \\
        --max-epochs 50 --warmup-epochs 5 \\
        --max-frames 400 --accumulate-grad-batches 4 \\
        --exp-name my_run
"""

import logging
import math
import os
import random
import sys
from argparse import ArgumentParser

import torch
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# ---------------------------------------------------------------------------
_AVSR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'auto_avsr')
if _AVSR_ROOT not in sys.path:
    sys.path.insert(0, _AVSR_ROOT)

from auto_avsr.datamodule.av_dataset import AVDataset  # noqa: E402
from auto_avsr.datamodule.data_module import (  # noqa: E402
    CustomBucketDataset,
    collate_pad,
)
from auto_avsr.datamodule.transforms import VideoTransform  # noqa: E402
from auto_avsr.espnet.nets.pytorch_backend.e2e_asr_conformer import E2E  # noqa: E402
from auto_avsr.espnet.nets.pytorch_backend.nets_utils import (  # noqa: E402
    make_non_pad_mask,
    th_accuracy,
)
from auto_avsr.espnet.nets.pytorch_backend.transformer.add_sos_eos import (
    add_sos_eos,  # noqa: E402
)
from auto_avsr.espnet.nets.pytorch_backend.transformer.mask import (
    target_mask,  # noqa: E402
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_FREEZE = 6
CTC_WEIGHT = 0.9

# 37 phonemes + <blank>(0) + <sos>/<eos>(38) = 39
NUM_PHONEME_CLASSES = 39

PHONEME_LIST = [
    '<blank>',  # 0
    '1',  # 1
    '@',  # 2
    'S',  # 3
    'Z',  # 4
    'a',  # 5
    'b',  # 6
    'd',  # 7
    'e',  # 8
    'e_X',  # 9
    'f',  # 10
    'g',  # 11
    'gZ',  # 12
    'g_j',  # 13
    'gz',  # 14
    'h',  # 15
    'i',  # 16
    'i_0',  # 17
    'j',  # 18
    'je',  # 19
    'k',  # 20
    'k_j',  # 21
    'ks',  # 22
    'l',  # 23
    'm',  # 24
    'n',  # 25
    'o',  # 26
    'o_X',  # 27
    'p',  # 28
    'r',  # 29
    's',  # 30
    't',  # 31
    'tS',  # 32
    'ts',  # 33
    'u',  # 34
    'v',  # 35
    'w',  # 36
    'z',  # 37
    '<eos>',  # 38
]
assert len(PHONEME_LIST) == NUM_PHONEME_CLASSES

# ---------------------------------------------------------------------------
# Viseme mapping (Azure X-SAMPA + Romanian)
# ---------------------------------------------------------------------------
_xsampa2visemes = {
    '{': 1,
    '@': 1,
    'V': 1,
    'A': 2,
    'O': 3,
    'E': 4,
    'U': 4,
    '3`': 5,
    'j': 6,
    'i': 6,
    'I': 6,
    'w': 7,
    'u': 7,
    'o': 8,
    'aU': 9,
    'OI': 10,
    'aI': 11,
    'h': 12,
    'r\\': 13,
    'l': 14,
    's': 15,
    'z': 15,
    'S': 16,
    'tS': 16,
    'dZ': 16,
    'Z': 16,
    'D': 17,
    'f': 18,
    'v': 18,
    'd': 19,
    't': 19,
    'n': 19,
    'T': 19,
    'k': 20,
    'g': 20,
    'N': 20,
    'p': 21,
    'b': 21,
    'm': 21,
}

_rosampa2visemes = {
    'a': 2,
    'e': 4,
    'e_X': 4,
    'je': 4,
    '1': 6,
    'i_0': 6,
    'o_X': 8,
    'r': 13,
    'ts': 15,
    'gZ': 16,
    'k_j': 20,
    'g_j': 20,
    'ks': [20, 15],
    'gz': [20, 15],
}

_VISEME_MAP = {**_xsampa2visemes, **_rosampa2visemes}

PHONEME_TO_VISEME = {}
for _pid, _pname in enumerate(PHONEME_LIST):
    if _pname in _VISEME_MAP:
        PHONEME_TO_VISEME[_pid] = _VISEME_MAP[_pname]


def phonemes_to_visemes(phoneme_ids):
    """Convert phoneme IDs to viseme IDs, expanding multi-viseme phones."""
    result = []
    for p in phoneme_ids:
        v = PHONEME_TO_VISEME.get(p)
        if v is None:
            continue
        if isinstance(v, list):
            result.extend(v)
        else:
            result.append(v)
    return result


# ---------------------------------------------------------------------------
# Cosine warm-up scheduler
# ---------------------------------------------------------------------------


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self, optimizer, warmup_epochs, total_epochs, steps_per_epoch, last_epoch=-1
    ):
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return [self._step_count / self.warmup_steps * lr for lr in self.base_lrs]
        decay = self.total_steps - self.warmup_steps
        cos_val = math.cos(math.pi * (self._step_count - self.warmup_steps) / decay)
        return [0.5 * lr * (1 + cos_val) for lr in self.base_lrs]


# ---------------------------------------------------------------------------
# Feature caching (frontend + proj -> disk)
# ---------------------------------------------------------------------------


def precompute_features(pretrained_path, root_dir, label_file, cache_dir, device):
    """Cache pre-encoder features (frontend + proj output) to disk."""
    meta_path = os.path.join(cache_dir, 'meta.pt')
    if os.path.exists(meta_path):
        logging.info('Cache exists at %s — skipping.', cache_dir)
        return

    os.makedirs(cache_dir, exist_ok=True)

    model = E2E(5049, 'video')
    ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval().to(device)

    label_path = os.path.join(root_dir, 'labels', label_file)
    dataset = AVDataset(
        root_dir=root_dir,
        label_path=label_path,
        subset='val',
        modality='video',
        audio_transform=None,
        video_transform=VideoTransform('val'),
    )

    lengths = []
    n = len(dataset)
    logging.info('Caching pre-encoder features for %d samples → %s', n, cache_dir)

    with torch.no_grad():
        for i in range(n):
            sample = dataset[i]
            video = sample['input'].unsqueeze(0).to(device)
            with torch.cuda.amp.autocast():
                x = model.frontend(video)
                x = model.proj_encoder(x)
            torch.save(
                {
                    'enc_out': x.squeeze(0).cpu().half(),
                    'target': sample['target'].clone(),
                },
                os.path.join(cache_dir, f'{i}.pt'),
            )
            lengths.append(x.shape[1])
            if (i + 1) % 500 == 0:
                logging.info('  cached %d / %d', i + 1, n)

    torch.save({'lengths': lengths}, meta_path)
    logging.info('Cached all %d samples.', n)
    del model
    torch.cuda.empty_cache()


class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir):
        meta = torch.load(os.path.join(cache_dir, 'meta.pt'), weights_only=True)
        self.input_lengths = meta['lengths']
        self.cache_dir = cache_dir

    def __getitem__(self, idx):
        data = torch.load(
            os.path.join(self.cache_dir, f'{idx}.pt'),
            weights_only=True,
            map_location='cpu',
        )
        return {'input': data['enc_out'].float(), 'target': data['target']}

    def __len__(self):
        return len(self.input_lengths)


# ---------------------------------------------------------------------------
# Augmentation & utilities
# ---------------------------------------------------------------------------


def time_mask(x, window=10, stride=25):
    """Apply time masking to batched features [B, T, D] in-place."""
    B, T, _ = x.shape
    for b in range(B):
        n_mask = int((T + stride - 0.1) // stride)
        mask_lens = torch.randint(0, window, size=(n_mask,))
        for ml in mask_lens:
            ml = ml.item()
            if T - ml <= 0:
                continue
            start = random.randrange(0, T - ml)
            x[b, start : start + ml] = 0
    return x


def edit_distance(seq1, seq2):
    """Levenshtein distance between two integer sequences."""
    n, m = len(seq1), len(seq2)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------


class PhonemeFineTuneModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args

        # Build fresh model (all random init)
        self.model = E2E(NUM_PHONEME_CLASSES, modality='video', ctc_weight=CTC_WEIGHT)

        # Load pretrained weights for frozen components only:
        # frontend, proj_encoder, encoder.embed, encoder layers 0-5
        pretrained = torch.load(
            args.pretrained_model_path, map_location='cpu', weights_only=False
        )
        own_state = self.model.state_dict()
        loaded = 0
        for key, val in pretrained.items():
            load = False
            if key.startswith(('frontend.', 'proj_encoder.', 'encoder.embed.')):
                load = True
            elif key.startswith('encoder.encoders.'):
                layer_idx = int(key.split('.')[2])
                load = layer_idx < N_FREEZE
            if load and key in own_state:
                own_state[key] = val
                loaded += 1
        self.model.load_state_dict(own_state)
        logging.info(
            'Loaded %d pretrained tensors (frontend + proj + encoder layers 0-%d).',
            loaded,
            N_FREEZE - 1,
        )

        # Freeze pretrained components
        for p in self.model.frontend.parameters():
            p.requires_grad = False
        for p in self.model.proj_encoder.parameters():
            p.requires_grad = False
        for i in range(N_FREEZE):
            for p in self.model.encoder.encoders[i].parameters():
                p.requires_grad = False

        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info('Frozen %d  |  trainable %d.', frozen, trainable)

        # Validation accumulators
        self._val_ctc_edits = 0
        self._val_ctc_ref_len = 0
        self._val_ctc_ver_edits = 0
        self._val_ctc_ver_ref_len = 0

    # ---- Encoder ----

    def _encode(self, proj_out, lengths):
        """Run encoder on cached proj features -> [B, T, D]."""
        device = proj_out.device
        padding_mask = make_non_pad_mask(lengths).to(device).unsqueeze(-2)

        if self.training:
            proj_out = time_mask(proj_out.clone())

        with torch.no_grad():
            x = self.model.encoder.embed(proj_out)
            for i in range(N_FREEZE):
                x, padding_mask = self.model.encoder.encoders[i](x, padding_mask)

        if isinstance(x, tuple):
            x = (x[0].detach(), x[1])
        else:
            x = x.detach()

        for i in range(N_FREEZE, 12):
            x, padding_mask = self.model.encoder.encoders[i](x, padding_mask)

        if isinstance(x, tuple):
            x = x[0]
        if self.model.encoder.normalize_before:
            x = self.model.encoder.after_norm(x)

        return x, padding_mask

    # ---- Forward ----

    def forward(self, proj_out, lengths, label):
        x, padding_mask = self._encode(proj_out, lengths)

        loss_ctc, _ = self.model.ctc(x, lengths, label)

        ys_in_pad, ys_out_pad = add_sos_eos(
            label, self.model.sos, self.model.eos, self.model.ignore_id
        )
        ys_mask = target_mask(ys_in_pad, self.model.ignore_id)
        pred_pad, _ = self.model.decoder(ys_in_pad, ys_mask, x, padding_mask)
        loss_att = self.model.criterion(pred_pad, ys_out_pad)

        loss = CTC_WEIGHT * loss_ctc + (1 - CTC_WEIGHT) * loss_att
        acc = th_accuracy(
            pred_pad.view(-1, self.model.odim),
            ys_out_pad,
            ignore_label=self.model.ignore_id,
        )
        return loss, loss_ctc, loss_att, acc, x

    # ---- Hooks ----

    def on_train_epoch_start(self):
        self.model.frontend.eval()
        self.model.proj_encoder.eval()
        for i in range(N_FREEZE):
            self.model.encoder.encoders[i].eval()

    # ---- Optimizer ----

    def configure_optimizers(self):
        encoder_lr = self.args.encoder_lr or self.args.lr

        encoder_params = []
        for i in range(N_FREEZE, 12):
            encoder_params.extend(
                p
                for p in self.model.encoder.encoders[i].parameters()
                if p.requires_grad
            )
        encoder_params.extend(
            p for p in self.model.encoder.after_norm.parameters() if p.requires_grad
        )
        encoder_ids = {id(p) for p in encoder_params}

        head_params = [
            p
            for p in self.model.parameters()
            if p.requires_grad and id(p) not in encoder_ids
        ]

        param_groups = [
            {'params': head_params, 'lr': self.args.lr},
            {'params': encoder_params, 'lr': encoder_lr},
        ]
        logging.info(
            'LR — heads: %.1e (%d params), encoder: %.1e (%d params)',
            self.args.lr,
            sum(p.numel() for p in head_params),
            encoder_lr,
            sum(p.numel() for p in encoder_params),
        )

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.98),
        )
        steps_per_epoch = math.ceil(
            len(self.trainer.datamodule.train_dataloader())
            / self.trainer.accumulate_grad_batches
        )
        scheduler = WarmupCosineScheduler(
            optimizer,
            self.args.warmup_epochs,
            self.args.max_epochs,
            steps_per_epoch,
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    # ---- Train / val ----

    def training_step(self, batch, batch_idx):
        loss, loss_ctc, loss_att, acc, _ = self(
            batch['inputs'], batch['input_lengths'], batch['targets']
        )
        bs = len(batch['inputs'])
        self.log(
            'train/loss',
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=bs,
            prog_bar=True,
        )
        self.log(
            'train/loss_ctc', loss_ctc, on_step=False, on_epoch=True, batch_size=bs
        )
        self.log(
            'train/loss_att', loss_att, on_step=False, on_epoch=True, batch_size=bs
        )
        self.log(
            'train/acc', acc, on_step=True, on_epoch=True, batch_size=bs, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_ctc, loss_att, acc, enc_out = self(
            batch['inputs'], batch['input_lengths'], batch['targets']
        )
        bs = len(batch['inputs'])
        self.log('val/loss', loss, batch_size=bs, prog_bar=True)
        self.log('val/loss_ctc', loss_ctc, batch_size=bs)
        self.log('val/loss_att', loss_att, batch_size=bs)
        self.log('val/acc', acc, batch_size=bs, prog_bar=True)

        # CTC greedy PER + VER
        lengths = batch['input_lengths']
        targets = batch['targets']
        with torch.no_grad():
            frame_ids = self.model.ctc.ctc_lo(enc_out).argmax(dim=-1)
        for b in range(bs):
            raw = frame_ids[b, : lengths[b]].tolist()
            collapsed = []
            prev = None
            for tok in raw:
                if tok != prev:
                    if tok != 0:
                        collapsed.append(tok)
                    prev = tok
            ref = targets[b][targets[b] != -1].tolist()
            if len(ref) > 0:
                self._val_ctc_edits += edit_distance(collapsed, ref)
                self._val_ctc_ref_len += len(ref)
                hyp_v = phonemes_to_visemes(collapsed)
                ref_v = phonemes_to_visemes(ref)
                if len(ref_v) > 0:
                    self._val_ctc_ver_edits += edit_distance(hyp_v, ref_v)
                    self._val_ctc_ver_ref_len += len(ref_v)

    def on_validation_epoch_end(self):
        if self._val_ctc_ref_len > 0:
            self.log(
                'val/ctc_per',
                self._val_ctc_edits / self._val_ctc_ref_len,
                prog_bar=True,
            )
        if self._val_ctc_ver_ref_len > 0:
            self.log(
                'val/ctc_ver',
                self._val_ctc_ver_edits / self._val_ctc_ver_ref_len,
                prog_bar=True,
            )
        self._val_ctc_edits = 0
        self._val_ctc_ref_len = 0
        self._val_ctc_ver_edits = 0
        self._val_ctc_ver_ref_len = 0


# ---------------------------------------------------------------------------
# Data module
# ---------------------------------------------------------------------------


class PhonemeDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_workers = min(os.cpu_count() or 4, 4)

    def _make_loader(self, subset):
        cache_dir = os.path.join(
            self.args.cache_dir, 'train' if subset == 'train' else 'val'
        )
        dataset = CachedDataset(cache_dir)
        max_frames = self.args.max_frames if subset == 'train' else 1000
        max_frames = max(max_frames, max(dataset.input_lengths))
        num_buckets = 50 if subset == 'train' else 1
        bucket = CustomBucketDataset(
            dataset, dataset.input_lengths, max_frames, num_buckets
        )
        return torch.utils.data.DataLoader(
            bucket,
            batch_size=None,
            num_workers=self.num_workers,
            shuffle=(subset == 'train'),
            collate_fn=collate_pad,
        )

    def train_dataloader(self):
        return self._make_loader('train')

    def val_dataloader(self):
        return self._make_loader('val')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = ArgumentParser()
    p.add_argument('--root-dir', type=str, required=True)
    p.add_argument('--train-file', type=str, required=True)
    p.add_argument('--val-file', type=str, required=True)
    p.add_argument('--pretrained-model-path', type=str, required=True)
    p.add_argument('--cache-dir', type=str, required=True)
    p.add_argument('--exp-dir', type=str, default='./exp')
    p.add_argument('--exp-name', type=str, required=True)
    p.add_argument('--max-epochs', type=int, default=50)
    p.add_argument('--warmup-epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--encoder-lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=0.03)
    p.add_argument('--gradient-clip', type=float, default=10.0)
    p.add_argument('--max-frames', type=int, default=400)
    p.add_argument('--accumulate-grad-batches', type=int, default=4)
    p.add_argument(
        '--ckpt-path',
        type=str,
        default=None,
        help='Resume training from this Lightning checkpoint',
    )
    p.add_argument(
        '--init-from',
        type=str,
        default=None,
        help='Load model weights from checkpoint but start fresh '
        'optimizer/scheduler (full warmup from epoch 0)',
    )
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    args = parse_args()
    seed_everything(42, workers=True)

    # Save CTC weight in hparams for test.py compatibility
    args.ctc_weight = CTC_WEIGHT

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for subset, label in [('train', args.train_file), ('val', args.val_file)]:
        precompute_features(
            args.pretrained_model_path,
            args.root_dir,
            label,
            os.path.join(args.cache_dir, subset),
            device,
        )

    model = PhonemeFineTuneModule(args)

    # --init-from: load weights only, fresh optimizer/scheduler
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location='cpu', weights_only=False)
        state = {
            k.replace('model.', ''): v
            for k, v in ckpt['state_dict'].items()
            if k.startswith('model.')
        }
        model.model.load_state_dict(state, strict=True)
        logging.info(
            'Loaded weights from %s (epoch %d) — fresh LR schedule.',
            args.init_from,
            ckpt['epoch'],
        )

    datamodule = PhonemeDataModule(args)

    ckpt_dir = os.path.join(args.exp_dir, args.exp_name)
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor='val/loss_ctc',
            mode='min',
            save_last=True,
            save_top_k=1,
            filename='best',
        ),
        LearningRateMonitor(logging_interval='step'),
    ]

    trainer = Trainer(
        default_root_dir=args.exp_dir,
        max_epochs=args.max_epochs,
        devices=1,
        accelerator='gpu',
        precision='16-mixed',
        gradient_clip_val=args.gradient_clip,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    main()
