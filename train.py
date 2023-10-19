from tqdm import tqdm
from pathlib import Path
from loguru import logger
from utils.tools import *
from utils.dataset import *
from torch.utils.data import DataLoader


class Trainer:

    def __init__(self,
                 model=None,
                 optimizer=None,
                 scheduler=None,
                 accelerator=None):
        # load model
        self.model = model
        if optimizer is None:
            raise ValueError("optimizer is not initialized!")
        self.optimizer = optimizer

        self.scheduler = scheduler

        self.accelerator = accelerator
        if self.accelerator is not None:
            logger.info("Will using accelerator to train !")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, epochs, train_data_loader=None, test_data_loader=None):
        train_total_loss = 0
        self.model.train()
        for epoch in range(1, epochs + 1):
            train_pbar = tqdm(total=epochs, desc=f'Epoch: {epoch}/{epochs},Micro Batches: {len(train_data_loader)}',
                              postfix=dict, mininterval=0.3)
            for step, batch in enumerate(train_data_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # backward, calculate gradient
                if self.accelerator is not None:
                    with self.accelerator.autocast():
                        # forward
                        outputs = self.model(**batch, use_cache=False)
                        loss = outputs.loss
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                else:
                    outputs = self.model(**batch, use_cache=False)
                    loss = outputs.loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.optimizer.zero_grad()  # zero the gradient
                # lr scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

                if self.accelerator is not None:
                    train_total_loss = self.accelerator.gather(loss).sum()
                else:
                    train_total_loss += loss.item()

                train_pbar.set_postfix(
                    **{'train average loss': train_total_loss / (step + 1), 'train loss': loss.item(),
                       'lr': get_lr(self.optimizer)})
                train_pbar.update(1)

            # test
            if test_data_loader is not None:
                test_total_loss = 0
                test_pbar = tqdm(total=epochs, desc=f'Epoch: {epoch}/{epochs},Micro Batches: {len(train_data_loader)}',
                                 postfix=dict, mininterval=0.3)
                self.model.eval()
                for step, batch in enumerate(test_data_loader):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch, use_cache=False)
                    loss = outputs.loss
                    # tqdm
                    test_pbar.set_postfix(
                        **{'test average loss': test_total_loss / (step + 1), 'test loss': loss.item()})
                    test_pbar.update(1)

    def save_model(self, out_dir: str = None):
        if not Path(out_dir).exists():
            Path(out_dir).mkdir()

        if self.model is not None:
            self.model.save_pretrained(out_dir, torch_dtype=torch.float16)


if __name__ == "__main__":
    seed_everything()

    ds_train = LlamaDataset(train_path="./data/train.txt", test_path="./data/test.txt", word2id=word2id)

    dl_train = DataLoader(dataset=ds_train,
                          batch_size=2,
                          drop_last=True,
                          shuffle=True,
                          collate_fn=data_collator,
                          )

    config = LlamaConfig(
        vocab_size=len(data_builder.get_vocab()),
        hidden_size=512,
        intermediate_size=2752,
        num_hidden_layers=8,
        num_attention_heads=16,
        num_key_value_heads=4,
        rope_scaling=None,
        hidden_act='silu',
        max_position_embeddings=128,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        pretraining_tp=1,
        max_new_tokens=100
    )
    model = LlamaForCausalLM(config)

    epochs = 20
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs,
                                                           eta_min=3e-5)

    trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler)
    trainer.train(epochs=epochs, train_data_loader=dl_train)
    trainer.save_model("./trained")
