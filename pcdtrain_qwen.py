import numpy as np
import torch
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer, Qwen2Config

from transformers import TrainingArguments, Trainer
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from datasets import PretrainDataCollator, PCDBuildingDataset
from torch import nn
import os

model_name = "./models/Qwen/Qwen3-0___6B-Base"

model = Qwen2ForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
# cfg = Qwen2Config.from_pretrained(model_name)
# model = Qwen2ForCausalLM(cfg)

tokenizer = Qwen2Tokenizer.from_pretrained(model_name)

model = model.float()

if __name__ == '__main__':
    from transformers import TrainerCallback
    # class LogCallBack(TrainerCallback):
    #     def on_step_end(self, args, state, control, **kwargs):
    #         cur_step = state.global_step
    #         if cur_step % 20 == 0:
    #             torch.cuda.empty_cache()
    #         if cur_step % state.eval_steps != 0:
    #             return
    #         # print(kwargs)
    #         model = kwargs["model"]
    #         ds = kwargs["train_dataloader"].dataset
    #         tokenizer = ds.tokenizer
    #         items = np.random.permutation(len(ds))[:10]
    #         with torch.no_grad():
    #             print()
    #             for i in range(items.shape[0]):
    #                 item = items[i].item()
    #                 input_inds, labels = ds[item]
    #                 start_ind = 0
    #                 while labels[start_ind] == tokenizer.pad_token_id:
    #                     start_ind += 1
    #                 input_inds = input_inds[:start_ind+1]
    #                 input_inds = torch.tensor([input_inds], dtype=torch.long).to(model.device)
    #                 output = model.generate(input_ids=input_inds, max_length=8192, use_cache=True, top_k=1)
    #
    #                 resp = tokenizer.batch_decode(output[:])[0]
    #                 # print(i)
    #                 # print(resp)
    #                 with open("./log_3d_buildings/%d.txt" % i, "w") as f:
    #                     f.write(resp)
    #                 print("\rsave objs: %d / %d" % (i+1, items.shape[0]), end="")
    #             print()
    #             torch.cuda.empty_cache()

    class LogCallBack(TrainerCallback):
        def __init__(self, every_steps: int = 500):
            self.every_steps = every_steps

        def on_step_end(self, args, state, control, **kwargs):
            cur_step = state.global_step or 0
            if cur_step == 0:
                return
            if cur_step % 20 == 0:
                torch.cuda.empty_cache()

            # 用固定步频触发，可自己调整到 1000/2000/5000
            if cur_step % self.every_steps != 0:
                return

            model = kwargs["model"]
            ds = kwargs["train_dataloader"].dataset
            tokenizer = ds.tokenizer

            os.makedirs("./log_3d_buildings", exist_ok=True)  # 确保目录存在

            items = np.random.permutation(len(ds))[:1]  # 先只可视化 1 个，别 10 个，快很多
            model.eval()
            with torch.no_grad():
                for i in items:
                    input_inds, labels = ds[int(i)]
                    # 只取到第一处非 pad 之前，保持生成更稳定
                    start_ind = 0
                    while start_ind < len(labels) and labels[start_ind] == tokenizer.pad_token_id:
                        start_ind += 1
                    input_inds = input_inds[:start_ind + 1]
                    input_inds = torch.tensor([input_inds], dtype=torch.long).to(model.device)

                    # 生成长度别用 8192，极慢且易跑飞
                    output = model.generate(
                        input_ids=input_inds,
                        max_new_tokens=512,
                        do_sample=False,
                        use_cache=True,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    resp = tokenizer.batch_decode(output, skip_special_tokens=False)[0]
                    with open(f"./log_3d_buildings/step{cur_step}.txt", "w", encoding="utf-8") as f:
                        f.write(resp)
            model.train()
            torch.cuda.empty_cache()

    # class MyTrainer(Trainer):
    #     def __init__(self, **kwargs):
    #         super().__init__(**kwargs)
    #         self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    #
    #     def compute_loss(self, model, inputs, return_outputs=False):
    #         input_ids = inputs["input_ids"]
    #         labels = inputs["labels"]
    #         outputs = model(input_ids=input_ids)
    #         logits = outputs[0]
    #
    #         loss = self.loss_fn(
    #             logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)
    #         )
    #         return loss
    #         # return CausalLMOutputWithPast(loss=loss, logits=logits)

    from transformers import Trainer
    import torch.nn as nn

    class MyTrainer(Trainer):
        def __init__(self, tokenizer=None, **kwargs):
            super().__init__(**kwargs)
            ignore_index = tokenizer.pad_token_id if tokenizer is not None else -100
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

        def compute_loss(
                self,
                model,
                inputs,
                return_outputs: bool = False,
                **kwargs
        ):
            labels = inputs["labels"]
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1).to(logits.device),
            )

            return (loss, outputs) if return_outputs else loss


            labels = inputs["labels"]

            # 让模型用 input_ids 做 forward，其它键 (如 attention_mask) 可直接传给模型
            outputs = model(input_ids=input_ids)

            # HuggingFace 模型一般返回 logits 或者 tuple
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            # 计算交叉熵损失
            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1).to(logits.device),
            )

            return (loss, outputs) if return_outputs else loss


    output_dir = './params'

    # args = TrainingArguments(
    #     output_dir=output_dir,
    #     do_train=True,
    #     per_device_train_batch_size=1,
    #     learning_rate=0.0001,
    #     num_train_epochs=100,
    #     save_steps=1000,
    #     save_total_limit=5,
    #     save_only_model=True,
    #     fp16=True,
    #     gradient_accumulation_steps=4,
    #     logging_steps=100,
    #     report_to='tensorboard',
    #     dataloader_pin_memory=False,
    #     dataloader_num_workers=0,
    #     remove_unused_columns=None
    # )
    # 进度条总进度是 epoch x len(dataset)

    # 改了之后的
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=1,
        learning_rate=1e-4,

        # ⇩ 用步数“盖帽”，并关闭按 epoch 跑满
        max_steps=10_000,  # 先跑 1e4 步做 sanity-run
        num_train_epochs=1,  # 让 max_steps 生效 只是为了通过那行比较，不会真的跑满 1 个 epoch

        # Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
        # The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence,
        # you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.

        save_steps=1000,
        save_total_limit=5,
        # save_only_model=True,
        save_only_model=False,

        fp16=True,
        gradient_accumulation_steps=4,

        # ⇩ 降低日志/评估开销
        logging_steps=50,
        eval_strategy="no", # 先不做评估/生成

        # ⇩ 关闭外部上报（省点干扰）
        report_to=[],  # 原来是 'tensorboard'

        # ⇩ 轻量提升吞吐（几乎不影响稳定性）
        dataloader_pin_memory=True,  # 原来是 False
        dataloader_num_workers=0,  # 先保持 0，稳定优先

        remove_unused_columns=None
    )

    trainer = MyTrainer(
        model=model,
        args=args,
        train_dataset=PCDBuildingDataset("./dataset", tokenizer),
        data_collator=PretrainDataCollator(tokenizer),
        callbacks=[LogCallBack(every_steps=500)]
    )

    from transformers.trainer_utils import get_last_checkpoint

    last_ckpt = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None

    print(last_ckpt)

    if last_ckpt is None:
        print("No checkpoint found. Training from scratch.")
        trainer.train()  # 不传 resume_from_checkpoint
    else:
        print(f"Resuming from checkpoint: {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)

    # trainer.train(resume_from_checkpoint=True)
    # trainer.train(resume_from_checkpoint=False)