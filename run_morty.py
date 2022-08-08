from datasets import load_dataset, load_metric
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m", "--model",
                        default="allenai/led-large-16384",
                        help="The model to pull from HF.")
    parser.add_argument("-i", "--input",
                        default=6712,
                        help="Max input length")
    parser.add_argument("-o", "--output",
                        default=512,
                        help="Max output length")
    args = parser.parse_args()

    model_name = args.model
    max_input_length = args.input
    max_output_length = args.output

    # load rouge
    rouge = load_metric("rouge")


    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # load data
    train_dataset = load_dataset('json', data_files='./ds_train.json', split='train')
    val_dataset = load_dataset('json', data_files='./ds_val.json', split='train')
    test_dataset = load_dataset('json', data_files='./ds_test.json', split='train')

    #max_input_length = 6712
    #max_output_length = 512
    batch_size = 2


    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        inputs = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_input_length,
        )
        outputs = tokenizer(
            batch["summary"],
            padding="max_length",
            truncation=True,
            max_length=max_output_length,
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask

        # create 0 global_attention_mask lists
        batch["global_attention_mask"] = len(batch["input_ids"]) * [
            [0 for _ in range(len(batch["input_ids"][0]))]
        ]

        # since above lists are references, the following line changes the 0 index for all samples
        batch["global_attention_mask"][0][0] = 1
        batch["labels"] = outputs.input_ids

        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in batch["labels"]
        ]

        return batch


    train_dataset = train_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["text", "summary"],
    )

    val_dataset = val_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["text", "summary"],
    )

    test_dataset = test_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["text", "summary"],
    )


    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )
    val_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )
    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )

    from transformers import AutoModelForSeq2SeqLM

    led = AutoModelForSeq2SeqLM.from_pretrained(model_name, gradient_checkpointing=True, use_cache=False)

    # set generate hyperparameters
    led.config.num_beams = 4
    led.config.max_length = 512
    led.config.min_length = 100
    led.config.length_penalty = 2.0
    led.config.early_stopping = True
    led.config.no_repeat_ngram_size = 3

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge2_output = rouge.compute(
            predictions=pred_str, references=label_str, rouge_types=["rouge2"]
        )["rouge2"].mid

        rouge1_output = rouge.compute(
            predictions=pred_str, references=label_str, rouge_types=["rouge1"]
        )["rouge1"].mid

        rougel_output = rouge.compute(
            predictions=pred_str, references=label_str, rouge_types=["rougeL"]
        )["rougeL"].mid

        return {
            "rouge1_precision": round(rouge1_output.precision, 4),
            "rouge1_recall": round(rouge1_output.recall, 4),
            "rouge1_fmeasure": round(rouge1_output.fmeasure, 4),
            "rouge2_precision": round(rouge2_output.precision, 4),
            "rouge2_recall": round(rouge2_output.recall, 4),
            "rouge2_fmeasure": round(rouge2_output.fmeasure, 4),
            "rougel_precision": round(rougel_output.precision, 4),
            "rougel_recall": round(rougel_output.recall, 4),
            "rougel_fmeasure": round(rougel_output.fmeasure, 4),
        }

    # enable fp16 training
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=True,
        output_dir="./",
        logging_steps=5,
        eval_steps=50,
        save_steps=2000,
        save_total_limit=1,
        gradient_accumulation_steps=4,
        num_train_epochs=20,
    )

    trainer = Seq2SeqTrainer(
        model=led,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    trainer.save_model()

    #print(trainer.evaluate(val_dataset))
    print(trainer.evaluate(test_dataset))
