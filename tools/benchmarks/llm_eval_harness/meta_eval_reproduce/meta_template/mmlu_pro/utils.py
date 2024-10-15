import datasets


def doc_to_text(doc: dict) -> str:
    return doc["input_final_prompts"][0]


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["input_question"],
            "gold": doc["input_correct_responses"][0],
        }
        return out_doc

    dataset = dataset.select_columns(
        [
            "input_question",
            "input_correct_responses",
            "input_final_prompts",
            "is_correct",
            "input_question_hash",
            "input_choice_list",
            "output_prediction_text",
        ]
    )
    dataset = dataset.rename_column("is_correct", "previously_is_correct")
    dataset = dataset.map(_process_doc)
    return dataset.map(_process_doc)


def doc_to_text_zeroshot_cot(doc: dict) -> str:
    prompt = (
        "<|start_header_id|>user<|end_header_id|>\n\nGiven the following question and candidate answers, choose the best answer.\nQuestion: "
        + doc["question"]
    )
    for i, option in enumerate(doc["options"]):
        prompt += "\n{}. {}".format(chr(i + 1 + 64), option)
    prompt += '\n\nYour response should end with "The best answer is [the_answer_letter]." where the [the_answer_letter] is a letter from the provided choices.\n\nLet\'s think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    return prompt


def process_docs_zeroshot_cot(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc_zeroshot_cot(doc: dict) -> dict:
        out_doc = {
            "problem": doc["question"],
            "gold": doc["answer"],
        }
        return out_doc

    dataset = dataset.select_columns(
        [
            "question",
            "answer",
            "options",
            "answer_index",
        ]
    )
    dataset = dataset.map(_process_doc_zeroshot_cot)
    return dataset.map(_process_doc_zeroshot_cot)
