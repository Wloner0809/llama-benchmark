lm_eval --model vllm \
        --model_args pretrained=/home/v-wangyu1/model-ckpt/iter_llama_sft-model-as-reward-model/LLaMA3.1_iter3_sft,dtype=auto,gpu_memory_utilization=0.95,max_model_len=8192,add_bos_token=True,seed=42 \
        --tasks meta_instruct \
        --batch_size auto \
        --output_path eval_results \
        --include_path /home/v-wangyu1/llama-benchmark/tools/benchmarks/llm_eval_harness/meta_eval_reproduce/work_dir \
        --seed 42  \
        --log_samples