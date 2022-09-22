#!/bin/bash

. path.sh

dir=result_model
average_num=10
decoding_chunk_size=-1
decode_checkpoint=$dir/avg${average_num}.pt
ctc_weight=0.5
reverse_weight=0.0
dict=result_model/lang_char.txt
decode_modes="attention_rescoring"
for mode in ${decode_modes}; do
    {
      result_dir=./test_codeswiting
      mkdir -p $result_dir
      python wenet/bin/recognize.py --gpu -1 \
        --mode $mode \
        --config $dir/train.yaml \
        --data_type "raw" \
        --test_data  test_cs/data.list \
        --checkpoint $decode_checkpoint \
        --bpe_model $dir/train_960_unigram2000.model \
        --beam_size 10 \
        --batch_size 1 \
        --penalty 0.0 \
        --dict $dict \
        --ctc_weight $ctc_weight \
        --reverse_weight $reverse_weight \
        --result_file $result_dir/text \
        ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
      python tools/compute-wer.py --char=1 --v=1 \
        test_cs/text $result_dir/text > $result_dir/wer
    }
done
