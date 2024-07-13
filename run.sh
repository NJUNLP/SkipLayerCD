export ENABLE_FP32=1

for algo in direct dola sl-h sl-d; do
    for model in mistral-7b deepseek-7b baichuan-2-7b llama-3-8b llama-2-13b; do
        for data_name in aqua gsm8k gsm-plus-digits; do
            ALGO=$algo MODEL=$model DATA_NAME=$data_name python run.py
        done

        for lang in en de fr es ru zh ja th te bn sw; do
            ALGO=$algo MODEL=$model DATA_NAME="mgsm-$lang" python run.py
        done
    done
done

for model in baichuan-2-7b llama-2-13b; do
    for data_name in aqua gsm8k gsm-plus-digits; do
        ALGO=vanilla MODEL=$model DATA_NAME=$data_name python run.py
    done

    for lang in en de fr es ru zh ja th te bn sw; do
        ALGO=vanilla MODEL=$model DATA_NAME="mgsm-$lang" python run.py
    done
done
