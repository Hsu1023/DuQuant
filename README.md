# Rotation and Permutation for Advanced Outlier Management and Efficient Quantization of LLMs

<h5 align="center">

[![arXiv](https://img.shields.io/badge/DuQuant-2406.01721-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2406.01721)
[![License](https://img.shields.io/badge/Code%20License-MIT-yellow)](https://github.com/Hsu1023/DuQuant/blob/main/LICENSE)
 <br>

</h5>

Welcome to the official code repository for `Rotation and Permutation for Advanced Outlier Management and Efficient Quantization of LLMs`.

## News
* [2024/09/06] 🔥 We release the code!
* [2024/06/03] 🚀 Our paper is available on arXiv!

## Install
```bash
pip install -r requirements.txt
```

## Run
### 1. Preprocess
```bash
python get_rot.py # need to be run only once for all models
python generate_act_scale_shift.py --model PATH_OF_MODEL # need to be run only once for each model (path can be hugging-face hub path or relative path)
```

### 2. Quantization
The bash script for `DuQuant` can be found in `run.sh`. You can choose the model to be quantized by providing model path after `--model` order. To evaluate `DuQuant + lwc` method, you can run `run_lwc.sh` script. In addition, you can add `--save_dir` to save the quantized models, and use `--resume` to reload the saved models.

Currently, we support LLaMA series (LLaMA 1,2 and 3), Vicuna series, and Mistral models. **A more detailed description will be provided soon.**

## Contact
For immediate queries or further information, please open an issue or contact <xuhb20@mails.tsinghua.edu.cn> or <haokun.lin@cripac.ia.ac.cn>.

## Acknowledgement
This repo is build upon the following projects:

* [OmniQuant](https://github.com/OpenGVLab/OmniQuant)
* [IntactKV](https://github.com/ruikangliu/IntactKV)
* [EAGLE](https://github.com/SafeAILab/EAGLE)
* [FastChat](https://github.com/lm-sys/FastChat)

We thank the authors for their code.

## Citation
Please cite our work if you use our code or discuss our findings in your own research:
```
@article{lin2024rotation,
  title={Rotation and Permutation for Advanced Outlier Management and Efficient Quantization of LLMs},
  author={Lin, Haokun and Xu, Haobo and Wu, Yichen and Cui, Jingzhi and Zhang, Yingtao and Mou, Linzhan and Song, Linqi and Sun, Zhenan and Wei, Ying},
  journal={arXiv preprint arXiv:2406.01721},
  year={2024}
}