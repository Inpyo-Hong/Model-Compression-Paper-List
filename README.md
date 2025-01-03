# Model-Compression-Paper-List (Quantization)

## Model Compression
- Hinton, Geoffrey. "Distilling the Knowledge in a Neural Network." arXiv preprint arXiv:1503.02531 (2015).
- Han, Song, Huizi Mao, and William J. Dally. "Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding." arXiv preprint arXiv:1510.00149 (2015).
- Han, Song, et al. "EIE: Efficient inference engine on compressed deep neural network." ACM SIGARCH Computer Architecture News 44.3 (2016): 243-254.
- Zoph, B. "Neural architecture search with reinforcement learning." arXiv preprint arXiv:1611.01578 (2016).
- Hubara, Itay, et al. "Quantized neural networks: Training neural networks with low precision weights and activations." Journal of Machine Learning Research 18.187 (2018): 1-30.
- Heo, Byeongho, et al. "A comprehensive overhaul of feature distillation." Proceedings of the IEEE/CVF international conference on computer vision. 2019.

## Quantization
- Courbariaux, Matthieu, Yoshua Bengio, and Jean-Pierre David. "Binaryconnect: Training deep neural networks with binary weights during propagations." Advances in neural information processing systems 28 (2015).
- Choi, Yoojin, Mostafa El-Khamy, and Jungwon Lee. "Towards the limit of network quantization." arXiv preprint arXiv:1612.01543 (2016).
- Lin, Darryl, Sachin Talathi, and Sreekanth Annapureddy. "Fixed point quantization of deep convolutional networks." International conference on machine learning. PMLR, 2016.
- Zhou, Shuchang, et al. "Dorefa-net: Training low bitwidth convolutional neural networks with low bitwidth gradients." arXiv preprint arXiv:1606.06160 (2016).
- Krishnamoorthi, Raghuraman. "Quantizing deep convolutional networks for efficient inference: A whitepaper." arXiv preprint arXiv:1806.08342 (2018).
- Hubara, Itay, et al. "Quantized neural networks: Training neural networks with low precision weights and activations." Journal of Machine Learning Research 18.187 (2018): 1-30.
- Fan, Angela, et al. "Training with quantization noise for extreme model compression." arXiv preprint arXiv:2004.07320 (2020).
- Gholami, Amir, et al. "A survey of quantization methods for efficient neural network inference." Low-Power Computer Vision. Chapman and Hall/CRC, 2022. 291-326.


## Zero-Shot Quantization (Data-Free Quantization)
- Nagel, Markus, et al. "Data-free quantization through weight equalization and bias correction." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.
- Choi, Yoojin, et al. "Data-free network quantization with adversarial knowledge distillation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. 2020.
- Cai, Yaohui, et al. "Zeroq: A novel zero shot quantization framework." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
- Xu, Shoukai, et al. "Generative low-bitwidth data free quantization." Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XII 16. Springer International Publishing, 2020.

### Data Generation (Zero-Shot Quantization)
- Xu, Shoukai, et al. "Generative low-bitwidth data free quantization." Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XII 16. Springer International Publishing, 2020.
- Zhang, Xiangguo, et al. "Diversifying sample generation for accurate data-free quantization." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.
- Choi, Kanghyun, et al. "Qimera: Data-free quantization with synthetic boundary supporting samples." Advances in Neural Information Processing Systems 34 (2021): 14835-14847.
- Zhong, Yunshan, et al. "Intraq: Learning synthetic images with intra-class heterogeneity for zero-shot network quantization." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
- Li, Huantong, et al. "Hard sample matters a lot in zero-shot quantization." Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition. 2023.
- Qian, Biao, et al. "Rethinking data-free quantization as a zero-sum game." Proceedings of the AAAI conference on artificial intelligence. Vol. 37. No. 8. 2023.
- Qian, Biao, et al. "Adaptive data-free quantization." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
- Chen, Xinrui, et al. "TexQ: zero-shot network quantization with texture feature distribution calibration." Advances in Neural Information Processing Systems 36 (2024).
- Bai, Jianhong, et al. "Robustness-Guided Image Synthesis for Data-Free Quantization." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 10. 2024.


### Model Training (Zero-Shot Quantization)
- Guo, Cong, et al. "Squant: On-the-fly data-free quantization via diagonal hessian approximation." arXiv preprint arXiv:2202.07471 (2022).
- Choi, Kanghyun, et al. "It's all in the teacher: Zero-shot quantization brought closer to the teacher." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
- Shang, Yuzhang, et al. "Enhancing Post-training Quantization Calibration through Contrastive Learning." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.
- Li, Yuhang, et al. "GenQ: Quantization in Low Data Regimes with Generative Synthetic Data." European Conference on Computer Vision. Springer, Cham, 2025.



## LLM Quantization
- Dettmers, Tim, et al. "Gpt3. int8 (): 8-bit matrix multiplication for transformers at scale." Advances in Neural Information Processing Systems 35 (2022): 30318-30332.
- Yao, Zhewei, et al. "Zeroquant: Efficient and affordable post-training quantization for large-scale transformers." Advances in Neural Information Processing Systems 35 (2022): 27168-27183.
- Wu, Xiaoxia, et al. "Understanding int4 quantization for language models: latency speedup, composability, and failure cases." International Conference on Machine Learning. PMLR, 2023.
- Liu, Zirui, et al. "Kivi: A tuning-free asymmetric 2bit quantization for kv cache." arXiv preprint arXiv:2402.02750 (2024).
- Zhang, Cheng, et al. "LQER: Low-Rank Quantization Error Reconstruction for LLMs." arXiv preprint arXiv:2402.02446 (2024).
- Huang, Wei, et al. "Billm: Pushing the limit of post-training quantization for llms." arXiv preprint arXiv:2402.04291 (2024).
- Guo, Jinyang, et al. "Compressing large language models by joint sparsification and quantization." Forty-first International Conference on Machine Learning. 2024.
- Yao, Zhewei, et al. "Exploring post-training quantization in llms from comprehensive study to low rank compensation." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 17. 2024.
- Li, Liang, et al. "Norm tweaking: High-performance low-bit quantization of large language models." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 17. 2024.
- Heo, Jung Hwan, et al. "Rethinking channel dimensions to isolate outliers for low-bit weight quantization of large language models." arXiv preprint arXiv:2309.15531 (2023).
- Liu, Jing, et al. "Qllm: Accurate and efficient low-bitwidth quantization for large language models." arXiv preprint arXiv:2310.08041 (2023).


## Generative Model Quantization
- Shang, Yuzhang, et al. "Post-training quantization on diffusion models." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023.
- Dong, Zhenyuan, and Sai Qian Zhang. "DiTAS: Quantizing Diffusion Transformers via Enhanced Activation Smoothing." arXiv preprint arXiv:2409.07756 (2024). (WACV 2025 Accepted)
