# shared env for the vLLM venv on Jetson Thor
# jetson-ai-lab torch 2.10 wheel needs system CUDA 13 libs + pip nvidia-* libs on LD_LIBRARY_PATH,
# and must NOT see the ROS Jazzy PYTHONPATH.
export VLLM_VENV="${VLLM_VENV:-$HOME/vllm-venv-g4}"   # vLLM 가상환경 (Gemma 4 용)
export PYTHONPATH=
_NVLIB=$(echo "$VLLM_VENV"/lib/python3.12/site-packages/nvidia/*/lib | tr ' ' ':')
export LD_LIBRARY_PATH="$_NVLIB:$VLLM_VENV/lib/python3.12/site-packages/torch/lib:/usr/local/cuda-13.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
