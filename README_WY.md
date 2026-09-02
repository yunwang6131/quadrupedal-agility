source .venv/bin/activate
# Cluade相关配置
服务器执行：
ss -lnt | grep 43761
应该看到：
127.0.0.1:43761
## 给claude code使用代理
export HTTP_PROXY=http://127.0.0.1:43761
export HTTPS_PROXY=http://127.0.0.1:43761

测试：
curl -I --connect-timeout 10 https://platform.claude.com

# 依赖
export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH


# 复现路径
cd ~/quadrupedal-agility
uv venv --python 3.8 .venv && source .venv/bin/activate
uv pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 \
  -f https://download.pytorch.org/whl/cu118/torch_stable.html
uv pip install -r requirements.txt scipy
# 手动下载 IsaacGym_Preview_4_Package.tar.gz 后：
uv pip install -e /path/to/isaacgym/python
export LD_LIBRARY_PATH=$PWD/.venv/lib:$LD_LIBRARY_PATH   # 修 libpython3.8.so.1.0 找不到

# 直接训 TSC，跳过 BBC 的 4 天
cd tsc/legged_gym/scripts
python train.py --proj_name agility --exptid base --device_id 0 --headless \
  --randomize_start --randomize_base_mass --randomize_base_com --push_robots


python train.py --proj_name agility --exptid dist --device_id 1 --resume --resumeid base --use_camera

# 服务器tmux
tmux new -s agility          # 新建会话
# 会话里正常跑：
cd ~/quadrupedal-agility
source .venv/bin/activate
export LD_LIBRARY_PATH=$PWD/.venv/lib:$LD_LIBRARY_PATH
cd tsc/legged_gym/scripts
python train.py --proj_name agility --exptid base --device_id 0 --headless \
  --randomize_start --randomize_base_mass --randomize_base_com --push_robots

## 看训练曲线
cd ~/quadrupedal-agility
./.venv/bin/tensorboard --logdir tsc/logs/agility/base --port 6006

# 录制TSC teacher
cd ~/quadrupedal-agility/tsc/legged_gym/scripts
python play.py --proj_name agility --exptid base --device_id 0 --headless \
  --record --record_steps 1500 --record_width 854 --record_height 480
# teacher 评估成功率
cd ~/quadrupedal-agility/tsc/legged_gym/scripts
python play.py --proj_name agility --exptid base --device_id 1 --headless
# TSC蒸馏
tmux attach -t agility

cd ~/quadrupedal-agility/tsc/legged_gym/scripts
python train.py --proj_name agility --exptid dist --device_id 2 \
  --resume --resumeid base --use_camera --max_iterations 15000
# 录制TSC tudent
python play.py --proj_name agility --exptid dist --device_id 2 --headless \
  --use_camera --record --record_steps 1500

# 录制BBC跳起来的视频
cd ~/quadrupedal-agility/bbc/legged_gym/scripts
python play.py --task go2_locomotion --load_run 0 --headless True --device_id 0 \
  --mode 5 --record --record_steps 750 --record_width 1280 --record_height 720
