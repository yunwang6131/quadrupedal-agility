# 环境说明（重要）

当前目录是通过 sshfs 挂载的**远程项目**，实际位于服务器 `agility`
(111.0.18.168) 的 `/home/wangyun/quadrupedal-agility`。

## 文件操作
直接读写当前目录即可，等同于操作服务器上的文件。

## 执行命令
**绝不在本地执行 python / pip / uv**（本地没有 GPU、没有 IsaacGym 环境）。
所有命令必须通过 SSH 送到服务器：

    ssh agility 'cd ~/quadrupedal-agility && source .venv/bin/activate && <命令>'

## 长时间训练
不要前台阻塞，用 tmux：

    ssh agility 'tmux new -d -s train "cd ~/quadrupedal-agility && source .venv/bin/activate && python tsc/legged_gym/scripts/train.py --proj_name agility --exptid base 2>&1 | tee ~/quadrupedal-agility/train.log"'

查看进度：

    ssh agility 'tail -50 ~/quadrupedal-agility/train.log'

## 性能注意
不要遍历或搜索这些目录，它们很大且无用：
`tsc/legged_gym/logs/`、`.venv/`、`weights/`、`resources/`
