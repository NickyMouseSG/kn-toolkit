if status is-interactive
# Commands to run in interactive sessions can go here
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
if test -f /Users/kn/miniconda3/bin/conda
    eval /Users/kn/miniconda3/bin/conda "shell.fish" "hook" $argv | source
else
    if test -f "/Users/kn/miniconda3/etc/fish/conf.d/conda.fish"
        . "/Users/kn/miniconda3/etc/fish/conf.d/conda.fish"
    else
        set -x PATH "/Users/kn/miniconda3/bin" $PATH
    end
end
# <<< conda initialize <<<
end

set tide_context_always_display true

abbr sv "~/server_utils/server.sh"
abbr lsq "python ~/server_utils/list_task.py"
abbr nv 'nvidia-smi --query-gpu=gpu_name,memory.total,memory.free --format=csv'
abbr nvp "gpustat -f"
# pip install py3nvml
abbr base "conda activate base"
abbr tc 'conda activate torch'
abbr zh 'conda activate zh'
abbr pyipdb "python -m ipdb -c continue"
abbr pypdb "python -m pdb -c continue"
# abbr aner "conda activate decouplener"
abbr fgA "python ~/server_utils/query_cluster.py --task available --all -f"
abbr fst "python ~/server_utils/query_cluster.py --task stat"
abbr fqn "python ~/server_utils/query_cluster.py --task query_node"
abbr bash_only "export NO_FISH=1; bash"
abbr tool "python -m kn_util.tools."

abbr gk "gpukill"
abbr g "gpu"
abbr dskill "knkill; and knkill deepspeed; and knkill acclerate"
abbr 'git?' 'copilot_git-assist'
abbr '??' 'copilot_what-the-shell'
abbr kntar 'tar -C DIR --use-compress-program=pigz -cvpf - . | split -b 4G -d - OUTPUT.tgz.'
abbr knuntar 'cat ./INPUT.tgz.* | tar --use-compress-program=unpigz -xvpf -'

abbr "knrsync" 'python $HOME/server_utils/rsync_tool.py'
abbr "skip_clone" "GIT_LFS_SKIP_SMUDGE=1 git clone"

function kill_nfs
    # 递归搜索当前文件夹内所有 .nfs 文件
    set nfs_files (find . -type f -name '*.nfs')
    # 遍历所有 .nfs 文件
    for nfs_file in $nfs_files
        # 使用 lsof 命令找到正在使用这个文件的进程
        set processes (lsof "$nfs_file" | awk 'NR>1 {print $2}' | sort -u)

        # 终止这些进程
        for process in $processes
            echo "Killing process: $process"
            kill -9 $process
        end
    end
end

function kill_jobs
    for job in (jobs -p)
        kill $job
        and echo "Killed job $job"
        or echo "Failed to kill job $job"
    end
end


function rsync_to
    set dst_ssh $argv[1]
    set src (pwd | sed "s|$HOME|~|")
    set dst (ssh $dst_ssh readlink -f $src)
    set src (pwd)
    set cmd "rsync -vaurP $src/ $dst_ssh:$dst/"
    ssh $dst_ssh mkdir -p $dst

    echo $cmd
    eval $cmd
end

function rsync_from
    set dst_ssh $argv[1]
    set src (pwd | sed "s|$HOME|~|")
    set dst (ssh $dst_ssh readlink -f $src)
    set src (pwd)
    set cmd "rsync -vaurP $dst_ssh:$dst/ $src/"
    mkdir -p $src

    echo $cmd
    eval $cmd
end

function play
    cd $HOME/WORKSPACE/playground
    mkdir -p $argv[1]
    cd $argv[1]
    readlink -f .
end

function task
    cd $HOME/TASKS
    mkdir -p $argv[1]
    cd $argv[1]
    readlink -f .
end

function new
    touch $argv
    readlink -f $argv
end

# function knrsync
#     set src $argv[1]
#     set dst $argv[2]
#     set cmd "python $HOME/server_utils/rsync_tool.py $src $dst"
#     echo $cmd
#     eval $cmd
# end

function gpu
    export CUDA_VISIBLE_DEVICES=$argv
end

function whichgpu
    echo $CUDA_VISIBLE_DEVICES
end

function rl
    readlink -f $argv
end

function fg 
    python ~/server_utils/query_cluster.py --task available $argv
end

function fu 
    python ~/server_utils/query_cluster.py --task usage $argv
end

function sg
  python $HOME/server_utils/dist_train.py --gpus $argv
end

function knkill
    if test "$argv[1]" = "ALL"
        set argv
    end

    if test -z "$argv[1]"
        set filter "python"
    else
        set filter $argv[1]
    end

    ps -u (whoami) --no-headers -o pid,comm= | grep -v -E "^\$|((string echo $PPID))|slurmstepd|python311|python39|tmux|bash|fish" | grep -- $filter | awk '{print $1}' | xargs kill -9
end

function _gpustr
    set variable $argv[1]
    set gpu_count (gpustat --no-header | wc -l)
    set last_gpu_index (math $gpu_count - 1)

    if test "$variable" = "-1"
        echo (seq 0 $last_gpu_index | string join ",")
    else
        echo $variable
    end

end

function gpukill
    set variable (_gpustr $argv[1])
    for i in (string split "," $variable)
        gpustat --id $i --json | grep pid | awk '{print $2}' | tr -d ',' | xargs kill -9
    end
end

function copilot_what-the-shell
    set TMPFILE (mktemp)
    function cleanup --on-event fish_exit; rm -f $copilot_what_the_shell_TMPFILE; end
    if github-copilot-cli what-the-shell $argv --shellout $TMPFILE
        if test -e "$TMPFILE"
            set FIXED_CMD (cat $TMPFILE)
            eval $FIXED_CMD
        else
            echo "Apologies! Extracting command failed"
        end
    else
        return 1
    end
end

function copilot_git-assist
    set TMPFILE (mktemp)
    function cleanup --on-event fish_exit; rm -f $copilot_what_the_shell_TMPFILE; end
    if github-copilot-cli git-assist $argv --shellout $TMPFILE
        if test -e "$TMPFILE"
            set FIXED_CMD (cat $TMPFILE)
            eval $FIXED_CMD
        else
            echo "Apologies! Extracting command failed"
        end
    else
        return 1
    end
end

function pcmd
  ps -p $argv[1] -o pid,ppid,cmd
end

function wait_gpu
    set fetch_gpu $argv[1]
    set -e argv[1]

    set cmd (string join " " -- $argv) # so that fish can handle --args correctly
    echo -e "\e[32m[WAIT] $cmd\e[0m"
    sg $fetch_gpu -m peace --threshold 0.95
    echo -e "\e[32m[START] $cmd\e[0m"
    sleep 30
    gpukill $fetch_gpu

    eval $cmd

    echo -e "\e[32m[END] $cmd\e[0m"
    sg $fetch_gpu
end