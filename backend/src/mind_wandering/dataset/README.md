# PAFE 数据准备

PAFE 原始视频未随本项目分发。获得作者授权的数据后，请整理一个 CSV：

```csv
subject_id,video_path,probe_time_sec,label
P001,data/raw/pafe/P001/session.mp4,40,focused
P001,data/raw/pafe/P001/session.mp4,80,not_focused
```

`video_path` 可为相对于 CSV 的路径或绝对路径。`probe_time_sec` 是探针在视频中的秒数；
`label` 使用 `focused`、`not_focused`，无法判断的 `skip` 会被丢弃。

未来三分类使用 `focused`、`deliberate_mw`、`spontaneous_mw`，运行准备和训练命令时
把 `--task binary` 改为 `--task three_class` 即可。
