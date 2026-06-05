# Git 与 GitHub 上手引导（零基础 · 针对本项目）

> 面向"没实操过 git"的你。照着从上到下做一遍即可。命令行里 `$` 后面的才是你要敲的命令，`#` 后面是注释别敲。
> 你的项目目录：`/mnt/nvme3n1/gameTheory`。你的邮箱：`halvorsenbelair967@gmail.com`。

---

## 0. 三个概念先搞懂（30 秒）

- **Git**：本地的版本管理工具，记录你代码每次改动的"快照"。已装好（2.34.1）。
- **仓库 (repository / repo)**：一个被 git 管理的项目文件夹。
- **GitHub**：一个托管 git 仓库的网站。"上传到 GitHub" = 把本地仓库的快照同步到 GitHub 上。

工作流就三步循环：**改文件 → `add`(选中改动) → `commit`(存快照) → `push`(传到 GitHub)**。

---

## 1. 一次性配置（只做一次）

告诉 git 你是谁（每次 commit 会记名字）：

```bash
$ git config --global user.name "你的名字或GitHub用户名"
$ git config --global user.email "halvorsenbelair967@gmail.com"
$ git config --global init.defaultBranch main   # 默认主分支叫 main
```

验证：

```bash
$ git config --global --list
```

---

## 2. 在 GitHub 网站上建一个空仓库

1. 浏览器打开 https://github.com ，登录（没账号先注册，用上面那个邮箱）。
2. 右上角 **+** → **New repository**。
3. 填写：
   - **Repository name**：例如 `chinese-chess-ai`
   - **Description**（可选）：中国象棋博弈 AI 课程设计
   - 选 **Private**（课程作业建议先私有，答辩后想公开再改）
   - **不要**勾 "Add a README / .gitignore / license"（保持空仓库，避免和本地冲突）
4. 点 **Create repository**。
5. 建好后页面会给你一个仓库地址，形如：
   - HTTPS：`https://github.com/你的用户名/chinese-chess-ai.git`
   - 记下它，第 4 步要用。

---

## 3. 先建好 .gitignore（关键！别传大文件）

本项目会产生**几个 G 的网络权重、棋谱数据、自对弈样本、编译产物**——这些**绝对不要**传到 GitHub（仓库会爆、还会很慢，GitHub 单文件上限 100MB）。

在项目根目录建 `.gitignore` 文件，内容见本目录下我已附的 `gitignore模板.txt`，把它复制成 `.gitignore`：

```bash
$ cp /mnt/nvme3n1/gameTheory/docx/gitignore模板.txt /mnt/nvme3n1/gameTheory/.gitignore
```

> 规则：**代码、配置、报告、脚本** → 传；**权重(.pt/.ckpt)、数据集、日志、build/ 产物、__pycache__** → 不传。大模型权重要分享的话用 GitHub Releases 或网盘，不要进 git 历史。

---

## 4. 把本地项目变成仓库并首次上传

在项目根目录依次执行：

```bash
$ cd /mnt/nvme3n1/gameTheory

$ git init                       # 把当前文件夹初始化为 git 仓库
$ git add .                      # 选中所有未被 .gitignore 排除的文件
$ git status                     # 看一眼将要提交什么（确认没有 .pt/数据/build）
$ git commit -m "init: 课程设计方案与报告框架"   # 存第一个快照

# 关联到你在第2步建的 GitHub 仓库（换成你自己的地址）
$ git remote add origin https://github.com/你的用户名/chinese-chess-ai.git

$ git branch -M main             # 主分支命名为 main
$ git push -u origin main        # 首次推送（-u 记住关联，以后直接 git push）
```

**首次 push 会要求登录**。HTTPS 方式下，**密码不是你的 GitHub 登录密码，而是 Personal Access Token (PAT)**，见第 5 节。

---

## 5. 认证：生成 Personal Access Token (PAT)

GitHub 早已禁用密码推送，必须用 token：

1. GitHub 网页 → 右上头像 → **Settings** → 最底部 **Developer settings** → **Personal access tokens** → **Tokens (classic)** → **Generate new token (classic)**。
2. Note 填 `chinese-chess-ai`；Expiration 选 90 days；勾选权限 **`repo`**（整个 repo 勾上即可）。
3. 生成后**复制那串 token（只显示一次，离开页面就看不到了，先存到安全的地方）**。
4. 回到终端 `git push` 时：
   - Username：你的 GitHub 用户名
   - Password：**粘贴刚才的 token**（粘贴时终端不显示字符，正常）。

让它记住、不用每次输：

```bash
$ git config --global credential.helper store   # 之后输一次token就会被保存
```

> （可选，更省事）也可以装 GitHub CLI `gh`，用 `gh auth login` 浏览器一键登录，之后 push 免输 token。本机暂未装 `gh`，需要的话告诉我，我帮你装。

---

## 6. 日常工作流（以后每天就重复这几条）

```bash
$ cd /mnt/nvme3n1/gameTheory
$ git status                     # 看改了哪些文件
$ git add .                      # 选中改动（或 git add 某个文件）
$ git commit -m "feat: 完成C++规则引擎与perft验证"   # 存快照，消息写清楚做了啥
$ git push                       # 传到 GitHub
```

**commit message 习惯**（让历史可读，也是报告里"开发过程"的素材）：
- `feat: ...` 新功能　`fix: ...` 修 bug　`docs: ...` 文档　`refactor: ...` 重构　`exp: ...` 实验

建议**每完成一个 Gate 就 commit 一次并打 tag**，对应优化方案里的里程碑：

```bash
$ git tag v0.1-alphabeta-baseline   # 例如 D2 末保底线冻结
$ git push --tags
```

---

## 7. 常见问题速查

| 现象 | 原因 / 解决 |
|---|---|
| push 报 `403` / 认证失败 | 用了密码而非 token；重新用第 5 节的 PAT |
| `remote origin already exists` | 已加过远程；用 `git remote set-url origin 新地址` 改 |
| 不小心 `git add` 了大文件/权重 | `git rm --cached 文件名` 取消跟踪，再把它写进 `.gitignore`，重新 commit |
| 已经 commit 了大文件还没 push | `git reset --soft HEAD~1` 撤回上一次 commit（改动还在），修好 .gitignore 重来 |
| 看历史 | `git log --oneline` |
| push 被拒说远程有更新 | `git pull --rebase` 拉下来再 push（多人协作/网页改过时） |

---

## 8. 这个项目推荐的提交节奏（结合一周日程）

- **D1**：init 仓库 + 规则引擎 + perft → commit + tag `v0.1`
- **D2**：Alpha-Beta 保底线 → commit + tag `v0.2-baseline`（保命线冻结，重要！）
- **D3**：网络+预训练+自对弈闭环 → commit
- **D4–D6**：每跑通一块就 commit（数据/权重不进 git，用 .gitignore 挡住）
- **D7**：报告 + 最终权重说明 → commit + tag `v1.0-final`

> 这样到答辩时，`git log` 本身就是一条清晰的开发时间线，可以直接截图放进报告"工程实现/开发过程"。
