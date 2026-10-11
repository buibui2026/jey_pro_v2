#!/usr/bin/env bash
# 统一 GitHub 推送入口（多会话/多分支共用）
#
# 解决两件事：
#   1. 本机 git 配了 http.https://github.com.proxy=127.0.0.1:7892，代理软件不常开——先直连，失败再走代理；
#   2. 令牌不进仓库文件、不进 git 历史——只从环境变量或 .git/github_token 读取。
#
# 用法（仓库内任意分支/任意会话）：
#   bash scripts/git_push.sh
# 令牌来源（优先级）：环境变量 GH_TOKEN > $(git rev-parse --git-dir)/github_token（单行纯令牌）
# 注意：令牌已在对话中暴露过，项目结束请到 GitHub 撤销重建。
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

TOKEN="${GH_TOKEN:-}"
GITDIR=$(git rev-parse --git-dir)
if [ -z "$TOKEN" ] && [ -f "$GITDIR/github_token" ]; then
  TOKEN=$(head -1 "$GITDIR/github_token" | tr -d '[:space:]')
fi
if [ -z "$TOKEN" ]; then
  echo "✗ 缺令牌：export GH_TOKEN=github_pat_xxx，或写入 $GITDIR/github_token（单行，.git 内不入库）" >&2
  exit 1
fi
AUTH="Authorization: Basic $(printf 'x-access-token:%s' "$TOKEN" | base64)"

echo "[1/2] 直连推送（绕过代理）…"
if git -c http.https://github.com.proxy= -c "http.https://github.com/.extraheader=$AUTH" push origin HEAD 2>&1; then
  echo "✓ 推送成功（直连）"; exit 0
fi
echo "[2/2] 直连失败，尝试本机代理 127.0.0.1:7892 …"
if git -c "http.https://github.com/.extraheader=$AUTH" push origin HEAD 2>&1; then
  echo "✓ 推送成功（代理）"; exit 0
fi
echo "✗ 两种方式均失败：检查网络，或把代理软件（7892 端口）打开后重试" >&2
exit 1
