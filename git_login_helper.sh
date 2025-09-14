#!/bin/bash

echo "=========================================="
echo "Git 重新登录助手"
echo "=========================================="

echo "当前Git配置:"
git config --global user.name
git config --global user.email
echo ""

echo "当前远程仓库:"
git remote -v
echo ""

echo "请按照以下步骤操作:"
echo ""
echo "1. 访问 https://github.com/settings/tokens"
echo "2. 登录 BorisGuo6 账户"
echo "3. 点击 'Generate new token (classic)'"
echo "4. 选择权限: repo (完整仓库访问)"
echo "5. 复制生成的token"
echo ""

echo "准备好token后，运行以下命令:"
echo "git push"
echo ""
echo "当提示输入用户名时: 输入 BorisGuo6"
echo "当提示输入密码时: 粘贴您的Personal Access Token"
echo ""

echo "=========================================="
