# StarWave-docs

StarWave 2.0.0 中文使用手册草稿。使用 Sphinx、sphinx_rtd_theme 和 MyST Markdown，输出可独立托管的静态 HTML。

## 本地构建

文档构建使用 Python 3.12，不需要安装 StarWave、PyTorch、CUDA 或 GPU。

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell 改用：.venv\Scripts\Activate.ps1
python -m pip install -r docs/requirements.txt
python -m sphinx -n -W --keep-going -b html -d _build/doctrees docs _build/html
python tools/check_site.py _build/html
python tools/check_api_docs.py
python -m http.server 8000 --directory _build/html --bind 127.0.0.1
```

打开 http://127.0.0.1:8000 。严格构建把警告视为错误；`check_site.py` 检查站内链接、锚点和静态资源，并检查文档包中不应出现的内容。

`check_api_docs.py` 检查每个已文档化参数的类型、默认值、描述覆盖及返回契约，并解析 Python 代码片段。可附加 `--wheel` 和公开 2.0.0 wheel 的文件名，按 SHA-256 与 AST 校验真实签名；此检查不导入或执行 StarWave。wheel 不属于文档项目或发布产物。

## 托管配置

`.github/workflows/docs.yml` 在推送和 PR 上构建并保留 HTML artifact。main 分支推送或手动 workflow_dispatch 在检查通过后部署到 GitHub Pages；PR 只构建，不部署。发布前需要仓库管理员在 Settings → Pages 选择 GitHub Actions。文件存在不等于网站已发布。

`.readthedocs.yaml` 是备用配置，不需要用它才能构建或部署到 GitHub Pages。是否适用 Read the Docs Community 应另行确认，不承诺免费托管资格。

## 内容与状态

手册、原创合成示例与构建配置位于本项目；不包含传播实现、二进制库或外部数据。文档不通过 autodoc 导入 StarWave，不启用 viewcode。API 签名已核验；GPU 示例、数值误差、性能、DataParallel 与完整 FWI 收敛尚待实机测试。详见手册的“文档状态与参考来源”。

发布前请审阅内容、示例、文档语言和许可证安排。本项目不替软件或文档所有者新增许可证声明。
