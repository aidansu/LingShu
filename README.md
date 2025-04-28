# LingShu 大模型应用开发平台

这是一个基于 FastAPI 和 Langchain 的大模型应用开发平台，可以帮助开发者快速搭建基于大模型的应用服务，。

## 1. 快速入门 

### 1.1. 环境准备

1. 安装 uv
> https://docs.astral.sh/uv/getting-started/installation/

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. 执行以下命令克隆项目，并初始化虚拟环境

```shell
git clone https://github.com/aidansu/LingShu.git
cd LingShu
uv sync
source .venv/bin/activate
```

### 1.2. 启动服务

使用下列命令启动 FastAPI

```shell
uvicorn main:app
```

现在您可以访问 `http://localhost:8000/docs` 查看 API 文档，并开始使用服务了。

## 2. 架构设计
LingShu采用分层架构设计，结合高性能异步框架与现代开发工具链，为大型语言模型（LLM）提供灵活、可扩展的服务化能力。以下是核心架构模块和技术栈说明：

### 2.1. 技术栈
* 环境管理: Python 3.10+
* 依赖管理: UV 0.6.17
* 服务器: Uvicorn 0.34.2
* API 层: FastAPI 0.115.12
* 模型服务: LangChain 0.3.0

