# 🔎 SearcherKit

SearcherKit 是面向搜索智能体的模块化运行时。它将智能体执行、异构搜索源、工具执行、模型适配、评估和训练集成统一到一套连贯的技术栈中。

![SearcherKit](docs/images/searcherkit_1.png)

## 功能

- **构建并运行搜索智能体** – 只需定义一次智能体行为，即可将其用于原型开发、评估、交互式演示和大规模 rollout（轨迹）生成。
- **连接任意搜索源** – 通过统一接口接入 Web API、Elasticsearch、内部知识库或本地文件。无需修改智能体代码，即可自由组合各种数据源。
- **轻松扩展规模** – 异步执行、灵活的并发控制、检查点恢复和详细的轨迹记录，让批量任务既快速又易于调试。模型和后端也可轻松替换。
- **可直接用于训练** – 同一套智能体和工具配置可用于评估、离线分析以及与训练流水线集成，无需重构代码。

---

## 为什么选择 SearcherKit？

- 🧩 **一套代码，多种用途** – 无需再在研究环境与生产环境之间复制逻辑。为实验构建的内容可直接用于评估和部署。
- 🔌 **即插即用的搜索源** – 几分钟内即可添加或移除搜索后端。智能体无需关心数据来自哪里。
- ⚡ **快速可靠** – 原生异步、智能并发和自动检查点机制，让你无需时刻看守也能运行持续数周的实验。
- 🏆 **经过强化学习验证** – 我们已使用 SearcherKit 在真实搜索任务上训练出具有竞争力的 8B 模型。同一套流水线现已开放，可直接用于你自己的训练。

### 🚀 快速开始

阅读[快速开始](docs/guide/index.qmd)，开始运行你的搜索智能体。

### 📖 文档

- [快速开始](docs/guide/index.qmd)
- [搜索本地文件](docs/guide/01-searching/01-search-files.qmd)
- [搜索网页](docs/guide/01-searching/02-search-web.qmd)

- [训练概览](docs/guide/02-training/01-training-overview.qmd)

- [项目架构](docs/guide/03-customizing/01-project-architecture.qmd)

- [CLI 参考](docs/guide/04-interface/01-use-the-cli-interface.qmd)
- [交互式 TUI](docs/guide/04-interface/02-use-the-interactive-tui.qmd)

### 🤝 参与贡献

SearcherKit 正在持续开发中。欢迎提交 issue 和 PR，包括复现报告、新格式解析器、搜索源适配器、运行配方、训练集成、错误报告和文档改进等。

如果 SearcherKit 对你的研究有所帮助，欢迎 ⭐ 这个仓库让更多人看到。

### 👤 团队

SearcherKit 由南方科技大学和香港中文大学（深圳）的[魏鸿鑫](https://hongxin001.github.io/)助理教授和[荆炳义](https://sai.cuhk.edu.cn/en/teacher/162)教授团队开发。

主要贡献者和维护者包括 [Li Hanyang](https://justinliii.github.io/)、[Zhang Haotian]()、[Yang Hanjie]()、[Wang Shuoyuan]()、[Lan Zijie]() 和 [Yu Zhengye]()。

### 📜 许可证

SearcherKit 使用 [MIT 许可证](LICENSE)。
