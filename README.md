# 🔎 SearcherKit

> Build, evaluate, and train search agents without locking your research to a
> single model, corpus, tool protocol, or training framework.

SearcherKit is a modular runtime for search-intensive agents. It brings agent
rollouts, heterogeneous search sources, tool execution, model adapters,
benchmark recipes, evaluation, and training integration into one coherent
stack.

![SearcherKit](docs/images/searcherkit_1.png){width=80% fig-align="center"}

### ✨ Highlights

- 🧠 **One toolkit for the full search-agent lifecycle.** Build agent scaffolds, generate training rollouts, run evaluations, and serve interactive experiences in one coherent framework. Reuse the same components from early prototypes to production-oriented workflows. Spend less time connecting fragmented tools, more time improving agent capabilities.

- 🌐 **Connect to various search sources.** Connect web services, APIs, Elasticsearch indexes, private knowledge bases, and local files through a common source interface. Combine sources as needed without modifying agent or tool design. Build custom retrieval pipelines that can evolve with your data and use case.

- ⚡ **Developer-friendly and built for large-scale experiments.** Modular providers, parsers, sources, and tools allow local models, hosted APIs, and retrieval backends can be mixed and matched. Native asynchronous execution and layered concurrency controls keep batch workloads fast and resource-aware. Checkpoint recovery and per-sample traces make long-running experiments easier to resume, inspect, and reproduce.

- 🔥 **Training-ready, with SOTA results from reinforcement learning.** Use SearcherKit as a high-throughput rollout runtime for SFT and RL workflows. The same agent and tool configuration can power interactive sessions, offline evaluation, training-data generation, and RL rollouts. Our RL experiments achieve state-of-the-art results with an 8B model on public search benchmarks。

### 🚀 Quick Start

See the [Quick Start](docs/guide/index.qmd)
to start running your search agent.

### 📖 Documentation

- [Quick start](docs/guide/index.qmd)
- [Search Files](docs/guide/01-searching/01-search-files.qmd)
- [Search Webpages](docs/guide/01-searching/02-search-web.qmd)

- [Training overview](docs/guide/02-training/01-training-overview.qmd)

- [Project Architecture](docs/guide/03-customizing/01-project-architecture.qmd)

- [CLI reference](docs/guide/04-interface/01-use-the-cli-interface.qmd)
- [Interactive TUI](docs/guide/04-interface/02-use-the-interactive-tui.qmd)

### 🤝 Contributing

SearcherKit is actively evolving. We welcome issues & PRs on reproduction reports, new model parsers, source adapters, benchmark recipes, training integrations, bug reports, and
documentation improvements etc. 

If SearcherKit helps your research or makes your agent stack easier to reason
about, consider starring the repository. It helps more search-agent builders
find the project.

### 👤 The Team

SearcherKit is built by contributors from Department of Statistics and Data Science at SUSTech under the guidence of Assistant Professor [Wei Hongxin](https://hongxin001.github.io/). Main contributors & maintainers are [Li Hanyang](https://justinliii.github.io/), [Zhang Haotian](), [Yang Hanjie](), [Wang Shuoyuan](), [Lan Zijie]() and [Yu Zhengye]().

### 📜 License

SearcherKit is licensed under the [MIT License](LICENSE).
